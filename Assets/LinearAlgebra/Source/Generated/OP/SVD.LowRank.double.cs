#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

using Random = Unity.Mathematics.Random;

namespace LinearAlgebra
{
    public static partial class SVD {

        // Truncated SVD via Golub-Kahan-Lanczos (GKL) bidiagonalization with full reorthogonalization.
        // Computes the top-k singular triplets of A (m x n, m >= n) by building a p-step Lanczos
        // bidiagonalization (p = min(k + oversample, n)) on A directly (NOT on AᵀA — avoids κ²
        // accuracy loss), then solving the small p×p bidiagonal SVD exactly via svdThin. DGKS double
        // reorthogonalization (Daniel-Gragg-Kaufman-Stewart) is applied to BOTH u and v bases at
        // every step for numerical stability.
        //
        // Performance implementation notes:
        //   - UL is stored as p×m (Lanczos u-vectors as contiguous ROWS) and VL as (p+1)×n
        //     (Lanczos v-vectors as contiguous ROWS) so every GEMV (A·v and Aᵀ·u) hits unit-stride
        //     memory in both the matrix and the vector.
        //   - Matvecs route through UnsafeOP.matVecDot / vecMatDot (cache-coherent, Burst-vectorized).
        //   - DGKS reortho is expressed as matVecDot (to gather all dot-products in one sweep) + j
        //     axpy calls (unit-stride over UL/VL rows), eliminating strided column gathers.
        //   - All arithmetic uses native double precision; stability comes from the two-pass DGKS
        //     reorthogonalization, not from higher-precision accumulation.
        //
        // lowRankApprox still uses svdThin (full SVD + slice) internally — it stays EXACT (Eckart-Young).
        //
        // The workspace (doubleSvdTruncatedWorkspace) bundles all scratch; allocate ONCE via
        // Arena.doubleSvdTruncatedWorkspace(m, n, k, oversample) and reuse across same-shape calls.

        /// <summary>
        /// GKL truncated SVD: the top-k singular triplets of A (m x n, m >= n) via Golub-Kahan-Lanczos
        /// bidiagonalization with full reorthogonalization. Uk (m x k), Sk (length k, descending),
        /// Vk (n x k) are caller-allocated; 0 &lt;= k &lt;= n. A is NOT modified.
        ///
        /// <paramref name="oversample"/> extra Lanczos steps (p = min(k+oversample, n)) improve accuracy.
        /// <paramref name="seed"/> seeds the starting vector (0 → default seed, reproducible).
        /// <paramref name="converged"/> is false if the inner bidiagonal QR did not converge, or if the
        /// Krylov space was exhausted before k triplets could be formed (rank-deficient A); remaining Sk
        /// are set to 0, Uk/Vk columns zeroed. The residual |β_last·P[p-1,t]| / (σ₀+ε) is also checked
        /// against 8·√ε; if it exceeds this tolerance, converged is set false. Here P = BsvdWs.U (left
        /// singular vectors of B); the residual norm is |β_last·P[p-1,t]| from Aᵀ·x_t − σ_t·y_t = β_last·P[p-1,t]·v_{p+1}.
        /// <paramref name="ws"/> is the GKL scratch; size it with
        /// Arena.doubleSvdTruncatedWorkspace(m, n, k, oversample) using the SAME k and oversample.
        /// </summary>
        public static void svdTruncated(in doubleMxN A, ref doubleMxN Uk, ref doubleN Sk, ref doubleMxN Vk,
                                        int k, int oversample, uint seed, int maxIter,
                                        ref doubleSvdTruncatedWorkspace ws, out bool converged)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("svdTruncated: A must have m >= n (more rows than columns)");
            if (k < 0 || k > n)
                throw new ArgumentException("svdTruncated: k must be in [0, A.N_Cols]");
            if (oversample < 0)
                throw new ArgumentException("svdTruncated: oversample must be >= 0");
            if (Uk.M_Rows != m || Uk.N_Cols != k)
                throw new ArgumentException("svdTruncated: Uk must be m x k");
            if (Sk.N != k)
                throw new ArgumentException("svdTruncated: Sk must have length k");
            if (Vk.M_Rows != n || Vk.N_Cols != k)
                throw new ArgumentException("svdTruncated: Vk must be n x k");
            if (maxIter < 1)
                throw new ArgumentException("svdTruncated: maxIter must be >= 1");

            int p = math.min(k + oversample, n);
            RequireSvdTruncatedWorkspace(in ws, m, n, p, "svdTruncated");

            converged = true;

            if (n == 0 || k == 0)
                return;

            double scaleEps = Consts.doubleEpsilon;
            int pDone = 0;
            bool alphaBreakdown = false;

            unsafe
            {
                double* UL_ptr   = ws.UL.Data.Ptr;   // layout: p×m — row j = u_j (m-vector, contiguous)
                double* VL_ptr   = ws.VL.Data.Ptr;   // layout: (p+1)×n — row j = v_j (n-vector, contiguous)
                double* uBuf_ptr = ws.uBuf.Data.Ptr; // m-vector scratch (also used as coeff temp in v-reortho)
                double* vBuf_ptr = ws.vBuf.Data.Ptr; // n-vector scratch (also used as coeff temp in u-reortho)
                double* A_ptr    = A.Data.Ptr;        // m×n row-major

                // --- Zero-initialize UL (p×m) and VL ((p+1)×n) ---
                UnsafeUtility.MemClear(UL_ptr, (long)ws.UL.Data.Length * UnsafeUtility.SizeOf<double>());
                UnsafeUtility.MemClear(VL_ptr, (long)ws.VL.Data.Length * UnsafeUtility.SizeOf<double>());

                // --- Seed v_0: deterministic pseudo-random unit vector in R^n → stored as VL[0,:] ---
                var rng = new Random(seed == 0 ? 0x9E3779B1u : seed);
                double* v0 = VL_ptr;  // VL[0,:] at offset 0 (contiguous n-vector)
                for (int i = 0; i < n; i++)
                    v0[i] = (double)(rng.NextFloat() * 2f - 1f);
                double seedNorm2 = UnsafeOP.vecDot(v0, v0, n);
                if (seedNorm2 > (double)0)
                    UnsafeOP.scalMul(v0, n, (double)1 / math.sqrt(seedNorm2));
                else
                    v0[0] = (double)1;

                for (int j = 0; j < p; j++)
                {
                    // ----- uBuf = A * VL[j,:] -----
                    // matVecDot accumulates (+=); zero uBuf first. VL[j,:] is at VL_ptr + j*n (contiguous).
                    UnsafeUtility.MemClear(uBuf_ptr, (long)m * UnsafeUtility.SizeOf<double>());
                    UnsafeOP.matVecDot(A_ptr, VL_ptr + j * n, uBuf_ptr, m, n);

                    // Subtract beta_{j-1} * UL[j-1,:]  (skipped at j=0)
                    if (j > 0)
                        UnsafeOP.axpy(uBuf_ptr, UL_ptr + (j - 1) * m, -ws.beta[j - 1], m);

                    // DGKS double reorthogonalize uBuf against UL[0..j-1,:]
                    // Use vBuf[0..j-1] as coefficient temp (n >= p >= j, so always in-bounds).
                    // matVecDot(UL_first_j x m, uBuf, vBuf_coeffs, j, m) = UL_j * uBuf → j dot-products.
                    if (j > 0)
                    {
                        for (int pass = 0; pass < 2; pass++)
                        {
                            UnsafeUtility.MemClear(vBuf_ptr, (long)j * UnsafeUtility.SizeOf<double>());
                            UnsafeOP.matVecDot(UL_ptr, uBuf_ptr, vBuf_ptr, j, m);
                            for (int l = 0; l < j; l++)
                                UnsafeOP.axpy(uBuf_ptr, UL_ptr + l * m, -vBuf_ptr[l], m);
                        }
                    }

                    // alpha_j = ||uBuf||
                    ws.alpha[j] = math.sqrt(UnsafeOP.vecDot(uBuf_ptr, uBuf_ptr, m));

                    // Update scale estimate after first step
                    if (j == 0)
                        scaleEps = Consts.doubleEpsilon * math.max((double)1, ws.alpha[0]);

                    // Early stop: Krylov space exhausted (alpha ≈ 0)
                    if (ws.alpha[j] <= scaleEps)
                    {
                        alphaBreakdown = true;
                        pDone = j;
                        break;
                    }

                    // UL[j,:] = uBuf / alpha_j  (scale in-place, then copy to UL row j)
                    double invA = (double)1 / ws.alpha[j];
                    UnsafeOP.scalMul(uBuf_ptr, m, invA);
                    UnsafeUtility.MemCpy(UL_ptr + j * m, uBuf_ptr, (long)m * UnsafeUtility.SizeOf<double>());

                    // ----- vBuf = Aᵀ * UL[j,:] - alpha_j * VL[j,:] -----
                    // vecMatDot zeros vBuf internally then accumulates. UL[j,:] is at UL_ptr + j*m (contiguous).
                    UnsafeOP.vecMatDot(UL_ptr + j * m, A_ptr, vBuf_ptr, m, n);
                    UnsafeOP.axpy(vBuf_ptr, VL_ptr + j * n, -ws.alpha[j], n);

                    // DGKS double reorthogonalize vBuf against VL[0..j,:]
                    // Use uBuf[0..j] as coefficient temp (m >= n >= p >= j+1, so always in-bounds).
                    // matVecDot(VL_first_j+1 x n, vBuf, uBuf_coeffs, j+1, n) = VL_{j+1} * vBuf → j+1 dots.
                    for (int pass = 0; pass < 2; pass++)
                    {
                        UnsafeUtility.MemClear(uBuf_ptr, (long)(j + 1) * UnsafeUtility.SizeOf<double>());
                        UnsafeOP.matVecDot(VL_ptr, vBuf_ptr, uBuf_ptr, j + 1, n);
                        for (int l = 0; l <= j; l++)
                            UnsafeOP.axpy(vBuf_ptr, VL_ptr + l * n, -uBuf_ptr[l], n);
                    }

                    // beta_j = ||vBuf||
                    ws.beta[j] = math.sqrt(UnsafeOP.vecDot(vBuf_ptr, vBuf_ptr, n));

                    pDone = j + 1;

                    // Early stop: invariant subspace reached (beta ≈ 0)
                    if (ws.beta[j] <= scaleEps)
                        break;

                    // VL[j+1,:] = vBuf / beta_j  (scale in-place, then copy to VL row j+1)
                    double invB = (double)1 / ws.beta[j];
                    UnsafeOP.scalMul(vBuf_ptr, n, invB);
                    UnsafeUtility.MemCpy(VL_ptr + (j + 1) * n, vBuf_ptr, (long)n * UnsafeUtility.SizeOf<double>());
                }
            }

            // --- Fill Lanczos bidiagonal d/e at full size p (zero-padded beyond pDone) ---
            // Running the inner SVD on the full p×p zero-padded bidiagonal exactly reproduces the
            // previous behaviour (where ws.B was p×p with zeros beyond pDone). The trailing zero
            // singular values sort to the end and are never read by the map-back code.
            for (int j = 0; j < p; j++)        ws.dB[j] = (j < pDone) ? ws.alpha[j] : (double)0;
            ws.eB[0] = (double)0;
            for (int j = 1; j < p; j++)        ws.eB[j] = (j < pDone) ? ws.beta[j - 1] : (double)0;

            // --- Inner SVD of the tiny p×p bidiagonal: skip Householder re-bidiagonalization ---
            // BsvdWs.U receives P (p x p), BsvdWs.S sigma (sorted desc), BsvdWs.V receives Q (p x p).
            // dB/eB are destroyed by bidiagonalSvdFromDE (filled again on every call above, so fine).
            if (!bidiagonalSvdFromDE(ref ws.dB, ref ws.eB, ref ws.UtB, ref ws.VtB,
                                     ref ws.BsvdWs.U, ref ws.BsvdWs.S, ref ws.BsvdWs.V, p, maxIter))
            {
                converged = false;
                return;
            }

            // --- Map back to A's singular triplets ---
            int kOut = math.min(k, pDone);

            // --- Residual-based convergence check ---
            // res_t = |β_last · P[pDone-1, t]|  where P = BsvdWs.U (LEFT singular vectors of B).
            {
                double betaLast = alphaBreakdown ? (double)0
                                : (pDone > 0)    ? ws.beta[pDone - 1]
                                :                  (double)0;
                double sigma0 = (ws.BsvdWs.S.N > 0) ? ws.BsvdWs.S[0] : (double)0;
                double resTol = (double)8 * Consts.doubleSqrtEps;
                double maxRelRes = (double)0;
                for (int t = 0; t < kOut; t++)
                {
                    double res = math.abs(betaLast * ws.BsvdWs.U[pDone - 1, t]);
                    double relRes = res / (sigma0 + Consts.doubleEpsilon);
                    if (relRes > maxRelRes) maxRelRes = relRes;
                }
                converged = (maxRelRes < resTol);
            }

            // Fill top kOut triplets.
            // Uk[:,t] = UL^T · P[:,t] = sum_l P[l,t] * UL[l,:]   (UL is p×m, P is p×p)
            // Vk[:,t] = VL^T · Q[:,t] = sum_l Q[l,t] * VL[l,:]   (VL is (p+1)×n, Q is p×p)
            // Use uBuf / vBuf as accumulators (they are free; main loop is done).
            unsafe
            {
                double* UL_ptr   = ws.UL.Data.Ptr;
                double* VL_ptr   = ws.VL.Data.Ptr;
                double* uBuf_ptr = ws.uBuf.Data.Ptr;
                double* vBuf_ptr = ws.vBuf.Data.Ptr;

                for (int t = 0; t < kOut; t++)
                {
                    Sk[t] = ws.BsvdWs.S[t];

                    // uBuf = sum_l P[l,t] * UL[l,:]
                    UnsafeUtility.MemClear(uBuf_ptr, (long)m * UnsafeUtility.SizeOf<double>());
                    for (int l = 0; l < pDone; l++)
                        UnsafeOP.axpy(uBuf_ptr, UL_ptr + l * m, ws.BsvdWs.U[l, t], m);
                    for (int i = 0; i < m; i++) Uk[i, t] = uBuf_ptr[i];

                    // vBuf = sum_l Q[l,t] * VL[l,:]
                    UnsafeUtility.MemClear(vBuf_ptr, (long)n * UnsafeUtility.SizeOf<double>());
                    for (int l = 0; l < pDone; l++)
                        UnsafeOP.axpy(vBuf_ptr, VL_ptr + l * n, ws.BsvdWs.V[l, t], n);
                    for (int i = 0; i < n; i++) Vk[i, t] = vBuf_ptr[i];
                }
            }

            // Zero remaining triplets if early stop produced fewer than k
            if (kOut < k)
            {
                converged = false;
                for (int t = kOut; t < k; t++)
                {
                    Sk[t] = (double)0;
                    for (int i = 0; i < m; i++) Uk[i, t] = (double)0;
                    for (int i = 0; i < n; i++) Vk[i, t] = (double)0;
                }
            }
        }

        /// <summary>svdTruncated (ref workspace) with default maxIter (75).</summary>
        public static void svdTruncated(in doubleMxN A, ref doubleMxN Uk, ref doubleN Sk, ref doubleMxN Vk,
                                        int k, int oversample, uint seed,
                                        ref doubleSvdTruncatedWorkspace ws, out bool converged)
            => svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, 75, ref ws, out converged);

        /// <summary>svdTruncated (ref workspace) with default seed and maxIter (75).</summary>
        public static void svdTruncated(in doubleMxN A, ref doubleMxN Uk, ref doubleN Sk, ref doubleMxN Vk,
                                        int k, int oversample,
                                        ref doubleSvdTruncatedWorkspace ws, out bool converged)
            => svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0x9E3779B1u, 75, ref ws, out converged);

        /// <summary>
        /// svdTruncated (ref workspace) with generous default Krylov width p = min(n, max(2k, k+12)).
        /// Pass a workspace from Arena.doubleSvdTruncatedWorkspace(m, n, k) (no oversample overload)
        /// which uses the same generous formula.
        /// </summary>
        public static void svdTruncated(in doubleMxN A, ref doubleMxN Uk, ref doubleN Sk, ref doubleMxN Vk,
                                        int k, ref doubleSvdTruncatedWorkspace ws, out bool converged)
            => svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, math.max(k, 12), 0x9E3779B1u, 75, ref ws, out converged);

        /// <summary>
        /// svdTruncated allocating all scratch from A's arena (explicit oversample/seed/maxIter).
        /// See the ref-workspace overload for semantics.
        /// </summary>
        public static void svdTruncated(in doubleMxN A, ref doubleMxN Uk, ref doubleN Sk, ref doubleMxN Vk,
                                        int k, int oversample, uint seed, int maxIter, out bool converged)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            if (k < 0 || k > n) throw new ArgumentException("svdTruncated: k must be in [0, A.N_Cols]");
            if (oversample < 0) throw new ArgumentException("svdTruncated: oversample must be >= 0");
            int p = math.min(k + oversample, n);
            var ws = new doubleSvdTruncatedWorkspace
            {
                UL     = A.tempdoubleMat(p, m),
                VL     = A.tempdoubleMat(p + 1, n),
                dB     = A.tempdoubleVec(p),
                eB     = A.tempdoubleVec(p),
                UtB    = A.tempdoubleMat(p, p),
                VtB    = A.tempdoubleMat(p, p),
                BsvdWs = new doubleSvdFullWorkspace
                {
                    U = A.tempdoubleMat(p, p),
                    S = A.tempdoubleVec(p),
                    V = A.tempdoubleMat(p, p)
                },
                uBuf  = A.tempdoubleVec(m),
                vBuf  = A.tempdoubleVec(n),
                alpha = A.tempdoubleVec(p),
                beta  = A.tempdoubleVec(p)
            };
            svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, maxIter, ref ws, out converged);
        }

        /// <summary>
        /// svdTruncated (allocating) with generous default Krylov width p = min(n, max(2k, k+12)),
        /// default seed (0x9E3779B1u), and default maxIter (75).
        /// </summary>
        public static void svdTruncated(in doubleMxN A, ref doubleMxN Uk, ref doubleN Sk, ref doubleMxN Vk,
                                        int k, out bool converged)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            if (k < 0 || k > n) throw new ArgumentException("svdTruncated: k must be in [0, A.N_Cols]");
            int p = math.min(n, math.max(2 * k, k + 12));
            var ws = new doubleSvdTruncatedWorkspace
            {
                UL     = A.tempdoubleMat(p, m),
                VL     = A.tempdoubleMat(p + 1, n),
                dB     = A.tempdoubleVec(p),
                eB     = A.tempdoubleVec(p),
                UtB    = A.tempdoubleMat(p, p),
                VtB    = A.tempdoubleMat(p, p),
                BsvdWs = new doubleSvdFullWorkspace
                {
                    U = A.tempdoubleMat(p, p),
                    S = A.tempdoubleVec(p),
                    V = A.tempdoubleMat(p, p)
                },
                uBuf  = A.tempdoubleVec(m),
                vBuf  = A.tempdoubleVec(n),
                alpha = A.tempdoubleVec(p),
                beta  = A.tempdoubleVec(p)
            };
            // Use oversample = max(k, 12) which gives p = min(k + max(k,12), n) = min(max(2k,k+12), n)
            svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, math.max(k, 12), 0x9E3779B1u, 75, ref ws, out converged);
        }

        /// <summary>
        /// Best rank-k approximation of A (m x n, m >= n) written into Ak (m x n, caller-allocated):
        /// Ak = Σ_{t&lt;k} σ_t u_t v_tᵀ = Uk diag(Sk) Vkᵀ. This is the matrix that minimizes
        /// ||A - Ak|| over all rank-k matrices (Eckart-Young); the Frobenius error is sqrt(Σ_{i&gt;=k} σ_i²).
        /// Uses the FULL Golub-Kahan SVD (svdThin) internally — EXACT, not approximate. 0 &lt;= k &lt;= n.
        /// A is NOT modified. <paramref name="converged"/> is the SVD's flag (when false Ak is undefined).
        /// <paramref name="ws"/> is full-SVD scratch reused across calls; size it with
        /// Arena.doubleSvdFullWorkspace(m, n).
        /// </summary>
        public static void lowRankApprox(in doubleMxN A, ref doubleMxN Ak, int k,
                                         ref doubleSvdFullWorkspace ws, out bool converged, int maxIter)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("lowRankApprox: A must have m >= n (more rows than columns)");
            if (k < 0 || k > n)
                throw new ArgumentException("lowRankApprox: k must be in [0, A.N_Cols]");
            if (Ak.M_Rows != m || Ak.N_Cols != n)
                throw new ArgumentException("lowRankApprox: Ak must be m x n");
            if (maxIter < 1)
                throw new ArgumentException("lowRankApprox: maxIter must be >= 1");
            RequireSvdFullWorkspace(in ws, m, n, "lowRankApprox");

            converged = true;

            // Ak starts at zero (rank 0); also the correct result for k == 0 / empty A.
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    Ak[i, j] = (double)0;

            if (n == 0 || k == 0)
                return;

            converged = svdThin(in A, ref ws.U, ref ws.S, ref ws.V, maxIter);
            if (!converged)
                return;

            // Ak += σ_t · u_t v_tᵀ for t < k (rank-1 accumulation; inner row update is unit-stride
            // in the Ak row, V column gathered per t).
            for (int t = 0; t < k; t++)
            {
                double s = ws.S[t];
                for (int i = 0; i < m; i++)
                {
                    double us = ws.U[i, t] * s;
                    for (int j = 0; j < n; j++)
                        Ak[i, j] += us * ws.V[j, t];
                }
            }
        }

        /// <summary>lowRankApprox (ref workspace) with default maxIter (75).</summary>
        public static void lowRankApprox(in doubleMxN A, ref doubleMxN Ak, int k,
                                         ref doubleSvdFullWorkspace ws, out bool converged)
            => lowRankApprox(in A, ref Ak, k, ref ws, out converged, 75);

        /// <summary>
        /// lowRankApprox allocating its full-SVD scratch from A's arena.
        /// See the ref-workspace overload for semantics.
        /// </summary>
        public static void lowRankApprox(in doubleMxN A, ref doubleMxN Ak, int k, out bool converged, int maxIter)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            var ws = new doubleSvdFullWorkspace
            {
                U = A.tempdoubleMat(m, n),
                S = A.tempdoubleVec(n),
                V = A.tempdoubleMat(n, n)
            };
            lowRankApprox(in A, ref Ak, k, ref ws, out converged, maxIter);
        }

        /// <summary>lowRankApprox (allocating) with default maxIter (75).</summary>
        public static void lowRankApprox(in doubleMxN A, ref doubleMxN Ak, int k, out bool converged)
            => lowRankApprox(in A, ref Ak, k, out converged, 75);
    }
}
