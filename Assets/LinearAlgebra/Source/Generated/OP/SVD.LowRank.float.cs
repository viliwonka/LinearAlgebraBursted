#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
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
        // every step for numerical stability. Matvecs and the final map-back use double-precision
        // accumulation for accuracy in the float variant.
        //
        // lowRankApprox still uses svdThin (full SVD + slice) internally — it stays EXACT (Eckart-Young).
        //
        // The workspace (floatSvdTruncatedWorkspace) bundles all scratch; allocate ONCE via
        // Arena.floatSvdTruncatedWorkspace(m, n, k, oversample) and reuse across same-shape calls.

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
        /// Arena.floatSvdTruncatedWorkspace(m, n, k, oversample) using the SAME k and oversample.
        /// </summary>
        public static void svdTruncated(in floatMxN A, ref floatMxN Uk, ref floatN Sk, ref floatMxN Vk,
                                        int k, int oversample, uint seed, int maxIter,
                                        ref floatSvdTruncatedWorkspace ws, out bool converged)
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

            // --- Zero-initialize UL, VL, and B so reuse is safe (early-stop leaves tails zeroed) ---
            for (int i = 0; i < m; i++)
                for (int j = 0; j < p; j++)
                    ws.UL[i, j] = (float)0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j <= p; j++)
                    ws.VL[i, j] = (float)0;
            for (int i = 0; i < p; i++)
                for (int j = 0; j < p; j++)
                    ws.B[i, j] = (float)0;

            // --- Seed v1: deterministic pseudo-random unit vector in R^n ---
            var rng = new Random(seed == 0 ? 0x9E3779B1u : seed);
            {
                double norm2 = 0;
                for (int i = 0; i < n; i++)
                {
                    float val = (float)(rng.NextFloat() * 2f - 1f);
                    ws.VL[i, 0] = val;
                    norm2 += (double)val * (double)val;
                }
                if (norm2 > 0)
                {
                    double invNorm = 1.0 / math.sqrt(norm2);
                    for (int i = 0; i < n; i++)
                        ws.VL[i, 0] = (float)((double)ws.VL[i, 0] * invNorm);
                }
                else
                {
                    // Degenerate: use e1
                    ws.VL[0, 0] = (float)1;
                }
            }

            float scaleEps = Consts.floatEpsilon; // updated after first matvec
            int pDone = 0;
            bool alphaBreakdown = false; // true when Krylov exhausted via alpha ≈ 0 (not beta ≈ 0)

            for (int j = 0; j < p; j++)
            {
                // ----- u = A * VL[:,j] − beta_{j-1} * UL[:,j-1] -----
                // A·v_j with double accumulation
                for (int i = 0; i < m; i++)
                {
                    double acc = 0;
                    for (int l = 0; l < n; l++)
                        acc += (double)A[i, l] * (double)ws.VL[l, j];
                    ws.uBuf[i] = (float)acc;
                }
                // Subtract beta_{j-1} * u_{j-1}  (skipped at j=0: beta_0 = 0)
                if (j > 0)
                {
                    float bPrev = ws.beta[j - 1];
                    for (int i = 0; i < m; i++)
                        ws.uBuf[i] -= bPrev * ws.UL[i, j - 1];
                }

                // DGKS double reorthogonalize uBuf against UL[:,0..j-1]
                for (int pass = 0; pass < 2; pass++)
                {
                    for (int l = 0; l < j; l++)
                    {
                        double dot = 0;
                        for (int i = 0; i < m; i++)
                            dot += (double)ws.uBuf[i] * (double)ws.UL[i, l];
                        float dotF = (float)dot;
                        for (int i = 0; i < m; i++)
                            ws.uBuf[i] -= dotF * ws.UL[i, l];
                    }
                }

                // Compute alpha_j = ||uBuf||
                {
                    double norm2 = 0;
                    for (int i = 0; i < m; i++)
                        norm2 += (double)ws.uBuf[i] * (double)ws.uBuf[i];
                    ws.alpha[j] = (float)math.sqrt(norm2);
                }

                // Update scale estimate after first step
                if (j == 0)
                    scaleEps = Consts.floatEpsilon * math.max((float)1, ws.alpha[0]);

                // Early stop: Krylov space exhausted (alpha ≈ 0 → stop at j steps done so far).
                // The j produced triplets are EXACT: beta_j (which would come after u[j]) is ~0
                // because A*v[j] lies in span(U_j). We track this separately so the residual check
                // below does NOT misuse beta[j-1] (the previous step's coupling) as the residual.
                if (ws.alpha[j] <= scaleEps)
                {
                    alphaBreakdown = true;
                    pDone = j;
                    break;
                }

                // UL[:,j] = uBuf / alpha_j
                {
                    float invA = (float)1 / ws.alpha[j];
                    for (int i = 0; i < m; i++)
                        ws.UL[i, j] = ws.uBuf[i] * invA;
                }

                // ----- w = A^T * UL[:,j] − alpha_j * VL[:,j] -----
                // A^T · u_j with double accumulation
                for (int i = 0; i < n; i++)
                {
                    double acc = 0;
                    for (int l = 0; l < m; l++)
                        acc += (double)A[l, i] * (double)ws.UL[l, j];
                    ws.vBuf[i] = (float)acc;
                }
                // Subtract alpha_j * v_j
                {
                    float aJ = ws.alpha[j];
                    for (int i = 0; i < n; i++)
                        ws.vBuf[i] -= aJ * ws.VL[i, j];
                }

                // DGKS double reorthogonalize vBuf against VL[:,0..j]
                for (int pass = 0; pass < 2; pass++)
                {
                    for (int l = 0; l <= j; l++)
                    {
                        double dot = 0;
                        for (int i = 0; i < n; i++)
                            dot += (double)ws.vBuf[i] * (double)ws.VL[i, l];
                        float dotF = (float)dot;
                        for (int i = 0; i < n; i++)
                            ws.vBuf[i] -= dotF * ws.VL[i, l];
                    }
                }

                // beta_j = ||vBuf||
                {
                    double norm2 = 0;
                    for (int i = 0; i < n; i++)
                        norm2 += (double)ws.vBuf[i] * (double)ws.vBuf[i];
                    ws.beta[j] = (float)math.sqrt(norm2);
                }

                pDone = j + 1;

                // Early stop: beta ≈ 0 → invariant subspace reached
                if (ws.beta[j] <= scaleEps)
                    break;

                // VL[:,j+1] = vBuf / beta_j  (always safe: VL is n x (p+1))
                {
                    float invB = (float)1 / ws.beta[j];
                    for (int i = 0; i < n; i++)
                        ws.VL[i, j + 1] = ws.vBuf[i] * invB;
                }
            }

            // --- Form upper-bidiagonal B (p x p): already zero-initialised; fill converged part ---
            for (int j = 0; j < pDone; j++)
                ws.B[j, j] = ws.alpha[j];
            for (int j = 0; j < pDone - 1; j++)
                ws.B[j, j + 1] = ws.beta[j];

            // --- Inner SVD of the tiny p x p bidiagonal via svdThin ---
            // BsvdWs.U receives P (p x p), BsvdWs.S sigma (sorted desc), BsvdWs.V receives Q (p x p)
            if (!svdThin(in ws.B, ref ws.BsvdWs.U, ref ws.BsvdWs.S, ref ws.BsvdWs.V, maxIter))
            {
                converged = false;
                return;
            }

            // --- Map back to A's singular triplets ---
            int kOut = math.min(k, pDone);

            // --- Residual-based convergence check ---
            // res_t = |β_last · P[pDone-1, t]|  where P = BsvdWs.U (LEFT singular vectors of B).
            // Derivation: Aᵀ·x_t − σ_t·y_t = β_last·P[pDone-1,t]·v_{pDone+1}, so the residual
            // norm is |β_last·P[pDone-1,t]|.  β_last = ws.beta[pDone-1] is the Lanczos beta that
            // drives v_{pDone+1}; for p=n (full Krylov) or after beta-breakdown, β_last ≈ 0.
            // FIX 1: must index BsvdWs.U (not .V which are the RIGHT singular vectors of B).
            // FIX 2: after alpha-breakdown the Krylov space is exhausted so β_last = 0 (exact
            // triplets); ws.beta[pDone-1] is the previous step's coupling, NOT the residual.
            {
                float betaLast = alphaBreakdown ? (float)0
                                : (pDone > 0)    ? ws.beta[pDone - 1]
                                :                  (float)0;
                float sigma0 = (ws.BsvdWs.S.N > 0) ? ws.BsvdWs.S[0] : (float)0;
                float resTol = (float)8 * Consts.floatSqrtEps;
                float maxRelRes = (float)0;
                for (int t = 0; t < kOut; t++)
                {
                    float res = math.abs(betaLast * ws.BsvdWs.U[pDone - 1, t]); // FIX 1: U not V
                    float relRes = res / (sigma0 + Consts.floatEpsilon);
                    if (relRes > maxRelRes) maxRelRes = relRes;
                }
                converged = (maxRelRes < resTol);
            }

            // Fill top kOut triplets
            for (int t = 0; t < kOut; t++)
            {
                Sk[t] = ws.BsvdWs.S[t];

                // Uk[:,t] = UL * P[:,t]  (m-vector, double-precision accumulation)
                for (int i = 0; i < m; i++)
                {
                    double acc = 0;
                    for (int l = 0; l < p; l++)
                        acc += (double)ws.UL[i, l] * (double)ws.BsvdWs.U[l, t];
                    Uk[i, t] = (float)acc;
                }

                // Vk[:,t] = VL[:,0..p-1] * Q[:,t]  (n-vector, double-precision accumulation)
                for (int i = 0; i < n; i++)
                {
                    double acc = 0;
                    for (int l = 0; l < p; l++)
                        acc += (double)ws.VL[i, l] * (double)ws.BsvdWs.V[l, t];
                    Vk[i, t] = (float)acc;
                }
            }

            // Zero remaining triplets if early stop produced fewer than k
            if (kOut < k)
            {
                converged = false;
                for (int t = kOut; t < k; t++)
                {
                    Sk[t] = (float)0;
                    for (int i = 0; i < m; i++) Uk[i, t] = (float)0;
                    for (int i = 0; i < n; i++) Vk[i, t] = (float)0;
                }
            }
        }

        /// <summary>svdTruncated (ref workspace) with default maxIter (75).</summary>
        public static void svdTruncated(in floatMxN A, ref floatMxN Uk, ref floatN Sk, ref floatMxN Vk,
                                        int k, int oversample, uint seed,
                                        ref floatSvdTruncatedWorkspace ws, out bool converged)
            => svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, 75, ref ws, out converged);

        /// <summary>svdTruncated (ref workspace) with default seed and maxIter (75).</summary>
        public static void svdTruncated(in floatMxN A, ref floatMxN Uk, ref floatN Sk, ref floatMxN Vk,
                                        int k, int oversample,
                                        ref floatSvdTruncatedWorkspace ws, out bool converged)
            => svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0x9E3779B1u, 75, ref ws, out converged);

        /// <summary>
        /// svdTruncated (ref workspace) with generous default Krylov width p = min(n, max(2k, k+12)).
        /// Pass a workspace from Arena.floatSvdTruncatedWorkspace(m, n, k) (no oversample overload)
        /// which uses the same generous formula.
        /// </summary>
        public static void svdTruncated(in floatMxN A, ref floatMxN Uk, ref floatN Sk, ref floatMxN Vk,
                                        int k, ref floatSvdTruncatedWorkspace ws, out bool converged)
            => svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, math.max(k, 12), 0x9E3779B1u, 75, ref ws, out converged);

        /// <summary>
        /// svdTruncated allocating all scratch from A's arena (explicit oversample/seed/maxIter).
        /// See the ref-workspace overload for semantics.
        /// </summary>
        public static void svdTruncated(in floatMxN A, ref floatMxN Uk, ref floatN Sk, ref floatMxN Vk,
                                        int k, int oversample, uint seed, int maxIter, out bool converged)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            if (k < 0 || k > n) throw new ArgumentException("svdTruncated: k must be in [0, A.N_Cols]");
            if (oversample < 0) throw new ArgumentException("svdTruncated: oversample must be >= 0");
            int p = math.min(k + oversample, n);
            var ws = new floatSvdTruncatedWorkspace
            {
                UL     = A.tempfloatMat(m, p),
                VL     = A.tempfloatMat(n, p + 1),
                B      = A.tempfloatMat(p, p),
                BsvdWs = new floatSvdFullWorkspace
                {
                    U = A.tempfloatMat(p, p),
                    S = A.tempfloatVec(p),
                    V = A.tempfloatMat(p, p)
                },
                uBuf  = A.tempfloatVec(m),
                vBuf  = A.tempfloatVec(n),
                alpha = A.tempfloatVec(p),
                beta  = A.tempfloatVec(p)
            };
            svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, maxIter, ref ws, out converged);
        }

        /// <summary>
        /// svdTruncated (allocating) with generous default Krylov width p = min(n, max(2k, k+12)),
        /// default seed (0x9E3779B1u), and default maxIter (75).
        /// </summary>
        public static void svdTruncated(in floatMxN A, ref floatMxN Uk, ref floatN Sk, ref floatMxN Vk,
                                        int k, out bool converged)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            if (k < 0 || k > n) throw new ArgumentException("svdTruncated: k must be in [0, A.N_Cols]");
            int p = math.min(n, math.max(2 * k, k + 12));
            var ws = new floatSvdTruncatedWorkspace
            {
                UL     = A.tempfloatMat(m, p),
                VL     = A.tempfloatMat(n, p + 1),
                B      = A.tempfloatMat(p, p),
                BsvdWs = new floatSvdFullWorkspace
                {
                    U = A.tempfloatMat(p, p),
                    S = A.tempfloatVec(p),
                    V = A.tempfloatMat(p, p)
                },
                uBuf  = A.tempfloatVec(m),
                vBuf  = A.tempfloatVec(n),
                alpha = A.tempfloatVec(p),
                beta  = A.tempfloatVec(p)
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
        /// Arena.floatSvdFullWorkspace(m, n).
        /// </summary>
        public static void lowRankApprox(in floatMxN A, ref floatMxN Ak, int k,
                                         ref floatSvdFullWorkspace ws, out bool converged, int maxIter)
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
                    Ak[i, j] = (float)0;

            if (n == 0 || k == 0)
                return;

            converged = svdThin(in A, ref ws.U, ref ws.S, ref ws.V, maxIter);
            if (!converged)
                return;

            // Ak += σ_t · u_t v_tᵀ for t < k (rank-1 accumulation; inner row update is unit-stride
            // in the Ak row, V column gathered per t).
            for (int t = 0; t < k; t++)
            {
                float s = ws.S[t];
                for (int i = 0; i < m; i++)
                {
                    float us = ws.U[i, t] * s;
                    for (int j = 0; j < n; j++)
                        Ak[i, j] += us * ws.V[j, t];
                }
            }
        }

        /// <summary>lowRankApprox (ref workspace) with default maxIter (75).</summary>
        public static void lowRankApprox(in floatMxN A, ref floatMxN Ak, int k,
                                         ref floatSvdFullWorkspace ws, out bool converged)
            => lowRankApprox(in A, ref Ak, k, ref ws, out converged, 75);

        /// <summary>
        /// lowRankApprox allocating its full-SVD scratch from A's arena.
        /// See the ref-workspace overload for semantics.
        /// </summary>
        public static void lowRankApprox(in floatMxN A, ref floatMxN Ak, int k, out bool converged, int maxIter)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            var ws = new floatSvdFullWorkspace
            {
                U = A.tempfloatMat(m, n),
                S = A.tempfloatVec(n),
                V = A.tempfloatMat(n, n)
            };
            lowRankApprox(in A, ref Ak, k, ref ws, out converged, maxIter);
        }

        /// <summary>lowRankApprox (allocating) with default maxIter (75).</summary>
        public static void lowRankApprox(in floatMxN A, ref floatMxN Ak, int k, out bool converged)
            => lowRankApprox(in A, ref Ak, k, out converged, 75);
    }
}
