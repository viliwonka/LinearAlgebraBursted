#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Internal;

using Random = Unity.Mathematics.Random;

namespace LinearAlgebra
{
    public static partial class SVD {

        // Truncated SVD via Golub-Kahan-Lanczos (GKL) bidiagonalization.
        // Computes the top-k singular triplets of A (m x n, m >= n) by building a p-step Lanczos
        // bidiagonalization (p = min(k + oversample, n)) on A directly (NOT on AᵀA — avoids κ²
        // accuracy loss), then solving the small p×p bidiagonal SVD exactly via svdThin.
        //
        // Two reorthogonalization strategies are available via the `partialReorth` toggle:
        //   false — DGKS double reorthogonalization (Daniel-Gragg-Kaufman-Stewart) applied to BOTH
        //           u and v bases at EVERY step (full reorth). Maximum stability, O(p²(m+n)) cost.
        //   true  — Partial reorthogonalization via the ω-recurrence (Larsen/PROPACK lanbpro):
        //           maintain scalar μ/ν estimates of orthogonality loss; only trigger a full DGKS
        //           sweep when the estimate exceeds δ = sqrt(ε/p). Cuts cost at large p.
        //           Extended Local Reorthogonalization (ELR) is always applied.
        //
        // Performance implementation notes:
        //   - UL is stored as p×m (Lanczos u-vectors as contiguous ROWS) and VL as (p+1)×n
        //     (Lanczos v-vectors as contiguous ROWS) so every GEMV (A·v and Aᵀ·u) hits unit-stride
        //     memory in both the matrix and the vector.
        //   - Matvecs route through Unsafe_OP.matVecDot / vecMatDot (cache-coherent, Burst-vectorized).
        //   - DGKS reortho is expressed as matVecDot (to gather all dot-products in one sweep) + j
        //     axpy calls (unit-stride over UL/VL rows), eliminating strided column gathers.
        //   - All arithmetic uses native fProxy precision; stability comes from reorthogonalization,
        //     not from higher-precision accumulation.
        //
        // lowRankApprox uses svdThin instead (full SVD + slice — see its own doc). Scratch layout:
        // see fProxySVDTruncatedCache.

        /// <summary>
        /// GKL truncated SVD: the top-k singular triplets of A (m x n, m >= n) via Golub-Kahan-Lanczos
        /// bidiagonalization. Uk (m x k), Sk (length k, descending), Vk (n x k) are caller-allocated;
        /// 0 &lt;= k &lt;= n. A is NOT modified.
        ///
        /// <paramref name="oversample"/> extra Lanczos steps (p = min(k+oversample, n)) improve accuracy.
        /// <paramref name="seed"/> seeds the starting vector (0 → default seed, reproducible).
        /// <paramref name="partialReorth"/> selects the reorthogonalization strategy: true = partial
        /// reorth via ω-recurrence (faster at large p), false = full DGKS at every step (maximum stability).
        /// Default is true. Existing call sites without this parameter receive partialReorth = true.
        /// <paramref name="converged"/> is false if the inner bidiagonal QR did not converge, or if the
        /// Krylov space was exhausted before k triplets could be formed (rank-deficient A); remaining Sk
        /// are set to 0, Uk/Vk columns zeroed. The residual |β_last·P[p-1,t]| / (σ₀+ε) is also checked
        /// against 8·√ε; if it exceeds this tolerance, converged is set false.
        /// <paramref name="ws"/> is the GKL scratch; size it with
        /// Arena.fProxySVDTruncatedCache(m, n, k, oversample) using the SAME k and oversample.
        /// </summary>
        public static void svdTruncated(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                        int k, int oversample, uint seed, int maxIter,
                                        bool partialReorth, ref fProxySVDTruncatedCache ws, out bool converged)
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

            fProxy scaleEps = Consts.fProxyEpsilon;
            int pDone = 0;
            bool alphaBreakdown = false;

            unsafe
            {
                fProxy* UL_ptr   = ws.UL.Data.Ptr;   // layout: p×m — row j = u_j (m-vector, contiguous)
                fProxy* VL_ptr   = ws.VL.Data.Ptr;   // layout: (p+1)×n — row j = v_j (n-vector, contiguous)
                fProxy* uBuf_ptr = ws.uBuf.Data.Ptr; // m-vector scratch (also reused as coeff temp in v-reortho)
                fProxy* vBuf_ptr = ws.vBuf.Data.Ptr; // n-vector scratch (also reused as coeff temp in u-reortho)
                fProxy* A_ptr    = A.Data.Ptr;        // m×n row-major
                fProxy* mu_ptr   = ws.mu.Data.Ptr;   // length p+1: μ orthogonality estimates for U basis
                fProxy* nu_ptr   = ws.nu.Data.Ptr;   // length p+1: ν orthogonality estimates for V basis
                int     szfProxy = UnsafeUtility.SizeOf<fProxy>();

                // --- Zero-initialize UL (p×m), VL ((p+1)×n), mu, nu ---
                UnsafeUtility.MemClear(UL_ptr, (long)ws.UL.Data.Length * szfProxy);
                UnsafeUtility.MemClear(VL_ptr, (long)ws.VL.Data.Length * szfProxy);
                UnsafeUtility.MemClear(mu_ptr, (long)ws.mu.N * szfProxy);
                UnsafeUtility.MemClear(nu_ptr, (long)ws.nu.N * szfProxy);

                // --- Seed v_0: deterministic pseudo-random unit vector in R^n → stored as VL[0,:] ---
                var rng = new Random(seed == 0 ? 0x9E3779B1u : seed);
                fProxy* v0 = VL_ptr;  // VL[0,:] at offset 0 (contiguous n-vector)
                for (int i = 0; i < n; i++)
                    v0[i] = (fProxy)(rng.NextFloat() * 2f - 1f);
                fProxy seedNorm2 = Unsafe_OP.vecDot(v0, v0, n);
                if (seedNorm2 > (fProxy)0)
                    Unsafe_OP.scalMul(v0, n, (fProxy)1 / math.sqrt(seedNorm2));
                else
                    v0[0] = (fProxy)1;

                if (partialReorth)
                {
                    // ==== PARTIAL REORTHOGONALIZATION via ω-recurrence (Larsen/PROPACK) ====
                    // Maintain scalar μ_j(i) ≈ ⟨û_j,û_i⟩ and ν_{j+1}(i) ≈ ⟨v̂_{j+1},v̂_i⟩
                    // estimates. Trigger FULL DGKS sweep when max|ω| > δ or forceReorth.
                    // Extended Local Reorthogonalization (ELR) applied every step.
                    // Reference: Larsen Ph.D. 1998, lanbpro.m (PROPACK).
                    //
                    // Convention (0-based upper-bidiagonal):
                    //   û_j = (A v̂_j − β_{j-1} û_{j-1}) / α_j
                    //   v̂_{j+1} = (Aᵀ û_j − α_j v̂_j) / β_j
                    //   α_j = ws.alpha[j], β_j = ws.beta[j]
                    //
                    // METHOD-LOCAL constants (no class-level const — codegen limitation):
                    fProxy eps      = Consts.fProxyEpsilon;
                    fProxy eps1     = (fProxy)50 * eps;                           // 100*eps/2
                    fProxy delta    = math.sqrt(eps / (fProxy)p);                 // semiorthogonality trigger
                    fProxy gamma    = (fProxy)1 / math.sqrt((fProxy)2);           // ELR ratio (1/√2)
                    fProxy epsFloor = (fProxy)1.5f * eps;                         // reset level for orthogonalized ω
                    fProxy anorm    = (fProxy)0;                                  // running ‖A‖₂ estimate (order-of-mag)
                    bool forceReorth = false;                                     // interlock: force next half-step reorth

                    for (int j = 0; j < p; j++)
                    {
                        // ---- U-half: compute û_j ----

                        // uBuf = A·v̂_j − β_{j-1}·û_{j-1}   (β_{-1} term absent at j=0)
                        UnsafeUtility.MemClear(uBuf_ptr, (long)m * szfProxy);
                        Unsafe_OP.matVecDot(A_ptr, VL_ptr + j * n, uBuf_ptr, m, n);
                        if (j > 0)
                            Unsafe_OP.axpy(uBuf_ptr, UL_ptr + (j - 1) * m, -ws.beta[j - 1], m);

                        // α_j = ‖uBuf‖ (tentative)
                        ws.alpha[j] = math.sqrt(Unsafe_OP.vecDot(uBuf_ptr, uBuf_ptr, m));

                        // Update scale estimate after first step
                        if (j == 0)
                            scaleEps = Consts.fProxyEpsilon * math.max((fProxy)1, ws.alpha[0]);

                        // ELR-U — if α_j < γ·β_{j-1}, reortho uBuf against û_{j-1}
                        // and fold the projection back into the bidiagonal (lanbpro line 327).
                        // uNbrClean tracks whether the immediate neighbor was actually cleaned: true
                        // if ELR was unnecessary OR converged; false if the cap was hit while still
                        // shrinking (then the recurrence value must stand, NOT the eps floor).
                        bool uNbrClean = true;
                        if (j > 0 && ws.alpha[j] < gamma * ws.beta[j - 1])
                        {
                            uNbrClean = false;
                            fProxy normold = ws.alpha[j];
                            for (int it = 0; it < 4; it++)
                            {
                                fProxy t = Unsafe_OP.vecDot(UL_ptr + (j - 1) * m, uBuf_ptr, m);
                                Unsafe_OP.axpy(uBuf_ptr, UL_ptr + (j - 1) * m, -t, m);
                                ws.alpha[j] = math.sqrt(Unsafe_OP.vecDot(uBuf_ptr, uBuf_ptr, m));
                                ws.beta[j - 1] += t;   // fold projection into bidiagonal
                                if (ws.alpha[j] >= gamma * normold) { uNbrClean = true; break; }
                                normold = ws.alpha[j];
                            }
                        }

                        // anorm update (monotone running max; feeds T perturbation only).
                        // svdAnormBlock keeps lanbpro's α·β cross term so ‖A‖ is not underestimated.
                        anorm = math.max(anorm, (fProxy)1.01f * svdAnormBlock(ws.alpha[j], j > 0 ? ws.beta[j - 1] : (fProxy)0));

                        // μ-recurrence — estimate ⟨û_j, û_i⟩ for i = 0..j-1 (in-place safe).
                        // Convention: ν_j(i+1) for i=j-1 uses self-term ν_j(j)=1 (v̂_j is unit norm).
                        // After recurrence, set μ_j(j-1) = epsFloor (ELR immediate-neighbor floor).
                        fProxy mumax = (fProxy)0;
                        if (j > 0 && ws.alpha[j] > (fProxy)0)
                        {
                            for (int i = 0; i < j; i++)
                            {
                                // ν_j(i+1): self-term = 1 when i+1 == j; else stored nu[i+1]
                                fProxy nu_j_ip1 = (i + 1 == j) ? (fProxy)1 : nu_ptr[i + 1];
                                fProxy intermediate = ws.alpha[i] * nu_ptr[i] + ws.beta[i] * nu_j_ip1
                                                    - ws.beta[j - 1] * mu_ptr[i];
                                fProxy signI = (intermediate >= (fProxy)0) ? (fProxy)1 : (fProxy)(-1);
                                fProxy Tu = eps1 * (svdPythag(ws.alpha[j], ws.beta[j - 1])
                                                  + svdPythag(ws.alpha[i], ws.beta[i]))
                                          + eps1 * anorm;
                                mu_ptr[i] = (intermediate + signI * Tu) / ws.alpha[j];
                            }
                            // ELR immediate-neighbor floor — only when ELR actually cleaned û_{j-1}
                            // (else leave the recurrence estimate so the true loss can trigger reorth).
                            if (uNbrClean) mu_ptr[j - 1] = epsFloor;
                            for (int i = 0; i < j; i++)
                            {
                                fProxy absMu = math.abs(mu_ptr[i]);
                                if (absMu > mumax) mumax = absMu;
                            }
                        }
                        // Self-term μ_j(j) = 1 (for V-recurrence to read as mu_ptr[j])
                        mu_ptr[j] = (fProxy)1;

                        // reorth trigger U
                        bool reorthTriggerU = (j > 0) && (mumax > delta || forceReorth);
                        if (reorthTriggerU)
                        {
                            // FUTURE: windowing (compute_int strategy 0) to reorth vs subset only.
                            // Iterated classical GS vs ALL previous û_0..û_{j-1} ("twice is enough",
                            // Kahan/Parlett; lanbpro reorth.m): sweep, and keep sweeping only WHILE the
                            // norm still drops by more than γ (so the vector is not yet orthogonal),
                            // capped at 4. Only after it stops shrinking is the epsFloor reset honest.
                            // (vBuf is the j-length coefficient scratch.)
                            fProxy normrU = ws.alpha[j];
                            int nreU = 0;
                            while (true)
                            {
                                UnsafeUtility.MemClear(vBuf_ptr, (long)j * szfProxy);
                                Unsafe_OP.matVecDot(UL_ptr, uBuf_ptr, vBuf_ptr, j, m);
                                for (int l = 0; l < j; l++)
                                    Unsafe_OP.axpy(uBuf_ptr, UL_ptr + l * m, -vBuf_ptr[l], m);
                                fProxy normrOldU = normrU;
                                normrU = math.sqrt(Unsafe_OP.vecDot(uBuf_ptr, uBuf_ptr, m));
                                nreU++;
                                if (nreU > 4)
                                {
                                    // uBuf is numerically in span(UL): accept r = 0 (→ breakdown below).
                                    UnsafeUtility.MemClear(uBuf_ptr, (long)m * szfProxy);
                                    normrU = (fProxy)0;
                                    break;
                                }
                                if (normrU >= gamma * normrOldU) break;   // stopped shrinking → orthogonal
                            }
                            ws.alpha[j] = normrU;
                            for (int i = 0; i < j; i++) mu_ptr[i] = epsFloor;
                            forceReorth = !forceReorth;   // toggle: force next half-step (V-side)
                        }

                        // alpha breakdown — Krylov space exhausted
                        if (ws.alpha[j] <= scaleEps)
                        {
                            alphaBreakdown = true;
                            pDone = j;
                            break;
                        }

                        // UL[j,:] = uBuf / α_j
                        fProxy invA = (fProxy)1 / ws.alpha[j];
                        Unsafe_OP.scalMul(uBuf_ptr, m, invA);
                        UnsafeUtility.MemCpy(UL_ptr + j * m, uBuf_ptr, (long)m * szfProxy);

                        // ---- V-half: compute v̂_{j+1} ----

                        // vBuf = Aᵀ·û_j − α_j·v̂_j
                        Unsafe_OP.vecMatDot(UL_ptr + j * m, A_ptr, vBuf_ptr, m, n);
                        Unsafe_OP.axpy(vBuf_ptr, VL_ptr + j * n, -ws.alpha[j], n);

                        // β_j = ‖vBuf‖ (tentative)
                        ws.beta[j] = math.sqrt(Unsafe_OP.vecDot(vBuf_ptr, vBuf_ptr, n));

                        // ELR-V — if β_j < γ·α_j, reortho vBuf against v̂_j
                        // and fold projection into bidiagonal (lanbpro line 471). vNbrClean as in ELR-U.
                        bool vNbrClean = true;
                        if (ws.beta[j] < gamma * ws.alpha[j])
                        {
                            vNbrClean = false;
                            fProxy normold = ws.beta[j];
                            for (int it = 0; it < 4; it++)
                            {
                                fProxy t = Unsafe_OP.vecDot(VL_ptr + j * n, vBuf_ptr, n);
                                Unsafe_OP.axpy(vBuf_ptr, VL_ptr + j * n, -t, n);
                                ws.beta[j] = math.sqrt(Unsafe_OP.vecDot(vBuf_ptr, vBuf_ptr, n));
                                ws.alpha[j] += t;   // fold projection into bidiagonal
                                if (ws.beta[j] >= gamma * normold) { vNbrClean = true; break; }
                                normold = ws.beta[j];
                            }
                        }

                        // anorm update (post-ELR; α·β cross term retained — see the U-side anorm update above).
                        anorm = math.max(anorm, (fProxy)1.01f * svdAnormBlock(ws.alpha[j], ws.beta[j]));

                        // ν-recurrence — estimate ⟨v̂_{j+1}, v̂_i⟩ for i = 0..j (in-place safe).
                        // Convention: ν_j(i) for i==j uses self-term ν_j(j)=1 (v̂_j is unit norm).
                        // After recurrence, set ν_{j+1}(j) = epsFloor (ELR immediate-neighbor floor).
                        fProxy numax = (fProxy)0;
                        if (ws.beta[j] > (fProxy)0)
                        {
                            for (int i = 0; i <= j; i++)
                            {
                                // ν_j(i): self-term = 1 when i == j; else stored nu[i]
                                fProxy nu_j_i = (i == j) ? (fProxy)1 : nu_ptr[i];
                                // β_{i-1}·μ_j(i-1): dropped for i=0 (β_{-1}=0)
                                fProxy beta_im1_mu_im1 = (i > 0) ? ws.beta[i - 1] * mu_ptr[i - 1] : (fProxy)0;
                                fProxy intermediate = ws.alpha[i] * mu_ptr[i] + beta_im1_mu_im1
                                                    - ws.alpha[j] * nu_j_i;
                                fProxy signI = (intermediate >= (fProxy)0) ? (fProxy)1 : (fProxy)(-1);
                                fProxy beta_im1_T = (i > 0) ? ws.beta[i - 1] : (fProxy)0;
                                fProxy Tv = eps1 * (svdPythag(ws.alpha[j], ws.beta[j])
                                                  + svdPythag(ws.alpha[i], beta_im1_T))
                                          + eps1 * anorm;
                                nu_ptr[i] = (intermediate + signI * Tv) / ws.beta[j];
                            }
                            // ELR immediate-neighbor floor — only when ELR actually cleaned v̂_j.
                            if (vNbrClean) nu_ptr[j] = epsFloor;
                            for (int i = 0; i <= j; i++)
                            {
                                fProxy absNu = math.abs(nu_ptr[i]);
                                if (absNu > numax) numax = absNu;
                            }
                        }

                        // reorth trigger V
                        bool reorthTriggerV = (numax > delta || forceReorth);
                        if (reorthTriggerV)
                        {
                            // Iterated classical GS vs ALL previous v̂_0..v̂_j (twice-is-enough; see
                            // the U-side block). uBuf is the (j+1)-length coefficient scratch.
                            fProxy normrV = ws.beta[j];
                            int nreV = 0;
                            while (true)
                            {
                                UnsafeUtility.MemClear(uBuf_ptr, (long)(j + 1) * szfProxy);
                                Unsafe_OP.matVecDot(VL_ptr, vBuf_ptr, uBuf_ptr, j + 1, n);
                                for (int l = 0; l <= j; l++)
                                    Unsafe_OP.axpy(vBuf_ptr, VL_ptr + l * n, -uBuf_ptr[l], n);
                                fProxy normrOldV = normrV;
                                normrV = math.sqrt(Unsafe_OP.vecDot(vBuf_ptr, vBuf_ptr, n));
                                nreV++;
                                if (nreV > 4)
                                {
                                    // vBuf is numerically in span(VL): accept r = 0 (→ breakdown below).
                                    UnsafeUtility.MemClear(vBuf_ptr, (long)n * szfProxy);
                                    normrV = (fProxy)0;
                                    break;
                                }
                                if (normrV >= gamma * normrOldV) break;   // stopped shrinking → orthogonal
                            }
                            ws.beta[j] = normrV;
                            for (int i = 0; i <= j; i++) nu_ptr[i] = epsFloor;
                            forceReorth = !forceReorth;   // toggle: force next half-step (U-side)
                        }

                        pDone = j + 1;

                        // beta breakdown — invariant subspace reached
                        if (ws.beta[j] <= scaleEps)
                            break;

                        // VL[j+1,:] = vBuf / β_j
                        fProxy invB = (fProxy)1 / ws.beta[j];
                        Unsafe_OP.scalMul(vBuf_ptr, n, invB);
                        UnsafeUtility.MemCpy(VL_ptr + (j + 1) * n, vBuf_ptr, (long)n * szfProxy);
                    }
                }
                else
                {
                    // ==== partialReorth == false: FULL DGKS double-reorthogonalization ====
                    // (byte-identical to the pre-change code path)
                    for (int j = 0; j < p; j++)
                    {
                        // ----- uBuf = A * VL[j,:] -----
                        UnsafeUtility.MemClear(uBuf_ptr, (long)m * szfProxy);
                        Unsafe_OP.matVecDot(A_ptr, VL_ptr + j * n, uBuf_ptr, m, n);

                        // Subtract beta_{j-1} * UL[j-1,:]  (skipped at j=0)
                        if (j > 0)
                            Unsafe_OP.axpy(uBuf_ptr, UL_ptr + (j - 1) * m, -ws.beta[j - 1], m);

                        // DGKS double reorthogonalize uBuf against UL[0..j-1,:]
                        if (j > 0)
                        {
                            for (int pass = 0; pass < 2; pass++)
                            {
                                UnsafeUtility.MemClear(vBuf_ptr, (long)j * szfProxy);
                                Unsafe_OP.matVecDot(UL_ptr, uBuf_ptr, vBuf_ptr, j, m);
                                for (int l = 0; l < j; l++)
                                    Unsafe_OP.axpy(uBuf_ptr, UL_ptr + l * m, -vBuf_ptr[l], m);
                            }
                        }

                        // alpha_j = ||uBuf||
                        ws.alpha[j] = math.sqrt(Unsafe_OP.vecDot(uBuf_ptr, uBuf_ptr, m));

                        // Update scale estimate after first step
                        if (j == 0)
                            scaleEps = Consts.fProxyEpsilon * math.max((fProxy)1, ws.alpha[0]);

                        // Early stop: Krylov space exhausted (alpha ≈ 0)
                        if (ws.alpha[j] <= scaleEps)
                        {
                            alphaBreakdown = true;
                            pDone = j;
                            break;
                        }

                        // UL[j,:] = uBuf / alpha_j
                        fProxy invA = (fProxy)1 / ws.alpha[j];
                        Unsafe_OP.scalMul(uBuf_ptr, m, invA);
                        UnsafeUtility.MemCpy(UL_ptr + j * m, uBuf_ptr, (long)m * szfProxy);

                        // ----- vBuf = Aᵀ * UL[j,:] - alpha_j * VL[j,:] -----
                        Unsafe_OP.vecMatDot(UL_ptr + j * m, A_ptr, vBuf_ptr, m, n);
                        Unsafe_OP.axpy(vBuf_ptr, VL_ptr + j * n, -ws.alpha[j], n);

                        // DGKS double reorthogonalize vBuf against VL[0..j,:]
                        for (int pass = 0; pass < 2; pass++)
                        {
                            UnsafeUtility.MemClear(uBuf_ptr, (long)(j + 1) * szfProxy);
                            Unsafe_OP.matVecDot(VL_ptr, vBuf_ptr, uBuf_ptr, j + 1, n);
                            for (int l = 0; l <= j; l++)
                                Unsafe_OP.axpy(vBuf_ptr, VL_ptr + l * n, -uBuf_ptr[l], n);
                        }

                        // beta_j = ||vBuf||
                        ws.beta[j] = math.sqrt(Unsafe_OP.vecDot(vBuf_ptr, vBuf_ptr, n));

                        pDone = j + 1;

                        // Early stop: invariant subspace reached (beta ≈ 0)
                        if (ws.beta[j] <= scaleEps)
                            break;

                        // VL[j+1,:] = vBuf / beta_j
                        fProxy invB = (fProxy)1 / ws.beta[j];
                        Unsafe_OP.scalMul(vBuf_ptr, n, invB);
                        UnsafeUtility.MemCpy(VL_ptr + (j + 1) * n, vBuf_ptr, (long)n * szfProxy);
                    }
                }
            }

            // --- Fill Lanczos bidiagonal d/e at full size p (zero-padded beyond pDone) ---
            for (int j = 0; j < p; j++)        ws.dB[j] = (j < pDone) ? ws.alpha[j] : (fProxy)0;
            ws.eB[0] = (fProxy)0;
            for (int j = 1; j < p; j++)        ws.eB[j] = (j < pDone) ? ws.beta[j - 1] : (fProxy)0;

            // --- Inner SVD of the tiny p×p bidiagonal ---
            if (!bidiagonalSvdFromDE(ref ws.dB, ref ws.eB, ref ws.UtB, ref ws.VtB,
                                     ref ws.BsvdWs.U, ref ws.BsvdWs.S, ref ws.BsvdWs.V, p, maxIter))
            {
                converged = false;
                return;
            }

            // --- Map back to A's singular triplets ---
            int kOut = math.min(k, pDone);

            // --- Residual-based convergence check ---
            {
                fProxy betaLast = alphaBreakdown ? (fProxy)0
                                : (pDone > 0)    ? ws.beta[pDone - 1]
                                :                  (fProxy)0;
                fProxy sigma0 = (ws.BsvdWs.S.N > 0) ? ws.BsvdWs.S[0] : (fProxy)0;
                fProxy resTol = (fProxy)8 * Consts.fProxySqrtEps;
                fProxy maxRelRes = (fProxy)0;
                for (int t = 0; t < kOut; t++)
                {
                    fProxy res = math.abs(betaLast * ws.BsvdWs.U[pDone - 1, t]);
                    fProxy relRes = res / (sigma0 + Consts.fProxyEpsilon);
                    if (relRes > maxRelRes) maxRelRes = relRes;
                }
                converged = (maxRelRes < resTol);
            }

            // Fill top kOut triplets.
            unsafe
            {
                fProxy* UL_ptr   = ws.UL.Data.Ptr;
                fProxy* VL_ptr   = ws.VL.Data.Ptr;
                fProxy* uBuf_ptr = ws.uBuf.Data.Ptr;
                fProxy* vBuf_ptr = ws.vBuf.Data.Ptr;

                for (int t = 0; t < kOut; t++)
                {
                    Sk[t] = ws.BsvdWs.S[t];

                    // uBuf = sum_l P[l,t] * UL[l,:]
                    UnsafeUtility.MemClear(uBuf_ptr, (long)m * UnsafeUtility.SizeOf<fProxy>());
                    for (int l = 0; l < pDone; l++)
                        Unsafe_OP.axpy(uBuf_ptr, UL_ptr + l * m, ws.BsvdWs.U[l, t], m);
                    for (int i = 0; i < m; i++) Uk[i, t] = uBuf_ptr[i];

                    // vBuf = sum_l Q[l,t] * VL[l,:]
                    UnsafeUtility.MemClear(vBuf_ptr, (long)n * UnsafeUtility.SizeOf<fProxy>());
                    for (int l = 0; l < pDone; l++)
                        Unsafe_OP.axpy(vBuf_ptr, VL_ptr + l * n, ws.BsvdWs.V[l, t], n);
                    for (int i = 0; i < n; i++) Vk[i, t] = vBuf_ptr[i];
                }
            }

            // Zero remaining triplets if early stop produced fewer than k
            if (kOut < k)
            {
                converged = false;
                for (int t = kOut; t < k; t++)
                {
                    Sk[t] = (fProxy)0;
                    for (int i = 0; i < m; i++) Uk[i, t] = (fProxy)0;
                    for (int i = 0; i < n; i++) Vk[i, t] = (fProxy)0;
                }
            }
        }

        /// <summary>svdTruncated (ref workspace) with default partialReorth=true.</summary>
        public static void svdTruncated(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                        int k, int oversample, uint seed, int maxIter,
                                        ref fProxySVDTruncatedCache ws, out bool converged)
            => svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, maxIter, true, ref ws, out converged);

        /// <summary>svdTruncated (ref workspace) with default maxIter (75) and partialReorth=true.</summary>
        public static void svdTruncated(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                        int k, int oversample, uint seed,
                                        ref fProxySVDTruncatedCache ws, out bool converged)
            => svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, 75, true, ref ws, out converged);

        /// <summary>svdTruncated (ref workspace) with default seed and maxIter (75) and partialReorth=true.</summary>
        public static void svdTruncated(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                        int k, int oversample,
                                        ref fProxySVDTruncatedCache ws, out bool converged)
            => svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, 0x9E3779B1u, 75, true, ref ws, out converged);

        /// <summary>
        /// svdTruncated (ref workspace) with generous default Krylov width p = min(n, max(2k, k+12))
        /// and partialReorth=true.
        /// Pass a workspace from Arena.fProxySVDTruncatedCache(m, n, k) (no oversample overload)
        /// which uses the same generous formula.
        /// </summary>
        public static void svdTruncated(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                        int k, ref fProxySVDTruncatedCache ws, out bool converged)
            => svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, math.max(k, 12), 0x9E3779B1u, 75, true, ref ws, out converged);

        /// <summary>
        /// svdTruncated allocating all scratch from A's arena (explicit oversample/seed/maxIter/partialReorth).
        /// See the ref-workspace overload for semantics.
        /// </summary>
        public static void svdTruncated(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                        int k, int oversample, uint seed, int maxIter, bool partialReorth, out bool converged)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            if (k < 0 || k > n) throw new ArgumentException("svdTruncated: k must be in [0, A.N_Cols]");
            if (oversample < 0) throw new ArgumentException("svdTruncated: oversample must be >= 0");
            int p = math.min(k + oversample, n);
            var ws = new fProxySVDTruncatedCache
            {
                UL     = A.fProxyTempMat(p, m),
                VL     = A.fProxyTempMat(p + 1, n),
                dB     = A.fProxyTempVec(p),
                eB     = A.fProxyTempVec(p),
                UtB    = A.fProxyTempMat(p, p),
                VtB    = A.fProxyTempMat(p, p),
                BsvdWs = new fProxySVDFullCache
                {
                    U = A.fProxyTempMat(p, p),
                    S = A.fProxyTempVec(p),
                    V = A.fProxyTempMat(p, p)
                },
                uBuf  = A.fProxyTempVec(m),
                vBuf  = A.fProxyTempVec(n),
                alpha = A.fProxyTempVec(p),
                beta  = A.fProxyTempVec(p),
                mu    = A.fProxyTempVec(p + 1),
                nu    = A.fProxyTempVec(p + 1)
            };
            svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, maxIter, partialReorth, ref ws, out converged);
        }

        /// <summary>
        /// svdTruncated allocating all scratch from A's arena (explicit oversample/seed/maxIter),
        /// with default partialReorth=true.
        /// See the ref-workspace overload for semantics.
        /// </summary>
        public static void svdTruncated(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                        int k, int oversample, uint seed, int maxIter, out bool converged)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            if (k < 0 || k > n) throw new ArgumentException("svdTruncated: k must be in [0, A.N_Cols]");
            if (oversample < 0) throw new ArgumentException("svdTruncated: oversample must be >= 0");
            int p = math.min(k + oversample, n);
            var ws = new fProxySVDTruncatedCache
            {
                UL     = A.fProxyTempMat(p, m),
                VL     = A.fProxyTempMat(p + 1, n),
                dB     = A.fProxyTempVec(p),
                eB     = A.fProxyTempVec(p),
                UtB    = A.fProxyTempMat(p, p),
                VtB    = A.fProxyTempMat(p, p),
                BsvdWs = new fProxySVDFullCache
                {
                    U = A.fProxyTempMat(p, p),
                    S = A.fProxyTempVec(p),
                    V = A.fProxyTempMat(p, p)
                },
                uBuf  = A.fProxyTempVec(m),
                vBuf  = A.fProxyTempVec(n),
                alpha = A.fProxyTempVec(p),
                beta  = A.fProxyTempVec(p),
                mu    = A.fProxyTempVec(p + 1),
                nu    = A.fProxyTempVec(p + 1)
            };
            svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, oversample, seed, maxIter, true, ref ws, out converged);
        }

        /// <summary>
        /// svdTruncated (allocating) with generous default Krylov width p = min(n, max(2k, k+12)),
        /// default seed (0x9E3779B1u), default maxIter (75), and default partialReorth=true.
        /// </summary>
        public static void svdTruncated(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                        int k, out bool converged)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            if (k < 0 || k > n) throw new ArgumentException("svdTruncated: k must be in [0, A.N_Cols]");
            int p = math.min(n, math.max(2 * k, k + 12));
            var ws = new fProxySVDTruncatedCache
            {
                UL     = A.fProxyTempMat(p, m),
                VL     = A.fProxyTempMat(p + 1, n),
                dB     = A.fProxyTempVec(p),
                eB     = A.fProxyTempVec(p),
                UtB    = A.fProxyTempMat(p, p),
                VtB    = A.fProxyTempMat(p, p),
                BsvdWs = new fProxySVDFullCache
                {
                    U = A.fProxyTempMat(p, p),
                    S = A.fProxyTempVec(p),
                    V = A.fProxyTempMat(p, p)
                },
                uBuf  = A.fProxyTempVec(m),
                vBuf  = A.fProxyTempVec(n),
                alpha = A.fProxyTempVec(p),
                beta  = A.fProxyTempVec(p),
                mu    = A.fProxyTempVec(p + 1),
                nu    = A.fProxyTempVec(p + 1)
            };
            // Use oversample = max(k, 12) which gives p = min(k + max(k,12), n) = min(max(2k,k+12), n)
            svdTruncated(in A, ref Uk, ref Sk, ref Vk, k, math.max(k, 12), 0x9E3779B1u, 75, true, ref ws, out converged);
        }

        /// <summary>
        /// Best rank-k approximation of A (m x n, m >= n) written into Ak (m x n, caller-allocated):
        /// Ak = Σ_{t&lt;k} σ_t u_t v_tᵀ = Uk diag(Sk) Vkᵀ. This is the matrix that minimizes
        /// ||A - Ak|| over all rank-k matrices (Eckart-Young); the Frobenius error is sqrt(Σ_{i&gt;=k} σ_i²).
        /// Uses the FULL Golub-Kahan SVD (svdThin) internally — EXACT, not approximate. 0 &lt;= k &lt;= n.
        /// A is NOT modified. <paramref name="converged"/> is the SVD's flag (when false Ak is undefined).
        /// <paramref name="ws"/> is full-SVD scratch reused across calls; size it with
        /// Arena.fProxySVDFullCache(m, n).
        /// </summary>
        public static void lowRankApprox(in fProxyMxN A, ref fProxyMxN Ak, int k,
                                         ref fProxySVDFullCache ws, out bool converged, int maxIter)
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
            RequireSvdFullWorkspace(in ws, m, n);

            converged = true;

            // Ak starts at zero (rank 0); also the correct result for k == 0 / empty A.
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    Ak[i, j] = (fProxy)0;

            if (n == 0 || k == 0)
                return;

            converged = svdThin(in A, ref ws.U, ref ws.S, ref ws.V, maxIter);
            if (!converged)
                return;

            // Ak += σ_t · u_t v_tᵀ for t < k
            for (int t = 0; t < k; t++)
            {
                fProxy s = ws.S[t];
                for (int i = 0; i < m; i++)
                {
                    fProxy us = ws.U[i, t] * s;
                    for (int j = 0; j < n; j++)
                        Ak[i, j] += us * ws.V[j, t];
                }
            }
        }

        /// <summary>lowRankApprox (ref workspace) with default maxIter (75).</summary>
        public static void lowRankApprox(in fProxyMxN A, ref fProxyMxN Ak, int k,
                                         ref fProxySVDFullCache ws, out bool converged)
            => lowRankApprox(in A, ref Ak, k, ref ws, out converged, 75);

        /// <summary>
        /// lowRankApprox allocating its full-SVD scratch from A's arena.
        /// See the ref-workspace overload for semantics.
        /// </summary>
        public static void lowRankApprox(in fProxyMxN A, ref fProxyMxN Ak, int k, out bool converged, int maxIter)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            var ws = new fProxySVDFullCache
            {
                U = A.fProxyTempMat(m, n),
                S = A.fProxyTempVec(n),
                V = A.fProxyTempMat(n, n)
            };
            lowRankApprox(in A, ref Ak, k, ref ws, out converged, maxIter);
        }

        /// <summary>lowRankApprox (allocating) with default maxIter (75).</summary>
        public static void lowRankApprox(in fProxyMxN A, ref fProxyMxN Ak, int k, out bool converged)
            => lowRankApprox(in A, ref Ak, k, out converged, 75);
    }
}
