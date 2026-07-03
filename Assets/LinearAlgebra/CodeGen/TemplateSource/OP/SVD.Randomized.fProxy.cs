#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;

using Random = Unity.Mathematics.Random;

namespace LinearAlgebra
{
    public static partial class SVD {

        // Randomized SVD (Halko-Martinsson-Tropp): approximate the top-k singular triplets of a large
        // A (m x n, m >= n) by RANDOM PROJECTION, much faster than a full SVD when k << n. The heavy
        // work is matrix multiplies (GEMM — the library's fastest, fully-vectorized kernel), not the
        // rotation-bound dense SVD: only a tiny ℓ x ℓ / n x ℓ problem is solved exactly.
        //
        //   ℓ = min(k + oversample, n)            (oversample p improves accuracy; HMT suggest p ~ 5-10)
        //   Ω  = n x ℓ standard-normal Gaussian
        //   Y  = A Ω,  Q = orth(Y)                (randomized range finder)
        //   repeat powerIters: Y = A (Aᵀ Q), Q = orth(Y)   (subspace iteration — sharpens accuracy when
        //                                                    the spectrum decays slowly)
        //   B  = Qᵀ A  (ℓ x n);  B = Ũ Σ Wᵀ  (exact small SVD, via Bᵀ);  U = Q Ũ
        // The leading k columns of (U, Σ, W) are the approximate top-k SVD.
        //
        // Buffers come from A's temp pool (allocating overloads) or a caller-provided
        // fProxySVDRandomizedCache (ref-workspace overloads) — see that struct for layout.

        // Default sketch seed (golden-ratio constant). Inlined rather than a const field because this
        // type-independent member would otherwise be emitted into BOTH the float and double generated
        // partials of class SVD (CS0102 duplicate).

        /// <summary>
        /// Randomized truncated SVD: approximate top-k singular triplets of A (m x n, m >= n), so
        /// A ≈ Uk diag(Sk) Vkᵀ. Uk (m x k), Sk (length k), Vk (n x k) are caller-allocated and receive
        /// the approximate leading left vectors / singular values (descending) / right vectors.
        /// 1 &lt;= k &lt;= n.
        ///
        /// APPROXIMATE: accuracy improves with <paramref name="oversample"/> (extra sketch columns,
        /// p ~ 5-10) and <paramref name="powerIters"/> (subspace-iteration passes, 1-2 for slowly
        /// decaying spectra; 0 for sharply decaying / exactly low-rank). For an exactly rank-r matrix
        /// with k &gt;= r it is exact up to rounding. The cost is dominated by GEMMs (O(mnℓ)), so it
        /// beats the full O(mn²) SVD when k ≪ n. <paramref name="seed"/> seeds the Gaussian sketch
        /// (0 -&gt; a fixed default, making the call deterministic). Returns the inner SVD's convergence
        /// flag (false -&gt; outputs undefined). A is NOT modified.
        ///
        /// <paramref name="ws"/> holds all scratch; size it with
        /// Arena.fProxySVDRandomizedCache(m, n, k, oversample) using the SAME k and oversample.
        /// </summary>
        public static bool svdRandomized(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                         int k, int oversample, int powerIters, uint seed, int maxIter,
                                         ref fProxySVDRandomizedCache ws)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            RequireRandomizedArgs(m, n, k, oversample, powerIters, in Uk, in Sk, in Vk, maxIter);

            int l = math.min(k + oversample, n);   // sketch width ℓ (k <= ℓ <= n <= m)
            RequireSvdRandomizedWorkspace(in ws, m, n, l);

            // Ω (n x ℓ) standard normal; Y = A Ω (m x ℓ).
            var rng = new Random(seed == 0 ? 0x9E3779B1u : seed);
            var gauss = new fProxyGaussian((fProxy)0, (fProxy)1);
            Rand.randomInpl(ref rng, ref ws.Omega, ref gauss);

            Blas.dot(in A, in ws.Omega, ref ws.Y);          // Y = A Ω

            // Q = orth(Y): qrDecomposition overwrites Y with the thin orthonormal Q (m x ℓ).
            QR.qrDecomposition(ref ws.Y, ref ws.R, ref ws.qu, ref ws.qw);

            // Subspace iteration: Y = A (Aᵀ Q), re-orthonormalize.
            for (int it = 0; it < powerIters; it++)
            {
                Blas.dot(in A, in ws.Y, ref ws.Z, true);    // Z = Aᵀ Q   (n x ℓ)
                Blas.dot(in A, in ws.Z, ref ws.Y);          // Y = A Z    (m x ℓ)
                QR.qrDecomposition(ref ws.Y, ref ws.R, ref ws.qu, ref ws.qw);
            }

            // B = Qᵀ A (ℓ x n); solve its SVD exactly via Bᵀ (n x ℓ, tall): Bᵀ = Up Σ Vpᵀ, so
            // B = Vp Σ Upᵀ -> A ≈ Q B = (Q Vp) Σ Upᵀ.
            Blas.dot(in ws.Y, in A, ref ws.B, true);        // B = Qᵀ A
            Blas.trans(in ws.B, ref ws.Bt);                 // Bᵀ (n x ℓ)

            bool ok = svdThin(in ws.Bt, ref ws.Up, ref ws.Sb, ref ws.Vp, maxIter);
            if (!ok)
                return false;

            Blas.dot(in ws.Y, in ws.Vp, ref ws.UA);         // U = Q Vp   (m x ℓ)

            for (int t = 0; t < k; t++)
            {
                Sk[t] = ws.Sb[t];
                for (int i = 0; i < m; i++) Uk[i, t] = ws.UA[i, t];
                for (int i = 0; i < n; i++) Vk[i, t] = ws.Up[i, t];
            }
            return true;
        }

        /// <summary>svdRandomized (ref workspace) with oversample 10, powerIters 2, maxIter 75 and an explicit seed.</summary>
        public static bool svdRandomized(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                         int k, uint seed, ref fProxySVDRandomizedCache ws)
            => svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, 10, 2, seed, 75, ref ws);

        /// <summary>svdRandomized (ref workspace) with oversample 10, powerIters 2, maxIter 75 and the default seed.</summary>
        public static bool svdRandomized(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                         int k, ref fProxySVDRandomizedCache ws)
            => svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, 10, 2, 0x9E3779B1u, 75, ref ws);

        /// <summary>
        /// svdRandomized allocating all scratch (O(mℓ + nℓ)) from A's arena. See the ref-workspace
        /// overload for semantics.
        /// </summary>
        public static bool svdRandomized(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                         int k, int oversample, int powerIters, uint seed, int maxIter)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            RequireRandomizedArgs(m, n, k, oversample, powerIters, in Uk, in Sk, in Vk, maxIter);

            int l = math.min(k + oversample, n);
            var ws = new fProxySVDRandomizedCache
            {
                Omega = A.fProxyTempMat(n, l),
                Y = A.fProxyTempMat(m, l),
                R = A.fProxyTempMat(l, l),
                qu = A.fProxyTempVec(m),
                qw = A.fProxyTempVec(l),
                Z = A.fProxyTempMat(n, l),
                B = A.fProxyTempMat(l, n),
                Bt = A.fProxyTempMat(n, l),
                Up = A.fProxyTempMat(n, l),
                Sb = A.fProxyTempVec(l),
                Vp = A.fProxyTempMat(l, l),
                UA = A.fProxyTempMat(m, l)
            };
            return svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, oversample, powerIters, seed, maxIter, ref ws);
        }

        /// <summary>svdRandomized (allocating) with oversample 10, powerIters 2, maxIter 75 and an explicit seed.</summary>
        public static bool svdRandomized(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk,
                                         int k, uint seed)
            => svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, 10, 2, seed, 75);

        /// <summary>svdRandomized (allocating) with oversample 10, powerIters 2, maxIter 75 and the default seed.</summary>
        public static bool svdRandomized(in fProxyMxN A, ref fProxyMxN Uk, ref fProxyN Sk, ref fProxyMxN Vk, int k)
            => svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, 10, 2, 0x9E3779B1u, 75);
    }
}
