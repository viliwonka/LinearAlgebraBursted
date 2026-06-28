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
        /// flag (false -&gt; outputs undefined). A is NOT modified. Allocates O(mℓ + nℓ) scratch from
        /// A's arena.
        /// </summary>
        public static bool svdRandomized(in floatMxN A, ref floatMxN Uk, ref floatN Sk, ref floatMxN Vk,
                                         int k, int oversample, int powerIters, uint seed, int maxIter)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("svdRandomized: A must have m >= n (more rows than columns)");
            if (k < 1 || k > n)
                throw new ArgumentException("svdRandomized: k must be in [1, A.N_Cols]");
            if (oversample < 0)
                throw new ArgumentException("svdRandomized: oversample must be >= 0");
            if (powerIters < 0)
                throw new ArgumentException("svdRandomized: powerIters must be >= 0");
            if (Uk.M_Rows != m || Uk.N_Cols != k)
                throw new ArgumentException("svdRandomized: Uk must be m x k");
            if (Sk.N != k)
                throw new ArgumentException("svdRandomized: Sk must have length k");
            if (Vk.M_Rows != n || Vk.N_Cols != k)
                throw new ArgumentException("svdRandomized: Vk must be n x k");
            if (maxIter < 1)
                throw new ArgumentException("svdRandomized: maxIter must be >= 1");

            int l = math.min(k + oversample, n);   // sketch width ℓ (k <= ℓ <= n <= m)

            // Ω (n x ℓ) standard normal; Y = A Ω (m x ℓ).
            var Omega = A.tempfloatMat(n, l);
            var rng = new Random(seed == 0 ? 0x9E3779B1u : seed);
            var gauss = new floatGaussian((float)0, (float)1);
            floatRandomOP.randomInpl(ref rng, ref Omega, ref gauss);

            var Y = A.tempfloatMat(m, l);
            floatOP.dot(in A, in Omega, ref Y);          // Y = A Ω

            // Q = orth(Y): qrDecomposition overwrites Y with the thin orthonormal Q (m x ℓ).
            var R  = A.tempfloatMat(l, l);
            var qu = A.tempfloatVec(m);
            var qw = A.tempfloatVec(l);
            OrthoOP.qrDecomposition(ref Y, ref R, ref qu, ref qw);

            // Subspace iteration: Y = A (Aᵀ Q), re-orthonormalize.
            var Z = A.tempfloatMat(n, l);
            for (int it = 0; it < powerIters; it++)
            {
                floatOP.dot(in A, in Y, ref Z, true);    // Z = Aᵀ Q   (n x ℓ)
                floatOP.dot(in A, in Z, ref Y);          // Y = A Z    (m x ℓ)
                OrthoOP.qrDecomposition(ref Y, ref R, ref qu, ref qw);
            }

            // B = Qᵀ A (ℓ x n); solve its SVD exactly via Bᵀ (n x ℓ, tall): Bᵀ = Up Σ Vpᵀ, so
            // B = Vp Σ Upᵀ -> A ≈ Q B = (Q Vp) Σ Upᵀ.
            var B = A.tempfloatMat(l, n);
            floatOP.dot(in Y, in A, ref B, true);        // B = Qᵀ A

            var Bt = A.tempfloatMat(n, l);
            floatOP.trans(in B, ref Bt);                 // Bᵀ (n x ℓ)

            var Up = A.tempfloatMat(n, l);
            var Sb = A.tempfloatVec(l);
            var Vp = A.tempfloatMat(l, l);
            bool ok = svdGolubKahan(in Bt, ref Up, ref Sb, ref Vp, maxIter);
            if (!ok)
                return false;

            var UA = A.tempfloatMat(m, l);
            floatOP.dot(in Y, in Vp, ref UA);            // U = Q Vp   (m x ℓ)

            for (int t = 0; t < k; t++)
            {
                Sk[t] = Sb[t];
                for (int i = 0; i < m; i++) Uk[i, t] = UA[i, t];
                for (int i = 0; i < n; i++) Vk[i, t] = Up[i, t];
            }
            return true;
        }

        /// <summary>svdRandomized with oversample 10, powerIters 2, maxIter 75 and an explicit seed.</summary>
        public static bool svdRandomized(in floatMxN A, ref floatMxN Uk, ref floatN Sk, ref floatMxN Vk,
                                         int k, uint seed)
            => svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, 10, 2, seed, 75);

        /// <summary>svdRandomized with oversample 10, powerIters 2, maxIter 75 and the default seed.</summary>
        public static bool svdRandomized(in floatMxN A, ref floatMxN Uk, ref floatN Sk, ref floatMxN Vk, int k)
            => svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, 10, 2, 0x9E3779B1u, 75);
    }
}
