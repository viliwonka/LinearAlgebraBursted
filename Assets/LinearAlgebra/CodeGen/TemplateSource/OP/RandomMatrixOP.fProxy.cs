using System;
using Unity.Collections;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

namespace LinearAlgebra
{
    /// <summary>
    /// Random structured-matrix generators: multivariate normal, Haar-uniform random orthogonal,
    /// SPD matrices with a controlled eigenvalue range, general matrices with a target condition
    /// number or rank, and row-stochastic matrices.
    ///
    /// These are setup-time operations (not per-frame hot paths). Methods that need scratch space
    /// allocate <c>Allocator.Temp</c> buffers internally. Validation always happens before any
    /// allocation, so caller-error throws do not leak Temp buffers; buffers are disposed on the
    /// normal return path.
    ///
    /// <para><b>Multivariate-normal workflow</b>: factor Σ exactly ONCE via
    /// <c>Cholesky.choleskyDecomposition(in Sigma, ref L)</c> — check the returned
    /// <c>DirectSolveInfo.Solved</c> (false means Σ is not SPD; on a failed return L is partially
    /// overwritten and must not be reused). Then call <c>multivariateNormalInpl</c> or
    /// <c>multivariateNormalRowsInpl</c> many times with the same L. Σ may be a covariance or a
    /// correlation matrix (both SPD). Do NOT re-factor per sample.</para>
    ///
    /// fProxy-only.
    /// </summary>
    public static partial class fProxyRandomMatrix_OP
    {
        // =========================================================================
        // 1. Multivariate Normal   x = mean + L·z,  z ~ N(0,I),  Σ = L·Lᵀ
        // =========================================================================

        /// <summary>
        /// Draws one sample from N(<paramref name="mean"/>, Σ) using the pre-computed lower
        /// Cholesky factor L (Σ = L·Lᵀ, from <c>Cholesky.choleskyDecomposition</c>).
        /// Algorithm: fill <paramref name="zScratch"/> with N(0,1); then
        /// <c>dest = L·zScratch + mean</c>. Zero-alloc — caller provides scratch.
        /// <paramref name="dest"/> must not alias <paramref name="zScratch"/> (enforced by the
        /// underlying <c>dot</c> call), <paramref name="cholL"/>, or <paramref name="mean"/>
        /// (the post-dot <c>dest[i] += mean[i]</c> loop would double-count if dest aliased mean).
        /// Throws <see cref="ArgumentException"/> if any dimension is inconsistent.
        /// </summary>
        /// <param name="rng">Caller-owned RNG stream. Box-Muller advances the stream by
        /// 2·⌈n/2⌉ steps: exactly n steps for even n, n+1 for odd n.</param>
        /// <param name="cholL">Lower-triangular Cholesky factor (n×n square).</param>
        /// <param name="mean">Mean vector (length n).</param>
        /// <param name="dest">Output vector (length n; must not alias zScratch, cholL, or mean).</param>
        /// <param name="zScratch">Scratch vector (length n) for the N(0,1) draw.</param>
        public static void multivariateNormalInpl(ref Random rng, in fProxyMxN cholL, in fProxyN mean,
                                                  ref fProxyN dest, ref fProxyN zScratch)
        {
            if (!cholL.IsSquare)
                throw new ArgumentException("multivariateNormalInpl: cholL must be square");
            int n = cholL.M_Rows;
            if (mean.N != n)
                throw new ArgumentException("multivariateNormalInpl: mean.N must equal cholL dimension");
            if (dest.N != n)
                throw new ArgumentException("multivariateNormalInpl: dest.N must equal cholL dimension");
            if (zScratch.N != n)
                throw new ArgumentException("multivariateNormalInpl: zScratch.N must equal cholL dimension");

            // Fill zScratch with N(0,1) via Box-Muller
            var gauss = new fProxyGaussian((fProxy)0, (fProxy)1);
            fProxyRandom_OP.randomInpl(ref rng, ref zScratch, ref gauss);

            // dest = cholL · zScratch
            Linear_OP.dot(in cholL, in zScratch, ref dest);

            // dest += mean
            for (int i = 0; i < n; i++)
                dest[i] += mean[i];
        }

        /// <summary>
        /// Convenience overload: Temp-allocates the z scratch vector, delegates to the
        /// primitive overload, and disposes. See the primitive for full documentation.
        /// Allocates one n-element Temp vector internally (disposed before return).
        /// </summary>
        public static void multivariateNormalInpl(ref Random rng, in fProxyMxN cholL, in fProxyN mean,
                                                  ref fProxyN dest)
        {
            if (!cholL.IsSquare)
                throw new ArgumentException("multivariateNormalInpl: cholL must be square");
            int n = cholL.M_Rows;
            if (mean.N != n)
                throw new ArgumentException("multivariateNormalInpl: mean.N must equal cholL dimension");
            if (dest.N != n)
                throw new ArgumentException("multivariateNormalInpl: dest.N must equal cholL dimension");

            if (n == 0) return;

            var z = new fProxyN(n, Allocator.Temp);
            multivariateNormalInpl(ref rng, in cholL, in mean, ref dest, ref z);
            z.Dispose();
        }

        /// <summary>
        /// Fills each ROW of <paramref name="destRows"/> (shape count×n) with an independent
        /// sample from N(mean, Σ), reusing the same Cholesky factor L across all samples
        /// (amortises the factor — do NOT factor per sample). Inner scratch vectors (z and row)
        /// are allocated once as Temp and reused across the loop; both are disposed before return.
        /// Throws <see cref="ArgumentException"/> if dimensions are inconsistent.
        /// </summary>
        public static void multivariateNormalRowsInpl(ref Random rng, in fProxyMxN cholL, in fProxyN mean,
                                                      ref fProxyMxN destRows)
        {
            if (!cholL.IsSquare)
                throw new ArgumentException("multivariateNormalRowsInpl: cholL must be square");
            int n = cholL.M_Rows;
            if (mean.N != n)
                throw new ArgumentException("multivariateNormalRowsInpl: mean.N must equal cholL dimension");
            if (destRows.N_Cols != n)
                throw new ArgumentException("multivariateNormalRowsInpl: destRows.N_Cols must equal cholL dimension");

            int count = destRows.M_Rows;
            if (count == 0 || n == 0) return;

            // Allocate scratch once; amortise across all count samples
            var z   = new fProxyN(n, Allocator.Temp);
            var row = new fProxyN(n, Allocator.Temp);

            // One Box-Muller sampler across the whole loop so spare variates aren't wasted
            var gauss = new fProxyGaussian((fProxy)0, (fProxy)1);

            for (int r = 0; r < count; r++)
            {
                fProxyRandom_OP.randomInpl(ref rng, ref z, ref gauss);
                Linear_OP.dot(in cholL, in z, ref row);
                for (int c = 0; c < n; c++)
                    destRows[r, c] = row[c] + mean[c];
            }

            row.Dispose();
            z.Dispose();
        }

        // =========================================================================
        // 2. Random orthogonal (Haar-uniform)   dest ~ Haar(O(n))
        // =========================================================================

        /// <summary>
        /// Fills the square matrix <paramref name="dest"/> (n×n) with a Haar-uniform random
        /// orthogonal matrix using the Householder-QR method of Mezzadri (2007) / Stewart (1980).
        ///
        /// Algorithm:
        /// <list type="number">
        ///   <item>Fill an n×n scratch matrix G with i.i.d. N(0,1) entries.</item>
        ///   <item>QR-decompose G = Q·R (Householder).</item>
        ///   <item><b>Haar sign fix</b>: multiply column i of Q by <c>sign(R[i,i])</c>
        ///         (<c>sign(0)=+1</c>, no flip). Without this step Householder QR's Q is NOT
        ///         uniformly distributed over O(n) — the sign of each R diagonal is not equally
        ///         likely to be ±1, introducing a measurable bias. The sign flip corrects this
        ///         and yields the true Haar measure.</item>
        ///   <item>Copy the corrected Q into <paramref name="dest"/>.</item>
        /// </list>
        ///
        /// Temp scratch: G (n×n) and R (n×n) — both disposed before return. The QR step
        /// allocates an additional n-element Temp vector internally (disposed inside qrDecomposition).
        /// Throws <see cref="ArgumentException"/> if dest is not square.
        /// </summary>
        public static void randomOrthogonalInpl(ref Random rng, ref fProxyMxN dest)
        {
            if (!dest.IsSquare)
                throw new ArgumentException("randomOrthogonalInpl: dest must be square");
            int n = dest.M_Rows;
            if (n == 0) return;

            // Allocate after validation
            var G = new fProxyMxN(n, n, Allocator.Temp);
            var R = new fProxyMxN(n, n, Allocator.Temp);

            // Step 1: fill G with N(0,1)
            var gauss = new fProxyGaussian((fProxy)0, (fProxy)1);
            fProxyRandom_OP.randomInpl(ref rng, ref G, ref gauss);

            // Step 2: QR decomposition — G is overwritten with Q, R holds upper-triangular factor
            QR.qrDecomposition(ref G, ref R);

            // Step 3: Haar sign fix (Mezzadri 2007) — see algorithm doc above for the WHY.
            for (int i = 0; i < n; i++)
            {
                if (R[i, i] < (fProxy)0)
                {
                    for (int r = 0; r < n; r++)
                        G[r, i] = -G[r, i];
                }
            }

            // Step 4: copy corrected Q into dest
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    dest[r, c] = G[r, c];

            R.Dispose();
            G.Dispose();
        }

        // =========================================================================
        // 3. Random SPD with controlled spectrum
        // =========================================================================

        /// <summary>
        /// Fills the square matrix <paramref name="dest"/> (n×n) with a random symmetric
        /// positive-definite matrix whose eigenvalues are drawn uniformly in
        /// [<paramref name="minEig"/>, <paramref name="maxEig"/>].
        ///
        /// Algorithm: A = Q·Λ·Qᵀ where Q is Haar-uniform orthogonal (from
        /// <see cref="randomOrthogonalInpl"/>) and Λ = diag(λ₁,…,λₙ) with
        /// λᵢ ~ Uniform(minEig, maxEig). Qᵀ is computed BEFORE Q's columns are scaled by Λ.
        /// After forming A = QΛQᵀ, exact symmetry is enforced via A ← (A + Aᵀ)/2 to cancel
        /// floating-point asymmetry introduced by finite-precision matrix multiplication.
        /// The condition number satisfies κ(A) ≤ maxEig/minEig.
        ///
        /// Temp scratch: Q (n×n) and Qᵀ (n×n) — both disposed before return.
        /// Throws if dest is not square, <c>0 &lt; minEig ≤ maxEig</c> is violated, or either
        /// eigenvalue bound is non-finite (±Inf or NaN would otherwise silently produce NaN matrices).
        /// </summary>
        public static void randomSpdInpl(ref Random rng, ref fProxyMxN dest, fProxy minEig, fProxy maxEig)
        {
            if (!dest.IsSquare)
                throw new ArgumentException("randomSpdInpl: dest must be square");
            if (!math.isfinite(minEig) || !math.isfinite(maxEig))
                throw new ArgumentException("randomSpdInpl: minEig and maxEig must be finite");
            if (!(minEig > (fProxy)0))
                throw new ArgumentException("randomSpdInpl: minEig must be > 0");
            if (!(minEig <= maxEig))
                throw new ArgumentException("randomSpdInpl: minEig must be <= maxEig");

            int n = dest.M_Rows;
            if (n == 0) return;

            // Allocate after validation
            var Q  = new fProxyMxN(n, n, Allocator.Temp);
            var Qt = new fProxyMxN(n, n, Allocator.Temp);

            // Draw a Haar-uniform orthogonal Q
            randomOrthogonalInpl(ref rng, ref Q);

            // Qt = Qᵀ — must be computed BEFORE we scale Q's columns (otherwise Qt = (QΛ)ᵀ = ΛQᵀ)
            Linear_OP.trans(in Q, ref Qt);

            // Scale column i of Q by λᵢ ~ Uniform(minEig, maxEig) → QΛ in-place
            for (int i = 0; i < n; i++)
            {
                fProxy lambda = rng.NextFProxy(minEig, maxEig);
                for (int r = 0; r < n; r++)
                    Q[r, i] *= lambda;
            }

            // dest = QΛ · Qᵀ  (= Q_orig · Λ · Q_origᵀ)
            Linear_OP.dot(in Q, in Qt, ref dest);

            // Enforce exact symmetry: dest ← (dest + destᵀ) / 2
            // Operates only on the upper-triangle pairs to avoid redundant work.
            for (int r = 0; r < n; r++)
            {
                for (int c = r + 1; c < n; c++)
                {
                    fProxy avg = (dest[r, c] + dest[c, r]) * (fProxy)0.5;
                    dest[r, c] = avg;
                    dest[c, r] = avg;
                }
            }

            Qt.Dispose();
            Q.Dispose();
        }

        // =========================================================================
        // 4. Random matrix with target condition number   dest = U·Σ·Vᵀ
        // =========================================================================

        /// <summary>
        /// Fills the matrix <paramref name="dest"/> (m×n) with a random matrix whose condition
        /// number (ratio of largest to smallest singular value) is approximately
        /// <paramref name="cond"/>.
        ///
        /// Algorithm: dest = U·Σ·Vᵀ where U (m×m) and V (n×n) are INDEPENDENT Haar-uniform
        /// orthogonal matrices, and Σ (m×n) is diagonal with k = min(m,n) singular values
        /// logarithmically spaced in [1, cond]: σ₀ = cond (largest), σₖ₋₁ = 1 (smallest),
        /// σᵢ = cond^(1−i/(k−1)). For k = 1 (trivial rank-1 case) σ₀ = 1.
        ///
        /// For a symmetric or SPD matrix with controlled condition number, use
        /// <see cref="randomSpdInpl"/> with minEig=1, maxEig=cond instead.
        ///
        /// Temp scratch: U (m×m), V (n×n), Vᵀ (n×n), UΣ (m×n) — all disposed before return.
        /// Throws if <paramref name="cond"/> &lt; 1, is NaN, or is infinite (non-finite values
        /// would otherwise silently propagate NaN/Inf through the singular-value power computation).
        /// </summary>
        public static void randomMatrixWithConditionInpl(ref Random rng, ref fProxyMxN dest, fProxy cond)
        {
            if (!(cond >= (fProxy)1) || !math.isfinite(cond))
                throw new ArgumentException("randomMatrixWithConditionInpl: cond must be finite and >= 1");

            int m = dest.M_Rows;
            int n = dest.N_Cols;
            if (m == 0 || n == 0) return;

            int k = math.min(m, n);

            // U: m×m Haar-uniform orthogonal
            var U = new fProxyMxN(m, m, Allocator.Temp);
            randomOrthogonalInpl(ref rng, ref U);

            // V: n×n Haar-uniform orthogonal (independent of U), then Vᵀ
            var V  = new fProxyMxN(n, n, Allocator.Temp);
            var Vt = new fProxyMxN(n, n, Allocator.Temp);
            randomOrthogonalInpl(ref rng, ref V);
            Linear_OP.trans(in V, ref Vt);
            V.Dispose();

            // Build UΣ (m×n): column i of U scaled by σᵢ; remaining columns zero.
            var US = new fProxyMxN(m, n, Allocator.Temp);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    US[r, c] = (fProxy)0;

            for (int i = 0; i < k; i++)
            {
                fProxy sigma;
                if (k == 1)
                {
                    // Single singular value: condition number is 1 regardless of scale; use 1.
                    sigma = (fProxy)1;
                }
                else
                {
                    // Log-space: σ[0] = cond (largest), σ[k-1] = 1 (smallest).
                    // t = i/(k-1) ∈ [0,1],  σᵢ = cond^(1-t).
                    fProxy t = (fProxy)i / (fProxy)(k - 1);
                    sigma = math.pow(cond, (fProxy)1 - t);
                }
                for (int r = 0; r < m; r++)
                    US[r, i] = U[r, i] * sigma;
            }

            // dest = UΣ · Vᵀ
            Linear_OP.dot(in US, in Vt, ref dest);

            US.Dispose();
            Vt.Dispose();
            U.Dispose();
        }

        // =========================================================================
        // 5. Random matrix with target rank   dest = A·B
        // =========================================================================

        /// <summary>
        /// Fills the matrix <paramref name="dest"/> (m×n) with a random matrix of numerical rank
        /// exactly <paramref name="rank"/> (with probability 1 over the randomness).
        ///
        /// Algorithm: dest = A·B where A (m×rank) and B (rank×n) are filled with i.i.d. N(0,1)
        /// entries. The product has rank = <paramref name="rank"/> almost surely (the event that
        /// two independent Gaussian matrices are rank-deficient has measure zero).
        /// Special case: rank = 0 → zero matrix.
        ///
        /// Temp scratch: A (m×rank) and B (rank×n) — both disposed before return.
        /// Throws if rank &lt; 0 or rank &gt; min(m,n).
        /// </summary>
        public static void randomMatrixWithRankInpl(ref Random rng, ref fProxyMxN dest, int rank)
        {
            int m = dest.M_Rows;
            int n = dest.N_Cols;
            int maxRank = math.min(m, n);

            if (rank < 0 || rank > maxRank)
                throw new ArgumentException("randomMatrixWithRankInpl: rank must be in [0, min(m,n)]");

            if (rank == 0)
            {
                for (int r = 0; r < m; r++)
                    for (int c = 0; c < n; c++)
                        dest[r, c] = (fProxy)0;
                return;
            }

            // Allocate after validation
            var A = new fProxyMxN(m, rank, Allocator.Temp);
            var B = new fProxyMxN(rank, n, Allocator.Temp);

            var gauss = new fProxyGaussian((fProxy)0, (fProxy)1);
            fProxyRandom_OP.randomInpl(ref rng, ref A, ref gauss);
            fProxyRandom_OP.randomInpl(ref rng, ref B, ref gauss);

            // dest = A·B   (dot clears dest before accumulating)
            Linear_OP.dot(in A, in B, ref dest);

            B.Dispose();
            A.Dispose();
        }

        // =========================================================================
        // 6. Random row-stochastic
        // =========================================================================

        /// <summary>
        /// Fills <paramref name="dest"/> (m×n) with a random row-stochastic matrix: each row is
        /// a valid probability distribution (non-negative, sums to 1).
        ///
        /// Algorithm: fill with Uniform[0,1) values, then divide each row by its sum.
        /// If a row sum is 0 (astronomically unlikely with [0,1) draws, but guarded), the row is
        /// set to the uniform distribution 1/n. Zero-alloc; no Temp allocations.
        /// </summary>
        public static void randomStochasticInpl(ref Random rng, ref fProxyMxN dest)
        {
            int m = dest.M_Rows;
            int n = dest.N_Cols;
            if (m == 0 || n == 0) return;

            // Fill with Uniform[0,1)
            fProxyRandom_OP.nextUniformInpl(ref rng, ref dest);

            fProxy invN = (fProxy)1 / (fProxy)n;
            for (int r = 0; r < m; r++)
            {
                fProxy rowSum = (fProxy)0;
                for (int c = 0; c < n; c++)
                    rowSum += dest[r, c];

                if (rowSum > (fProxy)0)
                {
                    fProxy invSum = (fProxy)1 / rowSum;
                    for (int c = 0; c < n; c++)
                        dest[r, c] *= invSum;
                }
                else
                {
                    // Fallback (astronomically unlikely): uniform row
                    for (int c = 0; c < n; c++)
                        dest[r, c] = invN;
                }
            }
        }
    }
}
