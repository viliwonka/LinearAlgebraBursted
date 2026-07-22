using Unity.Collections;
using Unity.Mathematics;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    /// <summary>
    /// Constructs the tagged gallery matrices the Krylov battery drives every solver invoker
    /// through -- the ONE shared copy of generators otherwise duplicated per bespoke test file
    /// (DenseNonsym/ConvDiffDense/Poisson2D). See <see cref="GalleryProfiles"/> for the matching
    /// tags and <see cref="GalleryDenseMatrix"/>/<see cref="GalleryBSRMatrix"/> for the full entry
    /// list.
    /// </summary>
    internal static class fProxyKrylovBatteryGallery
    {
        public static fProxyMxN Build(ref Arena arena, GalleryDenseMatrix m)
        {
            switch (m)
            {
                case GalleryDenseMatrix.Laplacian1D_8: return arena.fProxyLaplacian1D(8);
                case GalleryDenseMatrix.MinIJ_5:        return arena.fProxyMinIJ(5);
                case GalleryDenseMatrix.Pei5_2:          return arena.fProxyPei(5, (fProxy)2);
                case GalleryDenseMatrix.Hilbert4:        return arena.fProxyHilbert(4);
                case GalleryDenseMatrix.Pascal5:         return arena.fProxyPascal(5);
                case GalleryDenseMatrix.Lehmer5:         return arena.fProxyLehmer(5);

                case GalleryDenseMatrix.Fiedler5:  return arena.fProxyFiedler(5);
                case GalleryDenseMatrix.Clement4:  return arena.fProxyClement(4);
                case GalleryDenseMatrix.Rosser8:   return arena.fProxyRosser();

                case GalleryDenseMatrix.DenseNonsym20:   return DenseNonsym(ref arena, 20, 0x51D01u);
                case GalleryDenseMatrix.ConvDiffDense40: return ConvDiffDense(ref arena, 40);
                case GalleryDenseMatrix.Grcar8:          return arena.fProxyGrcar(8);

                case GalleryDenseMatrix.Lauchli3_05:  return arena.fProxyLauchli(3, (fProxy)0.5);
                case GalleryDenseMatrix.Lauchli3_1e3: return arena.fProxyLauchli(3, (fProxy)1E-3);

                case GalleryDenseMatrix.WideRandom10x30: return WideRandom(ref arena, 10, 30, 10, 0x5EED2u);

                case GalleryDenseMatrix.RankDeficient20x10_Rank5: return WideRandom(ref arena, 20, 10, 5, 0x5EED3u);

                case GalleryDenseMatrix.RandSPDWellCond20: return RandSPD(ref arena, 20, (fProxy)1, (fProxy)10, 0x5EED4u);
                case GalleryDenseMatrix.RandSPDIllCond20:  return RandSPD(ref arena, 20, (fProxy)1E-3, (fProxy)1, 0x5EED5u);

                case GalleryDenseMatrix.TallRandom24x8: return arena.fProxyRandomMat(24, 8, (fProxy)(-1), (fProxy)1, 0x5EED6u);

                default: throw new System.ArgumentException("fProxyKrylovBatteryGallery.Build: unhandled GalleryDenseMatrix");
            }
        }

        public static fProxyBSR Build(ref Arena arena, GalleryBSRMatrix m)
        {
            switch (m)
            {
                case GalleryBSRMatrix.Poisson2D_20x20:       return Poisson2D(ref arena, 20, 20);
                case GalleryBSRMatrix.Laplacian2D_16x16:     return arena.fProxyLaplacian2D(16, 16);
                case GalleryBSRMatrix.RandomSparseSPD_120_2: return arena.fProxyRandomSparseSPD(120, 2, (fProxy)0.2, 0x5EED0u);
                case GalleryBSRMatrix.RandomSparseNonsym_80: return arena.fProxyRandomSparse(80, 80, 1, (fProxy)0.1, 0x5EED1u);
                default: throw new System.ArgumentException("fProxyKrylovBatteryGallery.Build: unhandled GalleryBSRMatrix");
            }
        }

        // Dense nonsymmetric, diagonally dominant (well-conditioned, nonsingular): random entries
        // plus a heavy diagonal. Not symmetric (random off-diagonals differ across the diagonal).
        static fProxyMxN DenseNonsym(ref Arena arena, int n, uint seed)
        {
            var A = arena.fProxyRandomMat(n, n, (fProxy)(-1), (fProxy)1, seed);
            for (int i = 0; i < n; i++) A[i, i] += (fProxy)(2 * n);
            return A;
        }

        // Dense 1D convection-diffusion: diagonal 6, super -1, sub -3 -- nonsymmetric,
        // diagonally dominant. Dense counterpart of the ConvDiff1D BSR generator used elsewhere.
        static fProxyMxN ConvDiffDense(ref Arena arena, int n)
        {
            var A = arena.fProxyMat(n, n);   // zero-initialized
            for (int i = 0; i < n; i++)
            {
                A[i, i] = (fProxy)6;
                if (i > 0) A[i, i - 1] = (fProxy)(-3);
                if (i < n - 1) A[i, i + 1] = (fProxy)(-1);
            }
            return A;
        }

        // m x n matrix of numerical rank `rank` (Rand.withRankInPlace).
        static fProxyMxN WideRandom(ref Arena arena, int m, int n, int rank, uint seed)
        {
            var A = arena.fProxyMat(m, n);
            var rng = new Random(seed == 0 ? 1u : seed);
            Rand.withRankInPlace(ref rng, ref A, rank);
            return A;
        }

        // n x n SPD matrix with eigenvalues in [minEig, maxEig] (Rand.spdInPlace).
        static fProxyMxN RandSPD(ref Arena arena, int n, fProxy minEig, fProxy maxEig, uint seed)
        {
            var A = arena.fProxyMat(n, n);
            var rng = new Random(seed == 0 ? 1u : seed);
            Rand.spdInPlace(ref rng, ref A, minEig, maxEig);
            return A;
        }

        // Scalar (BR=1) 2D 5-point Poisson stencil on a gx x gy grid.
        static fProxyBSR Poisson2D(ref Arena arena, int gx, int gy)
        {
            int n = gx * gy;
            var b = arena.fProxyBSRBuilder(n, n, 1, 1, 5 * n);
            for (int y = 0; y < gy; y++)
                for (int x = 0; x < gx; x++)
                {
                    int i = y * gx + x;
                    b.AddValue(i, i, (fProxy)4);
                    if (x > 0) b.AddValue(i, i - 1, (fProxy)(-1));
                    if (x < gx - 1) b.AddValue(i, i + 1, (fProxy)(-1));
                    if (y > 0) b.AddValue(i, i - gx, (fProxy)(-1));
                    if (y < gy - 1) b.AddValue(i, i + gx, (fProxy)(-1));
                }
            return b.ToBSR(ref arena);
        }
    }

    /// <summary>
    /// Dense non-diagonal SPD preconditioner (z = Nmat * r via a full mat-vec) -- the ONE shared
    /// copy of the SpdPre/BlockSpdPre structs otherwise duplicated per bespoke test file. Nmat must
    /// be symmetric positive definite (see <see cref="fProxyKrylovBatteryOracles.BuildDenseSpd"/>).
    /// </summary>
    internal struct fProxyDenseSpdPreconditioner : IfProxyPreconditioner
    {
        public fProxyMxN Nmat;   // n x n symmetric positive definite

        public bool IsIdentity => false;

        public void Apply(in fProxyN r, ref fProxyN z) => Blas.dot(in Nmat, in r, ref z);
    }

    /// <summary>
    /// Fresh (not solver-reported) relative residual oracles the Krylov battery's "Converges" /
    /// "Preconditioned convergence" checks verify against -- the ONE shared copy of
    /// RelResidualDense/RelResidualBSR otherwise duplicated per bespoke test file.
    /// </summary>
    internal static class fProxyKrylovBatteryOracles
    {
        // Non-diagonal SPD preconditioner matrix N = I + W^T W / invScale (W random n x n). Bit-
        // exactly symmetric (the (i,j) and (j,i) sums run the same k order), eigenvalues >= 1;
        // invScale tunes the eigenvalue spread / condition number.
        public static fProxyMxN BuildDenseSpd(ref Arena arena, int n, uint seed, fProxy invScale)
        {
            var W = arena.fProxyRandomMat(n, n, (fProxy)(-1), (fProxy)1, seed);
            var Nmat = arena.fProxyMat(n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy s = (fProxy)0;
                    for (int k = 0; k < n; k++) s += W[k, i] * W[k, j];
                    Nmat[i, j] = s / invScale + (i == j ? (fProxy)1 : (fProxy)0);
                }
            return Nmat;
        }

        public static fProxy RelResidualDense(in fProxyMxN A, in fProxyN x, in fProxyN b)
        {
            var Ax = Blas.dot(A, x);
            fProxy num = 0, den = 0;
            for (int i = 0; i < b.N; i++) { fProxy d = Ax[i] - b[i]; num += d * d; den += b[i] * b[i]; }
            return math.sqrt(num) / math.sqrt(math.max(den, (fProxy)1e-30));
        }

        public static fProxy RelResidualBSR(in fProxyBSR A, in fProxyN x, in fProxyN b)
        {
            var Ax = BSR.spMV(in A, in x);
            fProxy num = 0, den = 0;
            for (int i = 0; i < b.N; i++) { fProxy d = Ax[i] - b[i]; num += d * d; den += b[i] * b[i]; }
            return math.sqrt(num) / math.sqrt(math.max(den, (fProxy)1e-30));
        }

        // Block counterparts of RelResidualDense/RelResidualBSR (Frobenius norm over the whole s x n
        // block) for the block-battery family. AX is computed via the GENERAL block-apply formula
        // (Blas.dot(X, A, transposeB: true) / BSR.spMM, both correct for ANY square A) rather than
        // through a caller-supplied operator wrapper, so this oracle is never exposed to
        // fProxyDenseOperator.ApplyBlock's symmetric-only landmine (SS4).
        public static fProxy RelResidualBlockDense(in fProxyMxN A, in fProxyMxN X, in fProxyMxN B)
        {
            int s = B.M_Rows, n = B.N_Cols;
            var AX = new fProxyMxN(s, n, Allocator.Temp, true);
            Blas.dot(in X, in A, ref AX, false, true);
            fProxy num = 0, den = 0;
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++)
                { fProxy d = AX[i, c] - B[i, c]; num += d * d; den += B[i, c] * B[i, c]; }
            AX.Dispose();
            return math.sqrt(num) / math.sqrt(math.max(den, (fProxy)1e-30));
        }

        public static fProxy RelResidualBlockBSR(in fProxyBSR A, in fProxyMxN X, in fProxyMxN B)
        {
            int s = B.M_Rows, n = B.N_Cols;
            var AX = new fProxyMxN(s, n, Allocator.Temp, true);
            BSR.spMM(in A, in X, ref AX, s);
            fProxy num = 0, den = 0;
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++)
                { fProxy d = AX[i, c] - B[i, c]; num += d * d; den += B[i, c] * B[i, c]; }
            AX.Dispose();
            return math.sqrt(num) / math.sqrt(math.max(den, (fProxy)1e-30));
        }

        // Generic block-residual oracle via a caller-supplied operator's ApplyBlock -- used where no
        // raw fProxyMxN/fProxyBSR is at hand (e.g. CheckBlockAdditions, which only carries the
        // already-correctly-selected TOp). Safe for ANY TOp: the caller is responsible for having
        // picked the symmetric-safe/general variant appropriate for the underlying matrix (same
        // requirement every ApplyBlock call in this battery already has).
        public static fProxy RelResidualBlockOp<TOp>(in TOp Aop, in fProxyMxN X, in fProxyMxN B, int s, int n)
            where TOp : struct, IfProxyLinearOperator
        {
            var AX = new fProxyMxN(s, n, Allocator.Temp, true);
            Aop.ApplyBlock(in X, ref AX, s);
            fProxy num = 0, den = 0;
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++)
                { fProxy d = AX[i, c] - B[i, c]; num += d * d; den += B[i, c] * B[i, c]; }
            AX.Dispose();
            return math.sqrt(num) / math.sqrt(math.max(den, (fProxy)1e-30));
        }

        // Row j of B (length n) as an independent fProxyN -- the per-column extraction every
        // block-battery check that compares against a scalar solve needs.
        public static fProxyN Row(ref Arena arena, in fProxyMxN B, int j, int n)
        {
            var v = arena.fProxyVec(n);
            for (int c = 0; c < n; c++) v[c] = B[j, c];
            return v;
        }
    }
}
