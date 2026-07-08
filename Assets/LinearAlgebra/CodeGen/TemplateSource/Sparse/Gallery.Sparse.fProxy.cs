using System;
using Unity.Mathematics;
using LinearAlgebra;
using LinearAlgebra.Sparse;

namespace LinearAlgebra.Gallery
{
    /// <summary>
    /// Sparse (BSR) test-matrix generators — the block-sparse counterpart to the dense gallery.
    /// Unlike the dense generators these return a <see cref="fProxyBSR"/> and, crucially, never
    /// materialize a dense form, so they are usable at N ≈ 10⁴ and beyond at low fill (a dense
    /// 10000×10000 matrix is ~800 MB in float; the sparse encoding stores only the nonzero blocks).
    /// Opt in with <c>using LinearAlgebra.Gallery;</c>.
    ///
    /// Fill is at the BLOCK level: <c>density</c> is the target fraction of the blockRows×blockCols
    /// block grid that is nonzero (the diagonal blocks are always present). Actual DOF size is
    /// blockRows·BR. Determinism: fully seeded (<see cref="Unity.Mathematics.Random"/>), so a given
    /// (dims, density, seed) reproduces bit-for-bit.
    /// </summary>
    public static partial class fProxyGallery
    {
        // Off-diagonal block entries are drawn from [-OffScale, OffScale]; diagonal blocks are made
        // strongly enough dominant (per-row, exactly) that the matrix is guaranteed SPD / invertible
        // regardless of the random draw. See fProxyRandomSparseSPD.
        const float fProxySparseOffScale = 0.3f;

        /// <summary>
        /// Random symmetric positive-definite block-sparse matrix (blockRows·BR square). Each diagonal
        /// block is MᵀM + ρ·I with ρ chosen per block-row so the assembled matrix is STRICTLY block-row
        /// diagonally dominant — hence SPD by Gershgorin — for any random draw; off-diagonal blocks are
        /// added in symmetric (i,j)+(j,i) pairs at ~<paramref name="density"/> block fill. O(nnz) time
        /// and memory (no dense form). Use for CG/PCG/MINRES and the sparse eigensolvers.
        /// </summary>
        /// <param name="blockRows">Number of block rows/cols (square). DOF size = blockRows·BR.</param>
        /// <param name="BR">Square block size (2–6 hit the unrolled spMV kernels).</param>
        /// <param name="density">Target fraction of the block grid that is nonzero (e.g. 0.01–0.02).</param>
        /// <param name="seed">RNG seed (0 is remapped to 1; Unity.Mathematics.Random needs nonzero).</param>
        public static fProxyBSR fProxyRandomSparseSPD(this ref Arena arena, int blockRows, int BR, fProxy density, uint seed)
        {
            if (blockRows < 1) throw new ArgumentException("fProxyRandomSparseSPD: blockRows must be >= 1");
            if (BR < 1) throw new ArgumentException("fProxyRandomSparseSPD: BR must be >= 1");

            var rng = new Unity.Mathematics.Random(seed == 0 ? 1u : seed);
            int deg = (int)math.round((double)density * blockRows) - 1; // off-diagonals per row (diag is the +1)
            if (deg < 0) deg = 0;
            if (deg > blockRows - 1) deg = blockRows - 1;

            // Upper bound on each block-row's off-diagonal scalar-row L1, accumulated exactly as blocks
            // are added (incoming mirrors included), so the diagonal shift below is provably dominant.
            var offBound = arena.fProxyVec(blockRows);
            for (int i = 0; i < blockRows; i++) offBound[i] = (fProxy)0;

            int nnzbEstimate = blockRows + blockRows * deg * 2;
            var builder = arena.fProxyBSRBuilder(blockRows, blockRows, BR, BR, nnzbEstimate);

            var blk  = arena.fProxyMat(BR, BR);
            var blkT = arena.fProxyMat(BR, BR);
            fProxy blockRowBound = (fProxy)fProxySparseOffScale * BR; // max scalar-row L1 of one off-diag block

            // Pass 1: symmetric off-diagonal block pairs.
            for (int i = 0; i < blockRows; i++)
            {
                for (int d = 0; d < deg; d++)
                {
                    int j = rng.NextInt(0, blockRows);
                    if (j == i) continue;                 // skip self (diagonal handled in pass 2)
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                        {
                            fProxy v = (fProxy)rng.NextFloat(-fProxySparseOffScale, fProxySparseOffScale);
                            blk[r, c]  = v;
                            blkT[c, r] = v;
                        }
                    builder.AddBlock(i, j, in blk);
                    builder.AddBlock(j, i, in blkT);
                    offBound[i] += blockRowBound;
                    offBound[j] += blockRowBound;
                }
            }

            // Pass 2: diagonal blocks Di = MᵀM + ρ_i·I, ρ_i > (MᵀM within-block off-diag L1 bound BR²) +
            // (off-diagonal block L1 bound offBound[i]) => strict diagonal dominance => SPD.
            var Mi = arena.fProxyMat(BR, BR);
            for (int i = 0; i < blockRows; i++)
            {
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = (fProxy)rng.NextFloat(-1f, 1f);

                var Di = Blas.dot(Mi, Mi, true);          // MᵀM (SPD, symmetric)
                fProxy rho = (fProxy)(BR * BR) + offBound[i] + (fProxy)1;
                for (int d = 0; d < BR; d++) Di[d, d] += rho;
                builder.AddBlock(i, i, in Di);
            }

            return builder.ToBSR(ref arena);
        }

        /// <summary>
        /// Random block-sparse matrix that is NOT symmetric. Square (blockRows == blockCols) gives a
        /// strictly row-diagonally-dominant, hence invertible, non-symmetric matrix (use for BiCGSTAB /
        /// GMRES-style solvers). Tall (blockRows &gt; blockCols) gives a full-column-rank rectangular
        /// operator (the top blockCols square carries the dominant diagonal), suitable for the iterative
        /// least-squares solvers (cgls/lsqr/lsmr over a sparse operator). O(nnz), no dense form.
        /// </summary>
        /// <param name="blockRows">Block rows (m/BR).</param>
        /// <param name="blockCols">Block cols (n/BC). Equal → square; greater → tall rectangular.</param>
        /// <param name="BR">Square block size (BC == BR).</param>
        /// <param name="density">Target fraction of the block grid that is nonzero.</param>
        /// <param name="seed">RNG seed (0 → 1).</param>
        public static fProxyBSR fProxyRandomSparse(this ref Arena arena, int blockRows, int blockCols, int BR, fProxy density, uint seed)
        {
            if (blockRows < 1 || blockCols < 1) throw new ArgumentException("fProxyRandomSparse: block dims must be >= 1");
            if (blockRows < blockCols) throw new ArgumentException("fProxyRandomSparse: only square or tall (blockRows >= blockCols) is supported");
            if (BR < 1) throw new ArgumentException("fProxyRandomSparse: BR must be >= 1");

            var rng = new Unity.Mathematics.Random(seed == 0 ? 1u : seed);
            int diagCount = math.min(blockRows, blockCols);
            int deg = (int)math.round((double)density * blockCols) - 1;
            if (deg < 0) deg = 0;
            if (deg > blockCols - 1) deg = blockCols - 1;

            var offBound = arena.fProxyVec(blockRows);
            for (int i = 0; i < blockRows; i++) offBound[i] = (fProxy)0;

            int nnzbEstimate = diagCount + blockRows * (deg + 1);
            var builder = arena.fProxyBSRBuilder(blockRows, blockCols, BR, BR, nnzbEstimate);

            var blk = arena.fProxyMat(BR, BR);
            fProxy blockRowBound = (fProxy)fProxySparseOffScale * BR;

            // Off-diagonal blocks (NOT mirrored → non-symmetric). Columns range over the full block-col
            // grid so tall rows below the diagonal are populated too.
            for (int i = 0; i < blockRows; i++)
            {
                for (int d = 0; d < deg; d++)
                {
                    int j = rng.NextInt(0, blockCols);
                    if (j == i && i < diagCount) continue; // diagonal handled below
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            blk[r, c] = (fProxy)rng.NextFloat(-fProxySparseOffScale, fProxySparseOffScale);
                    builder.AddBlock(i, j, in blk);
                    offBound[i] += blockRowBound;
                }
            }

            // Dominant diagonal on the (square) top block: Di = MᵀM + ρ_i·I, ρ_i > BR² + offBound[i].
            var Mi = arena.fProxyMat(BR, BR);
            for (int i = 0; i < diagCount; i++)
            {
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = (fProxy)rng.NextFloat(-1f, 1f);

                var Di = Blas.dot(Mi, Mi, true);
                fProxy rho = (fProxy)(BR * BR) + offBound[i] + (fProxy)1;
                for (int dd = 0; dd < BR; dd++) Di[dd, dd] += rho;
                builder.AddBlock(i, i, in Di);
            }

            return builder.ToBSR(ref arena);
        }
    }
}
