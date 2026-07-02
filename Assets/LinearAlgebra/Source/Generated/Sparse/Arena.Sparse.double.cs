using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<doubleBSM> doubleBSMs;
        internal UnsafeList<doubleBSMBuilder> doubleBSMBuilders;
        internal UnsafeList<doubleBlockJacobi> doubleBlockJacobis;
    }

    // Core bump-allocator primitives for the block-sparse matrix type, mirroring how
    // doubleVec/doubleMat are declared directly on the Arena struct (see Arena.double.cs).
    // Lifecycle wiring (AllocationsCount / Clear / Dispose) lives in Arena.cs / ArenaCore.
    public unsafe partial struct Arena
    {
        /// <summary>
        /// Allocates a compressed block-sparse (BSR) matrix with the given block-grid shape
        /// and stored-block capacity (nnzb). Typically produced by doubleBSMBuilder.ToBSM
        /// rather than called directly. Arena-owned: disposed with the arena.
        /// </summary>
        public doubleBSM doubleBSM(int blockRows, int blockCols, int BR, int BC, int nnzb, bool uninit = false, bool symmetric = false)
        {
            var mat = new doubleBSM(blockRows, blockCols, BR, BC, nnzb, in this, uninit, symmetric);
            _core->doubleBSMs.Add(in mat);
            return mat;
        }

        /// <summary>
        /// Allocates a COO-of-blocks assembly builder for a blockRows x blockCols grid of
        /// BR x BC blocks. Accumulate triplets via AddBlock/AddValue, then call ToBSM(arena)
        /// once to compress into a doubleBSM. Arena-owned: disposed with the arena.
        /// </summary>
        public doubleBSMBuilder doubleBSMBuilder(int blockRows, int blockCols, int BR, int BC, int capacityHint = 8)
        {
            var builder = new doubleBSMBuilder(blockRows, blockCols, BR, BC, in this, capacityHint);
            _core->doubleBSMBuilders.Add(in builder);
            return builder;
        }

        /// <summary>
        /// Builds a block-Jacobi preconditioner from A's diagonal blocks (A must be square:
        /// BlockRows==BlockCols, BR==BC). Arena-owned: disposed with the arena.
        /// </summary>
        public doubleBlockJacobi doubleBlockJacobi(in doubleBSM A)
        {
            var pc = new doubleBlockJacobi(in A, in this);
            _core->doubleBlockJacobis.Add(in pc);
            return pc;
        }

        /// <summary>
        /// Materializes A^T as its own compressed BSM: every stored block at (blockRow,
        /// blockCol) becomes a block at (blockCol, blockRow), transposed in place (BR x BC ->
        /// BC x BR), then re-compressed via <see cref="doubleBSMBuilder"/> (the same
        /// triplet-sort/compress path as ToBSM). One-time O(nnz) cost -- the payoff is a
        /// cache-friendly FORWARD <see cref="Sparse_OP.spMV(in doubleBSM, in doubleN, ref doubleN)"/>
        /// over A^T in place of the scatter-heavy on-the-fly <see cref="Sparse_OP.spMVT"/>
        /// traversal on every Krylov iteration -- see <see cref="doubleBSMOperator"/>'s two-arg
        /// constructor and the cgls/lsqr allocating <see cref="doubleBSM"/> overloads in
        /// Solvers.double.cs, which build A^T once per solve and reuse it every iteration.
        ///
        /// If A.Symmetric (implies square, and A == A^T by construction -- see
        /// doubleBSM.Symmetric), returns A itself unchanged: transposing symmetric upper-block
        /// storage is a no-op, and materializing a redundant copy would only double memory for
        /// zero benefit. This is safe to feed straight into doubleBSMOperator's two-arg ctor:
        /// Sparse_OP.spMV already special-cases Symmetric internally, exactly matching what
        /// spMVT itself does for a symmetric A (forwards straight to spMV).
        /// </summary>
        public unsafe doubleBSM doubleBSMTranspose(in doubleBSM A)
        {
            if (A.Symmetric)
                return A;

            var builder = doubleBSMBuilder(A.BlockCols, A.BlockRows, A.BC, A.BR, A.Nnzb);

            int blockLen = A.BR * A.BC;
            double* blockT = stackalloc double[blockLen];

            for (int bi = 0; bi < A.BlockRows; bi++)
            {
                int rowStart = A.RowPtr[bi];
                int rowEnd = A.RowPtr[bi + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = A.ColInd[k];
                    double* block = A.Values.Ptr + k * blockLen;

                    // Transpose the BR x BC block into blockT (BC x BR, row-major): blockT's
                    // row c, col r == block's row r, col c.
                    for (int r = 0; r < A.BR; r++)
                        for (int c = 0; c < A.BC; c++)
                            blockT[c * A.BR + r] = block[r * A.BC + c];

                    builder.AddBlock(bj, bi, blockT);
                }
            }

            Arena self = this;
            return builder.ToBSM(ref self);
        }
    }
}
