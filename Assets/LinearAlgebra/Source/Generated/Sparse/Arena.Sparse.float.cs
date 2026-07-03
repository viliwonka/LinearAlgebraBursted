using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<floatBSR> floatBSRs;
        internal UnsafeList<floatBSRBuilder> floatBSRBuilders;
        internal UnsafeList<floatBlockJacobi> floatBlockJacobis;
    }

    // Core bump-allocator primitives for the block-sparse matrix type, mirroring how
    // floatVec/floatMat are declared directly on the Arena struct (see Arena.float.cs).
    // Lifecycle wiring (AllocationsCount / Clear / Dispose) lives in Arena.cs / ArenaCore.
    public unsafe partial struct Arena
    {
        /// <summary>
        /// Allocates a compressed block-sparse (BSR) matrix with the given block-grid shape
        /// and stored-block capacity (nnzb). Typically produced by floatBSRBuilder.ToBSR
        /// rather than called directly. Arena-owned: disposed with the arena.
        /// </summary>
        public floatBSR floatBSR(int blockRows, int blockCols, int BR, int BC, int nnzb, bool uninit = false, bool symmetric = false)
        {
            var mat = new floatBSR(blockRows, blockCols, BR, BC, nnzb, in this, uninit, symmetric);
            _core->floatBSRs.Add(in mat);
            return mat;
        }

        /// <summary>
        /// Allocates a COO-of-blocks assembly builder for a blockRows x blockCols grid of
        /// BR x BC blocks. Accumulate triplets via AddBlock/AddValue, then call ToBSR(arena)
        /// once to compress into a floatBSR. Arena-owned: disposed with the arena.
        /// </summary>
        public floatBSRBuilder floatBSRBuilder(int blockRows, int blockCols, int BR, int BC, int capacityHint = 8)
        {
            var builder = new floatBSRBuilder(blockRows, blockCols, BR, BC, in this, capacityHint);
            _core->floatBSRBuilders.Add(in builder);
            return builder;
        }

        /// <summary>
        /// Builds a block-Jacobi preconditioner from A's diagonal blocks (A must be square:
        /// BlockRows==BlockCols, BR==BC). Arena-owned: disposed with the arena.
        /// </summary>
        public floatBlockJacobi floatBlockJacobi(in floatBSR A)
        {
            var pc = new floatBlockJacobi(in A, in this);
            _core->floatBlockJacobis.Add(in pc);
            return pc;
        }

        /// <summary>
        /// Materializes A^T as its own compressed BSR: every stored block at (blockRow,
        /// blockCol) becomes a block at (blockCol, blockRow), transposed in place (BR x BC ->
        /// BC x BR), then re-compressed via <see cref="floatBSRBuilder"/> (the same
        /// triplet-sort/compress path as ToBSR). One-time O(nnz) cost -- the payoff is a
        /// cache-friendly FORWARD <see cref="Sparse_OP.spMV(in floatBSR, in floatN, ref floatN)"/>
        /// over A^T in place of the scatter-heavy on-the-fly <see cref="Sparse_OP.spMVT"/>
        /// traversal on every Krylov iteration -- see <see cref="floatBSROperator"/>'s two-arg
        /// constructor and the cgls/lsqr allocating <see cref="floatBSR"/> overloads in
        /// Solvers.float.cs, which build A^T once per solve and reuse it every iteration.
        ///
        /// If A.Symmetric (implies square, and A == A^T by construction -- see
        /// floatBSR.Symmetric), returns A itself unchanged: transposing symmetric upper-block
        /// storage is a no-op, and materializing a redundant copy would only double memory for
        /// zero benefit. This is safe to feed straight into floatBSROperator's two-arg ctor:
        /// Sparse_OP.spMV already special-cases Symmetric internally, exactly matching what
        /// spMVT itself does for a symmetric A (forwards straight to spMV).
        /// </summary>
        public unsafe floatBSR floatBSRTranspose(in floatBSR A)
        {
            if (A.Symmetric)
                return A;

            var builder = floatBSRBuilder(A.BlockCols, A.BlockRows, A.BC, A.BR, A.Nnzb);

            int blockLen = A.BR * A.BC;
            float* blockT = stackalloc float[blockLen];

            for (int bi = 0; bi < A.BlockRows; bi++)
            {
                int rowStart = A.RowPtr[bi];
                int rowEnd = A.RowPtr[bi + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = A.ColInd[k];
                    float* block = A.Values.Ptr + k * blockLen;

                    // Transpose the BR x BC block into blockT (BC x BR, row-major): blockT's
                    // row c, col r == block's row r, col c.
                    for (int r = 0; r < A.BR; r++)
                        for (int c = 0; c < A.BC; c++)
                            blockT[c * A.BR + r] = block[r * A.BC + c];

                    builder.AddBlock(bj, bi, blockT);
                }
            }

            Arena self = this;
            return builder.ToBSR(ref self);
        }
    }
}
