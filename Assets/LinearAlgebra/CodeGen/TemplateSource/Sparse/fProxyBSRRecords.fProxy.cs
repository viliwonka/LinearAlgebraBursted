using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Arena-owned, pointer-stable allocation record backing a fProxyBSR (docs/dev/rfc-memory-model.md
    /// §4 Option A, §7 step 4 -- mirrors fProxyVecRecord/fProxyMatRecord, see
    /// Arena/fProxyRecords.fProxy.cs). Lives inside ArenaCore's <see cref="ChunkedRecordTable{TRecord}"/>
    /// (ArenaCore.fProxyBSRRecords, see Arena.Sparse.fProxy.cs) and is addressed by fProxyBSR's
    /// private <c>fProxyBSRRecord*</c> field. Unlike fProxyVecRecord/fProxyMatRecord (a single
    /// UnsafeList payload), this carries the whole CSR-of-blocks triple (RowPtr/ColInd/Values) --
    /// fProxyBSR's scalar fields (BlockRows/BlockCols/BR/BC/Symmetric) stay ordinary inline fields,
    /// only the three growable buffers move into the record. No temp-pool counterpart exists for
    /// this family (BSR has no isTemp/fProxyTempBSR analogue).
    /// </summary>
    internal unsafe struct fProxyBSRRecord
    {
        /// <summary>Block-row pointer (length BlockRows+1).</summary>
        public UnsafeList<int> RowPtr;

        /// <summary>Block-column index per stored block (length Nnzb).</summary>
        public UnsafeList<int> ColInd;

        /// <summary>Flat, row-major-per-block values (length Nnzb*BR*BC).</summary>
        public UnsafeList<fProxy> Values;

        /// <summary>
        /// The arena this record belongs to -- reserved for future Copy()/cross-type shortcuts on
        /// fProxyBSR, mirroring fProxyVecRecord.Owner. Not read by anything yet: fProxyBSR has no
        /// Copy()/TempCopy() today.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;

        /// <summary>The table this record was carved from -- lets fProxyBSR.Dispose() call
        /// <c>Table-&gt;Free(SelfIndex)</c> on the right table.</summary>
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<fProxyBSRRecord>* Table;

        /// <summary>This record's slot index within <see cref="Table"/>.</summary>
        public int SelfIndex;
    }

    /// <summary>
    /// Preconditioner counterpart of <see cref="fProxyBSRRecord"/> -- backs fProxyBlockJacobi's
    /// single DInv payload. See ArenaCore.fProxyBlockJacobiRecords (Arena.Sparse.fProxy.cs).
    /// </summary>
    internal unsafe struct fProxyBlockJacobiRecord
    {
        /// <summary>Inverted diagonal blocks, flat row-major per block (length BlockRows*BR*BR).</summary>
        public UnsafeList<fProxy> DInv;

        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<fProxyBlockJacobiRecord>* Table;
        public int SelfIndex;
    }
}
