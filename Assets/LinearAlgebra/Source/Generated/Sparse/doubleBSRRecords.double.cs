using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Arena-owned, pointer-stable allocation record backing a doubleBSR (docs/dev/rfc-memory-model.md
    /// §4 Option A, §7 step 4 -- mirrors doubleVecRecord/doubleMatRecord, see
    /// Arena/doubleRecords.double.cs). Lives inside ArenaCore's <see cref="ChunkedRecordTable{TRecord}"/>
    /// (ArenaCore.doubleBSRRecords, see Arena.Sparse.double.cs) and is addressed by doubleBSR's
    /// private <c>doubleBSRRecord*</c> field. Unlike doubleVecRecord/doubleMatRecord (a single
    /// UnsafeList payload), this carries the whole CSR-of-blocks triple (RowPtr/ColInd/Values) --
    /// doubleBSR's scalar fields (BlockRows/BlockCols/BR/BC/Symmetric) stay ordinary inline fields,
    /// only the three growable buffers move into the record. No temp-pool counterpart exists for
    /// this family (BSR has no isTemp/doubleTempBSR analogue).
    /// </summary>
    internal unsafe struct doubleBSRRecord
    {
        /// <summary>Block-row pointer (length BlockRows+1).</summary>
        public UnsafeList<int> RowPtr;

        /// <summary>Block-column index per stored block (length Nnzb).</summary>
        public UnsafeList<int> ColInd;

        /// <summary>Flat, row-major-per-block values (length Nnzb*BR*BC).</summary>
        public UnsafeList<double> Values;

        /// <summary>
        /// The arena this record belongs to -- reserved for future Copy()/cross-type shortcuts on
        /// doubleBSR, mirroring doubleVecRecord.Owner. Not read by anything yet: doubleBSR has no
        /// Copy()/TempCopy() today.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;

        /// <summary>The table this record was carved from -- lets doubleBSR.Dispose() call
        /// <c>Table-&gt;Free(SelfIndex)</c> on the right table.</summary>
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<doubleBSRRecord>* Table;

        /// <summary>This record's slot index within <see cref="Table"/>.</summary>
        public int SelfIndex;
    }

    /// <summary>
    /// Preconditioner counterpart of <see cref="doubleBSRRecord"/> -- backs doubleBlockJacobi's
    /// single DInv payload. See ArenaCore.doubleBlockJacobiRecords (Arena.Sparse.double.cs).
    /// </summary>
    internal unsafe struct doubleBlockJacobiRecord
    {
        /// <summary>Inverted diagonal blocks, flat row-major per block (length BlockRows*BR*BR).</summary>
        public UnsafeList<double> DInv;

        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<doubleBlockJacobiRecord>* Table;
        public int SelfIndex;
    }
}
