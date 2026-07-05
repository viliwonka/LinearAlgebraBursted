using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Arena-owned, pointer-stable allocation record backing a floatBSR (docs/rfc-memory-model.md
    /// §4 Option A, §7 step 4 -- mirrors floatVecRecord/floatMatRecord, see
    /// Arena/floatRecords.float.cs). Lives inside ArenaCore's <see cref="ChunkedRecordTable{TRecord}"/>
    /// (ArenaCore.floatBSRRecords, see Arena.Sparse.float.cs) and is addressed by floatBSR's
    /// private <c>floatBSRRecord*</c> field. Unlike floatVecRecord/floatMatRecord (a single
    /// UnsafeList payload), this carries the whole CSR-of-blocks triple (RowPtr/ColInd/Values) --
    /// floatBSR's scalar fields (BlockRows/BlockCols/BR/BC/Symmetric) stay ordinary inline fields,
    /// only the three growable buffers move into the record. No temp-pool counterpart exists for
    /// this family (BSR has no isTemp/floatTempBSR analogue).
    /// </summary>
    internal unsafe struct floatBSRRecord
    {
        /// <summary>Block-row pointer (length BlockRows+1).</summary>
        public UnsafeList<int> RowPtr;

        /// <summary>Block-column index per stored block (length Nnzb).</summary>
        public UnsafeList<int> ColInd;

        /// <summary>Flat, row-major-per-block values (length Nnzb*BR*BC).</summary>
        public UnsafeList<float> Values;

        /// <summary>
        /// The arena this record belongs to -- reserved for future Copy()/cross-type shortcuts on
        /// floatBSR, mirroring floatVecRecord.Owner. Not read by anything yet: floatBSR has no
        /// Copy()/TempCopy() today.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;

        /// <summary>The table this record was carved from -- lets floatBSR.Dispose() call
        /// <c>Table-&gt;Free(SelfIndex)</c> on the right table.</summary>
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<floatBSRRecord>* Table;

        /// <summary>This record's slot index within <see cref="Table"/>.</summary>
        public int SelfIndex;
    }

    /// <summary>
    /// Preconditioner counterpart of <see cref="floatBSRRecord"/> -- backs floatBlockJacobi's
    /// single DInv payload. See ArenaCore.floatBlockJacobiRecords (Arena.Sparse.float.cs).
    /// </summary>
    internal unsafe struct floatBlockJacobiRecord
    {
        /// <summary>Inverted diagonal blocks, flat row-major per block (length BlockRows*BR*BR).</summary>
        public UnsafeList<float> DInv;

        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<floatBlockJacobiRecord>* Table;
        public int SelfIndex;
    }
}
