using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

//alsoExpand[uint]// widens this file's per-type expansion (normally int/short/long) to also emit a
//uintVecRecord/uintMatRecord pair - mirrors Arena.iProxy.cs's identical alsoExpand note: the
//iProxy-family record tables need a uint copy too, since Blas/OP.Dot.iProxy.cs's uint scratch
//allocations must be tracked exactly like every other int-family type.

namespace LinearAlgebra
{
    /// <summary>
    /// Arena-owned, pointer-stable allocation record backing an iProxyN (docs/dev/rfc-memory-model.md
    /// §4 Option A, §7 step 4 -- mirrors fProxyVecRecord, see fProxyRecords.fProxy.cs for the full
    /// rationale). Lives inside one of ArenaCore's <see cref="ChunkedRecordTable{TRecord}"/> tables
    /// (persistent or temp pool -- see <see cref="Table"/>, declared on Arena.iProxy.cs's ArenaCore
    /// partial) and is addressed by iProxyN's private <c>iProxyVecRecord*</c> field. Never copied by
    /// user code: a struct copy of iProxyN just copies the pointer to this SAME record, so every
    /// copy resolves to the one authoritative <see cref="Data"/>.
    /// </summary>
    internal unsafe struct iProxyVecRecord
    {
        /// <summary>The vector's authoritative backing storage.</summary>
        public UnsafeList<iProxy> Data;

        /// <summary>
        /// The arena this record belongs to -- lets iProxyN.Copy()/TempCopy() (and the cross-type
        /// allocation shortcuts in iProxyN.Shortcuts.cs) reach back to the owning arena without the
        /// struct itself storing an <see cref="Arena"/> handle.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;

        /// <summary>
        /// The exact table (persistent or temp pool) this record was carved from -- lets
        /// iProxyN.Dispose() call <c>Table-&gt;Free(SelfIndex)</c> on the right table without a
        /// separate "which pool" tag.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<iProxyVecRecord>* Table;

        /// <summary>This record's slot index within <see cref="Table"/>.</summary>
        public int SelfIndex;
    }

    /// <summary>Matrix counterpart of <see cref="iProxyVecRecord"/> -- backs iProxyMxN.</summary>
    internal unsafe struct iProxyMatRecord
    {
        public UnsafeList<iProxy> Data;
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<iProxyMatRecord>* Table;
        public int SelfIndex;
    }
}
