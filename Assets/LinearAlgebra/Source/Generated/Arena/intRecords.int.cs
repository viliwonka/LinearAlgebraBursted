using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;


namespace LinearAlgebra
{
    /// <summary>
    /// Arena-owned, pointer-stable allocation record backing an intN (docs/rfc-memory-model.md
    /// §4 Option A, §7 step 4 -- mirrors fProxyVecRecord, see fProxyRecords.fProxy.cs for the full
    /// rationale). Lives inside one of ArenaCore's <see cref="ChunkedRecordTable{TRecord}"/> tables
    /// (persistent or temp pool -- see <see cref="Table"/>, declared on Arena.int.cs's ArenaCore
    /// partial) and is addressed by intN's private <c>intVecRecord*</c> field. Never copied by
    /// user code: a struct copy of intN just copies the pointer to this SAME record, so every
    /// copy resolves to the one authoritative <see cref="Data"/>.
    /// </summary>
    internal unsafe struct intVecRecord
    {
        /// <summary>The vector's authoritative backing storage.</summary>
        public UnsafeList<int> Data;

        /// <summary>
        /// The arena this record belongs to -- lets intN.Copy()/TempCopy() (and the cross-type
        /// allocation shortcuts in intN.Shortcuts.cs) reach back to the owning arena without the
        /// struct itself storing an <see cref="Arena"/> handle.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;

        /// <summary>
        /// The exact table (persistent or temp pool) this record was carved from -- lets
        /// intN.Dispose() call <c>Table-&gt;Free(SelfIndex)</c> on the right table without a
        /// separate "which pool" tag.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<intVecRecord>* Table;

        /// <summary>This record's slot index within <see cref="Table"/>.</summary>
        public int SelfIndex;
    }

    /// <summary>Matrix counterpart of <see cref="intVecRecord"/> -- backs intMxN.</summary>
    internal unsafe struct intMatRecord
    {
        public UnsafeList<int> Data;
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<intMatRecord>* Table;
        public int SelfIndex;
    }
}
