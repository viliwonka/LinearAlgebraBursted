using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;


namespace LinearAlgebra
{
    /// <summary>
    /// Arena-owned, pointer-stable allocation record backing an uintN (docs/dev/rfc-memory-model.md
    /// §4 Option A, §7 step 4 -- mirrors fProxyVecRecord, see fProxyRecords.fProxy.cs for the full
    /// rationale). Lives inside one of ArenaCore's <see cref="ChunkedRecordTable{TRecord}"/> tables
    /// (persistent or temp pool -- see <see cref="Table"/>, declared on Arena.uint.cs's ArenaCore
    /// partial) and is addressed by uintN's private <c>uintVecRecord*</c> field. Never copied by
    /// user code: a struct copy of uintN just copies the pointer to this SAME record, so every
    /// copy resolves to the one authoritative <see cref="Data"/>.
    /// </summary>
    internal unsafe struct uintVecRecord
    {
        /// <summary>The vector's authoritative backing storage.</summary>
        public UnsafeList<uint> Data;

        /// <summary>
        /// The arena this record belongs to -- lets uintN.Copy()/TempCopy() (and the cross-type
        /// allocation shortcuts in uintN.Shortcuts.cs) reach back to the owning arena without the
        /// struct itself storing an <see cref="Arena"/> handle.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;

        /// <summary>
        /// The exact table (persistent or temp pool) this record was carved from -- lets
        /// uintN.Dispose() call <c>Table-&gt;Free(SelfIndex)</c> on the right table without a
        /// separate "which pool" tag.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<uintVecRecord>* Table;

        /// <summary>This record's slot index within <see cref="Table"/>.</summary>
        public int SelfIndex;
    }

    /// <summary>Matrix counterpart of <see cref="uintVecRecord"/> -- backs uintMxN.</summary>
    internal unsafe struct uintMatRecord
    {
        public UnsafeList<uint> Data;
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<uintMatRecord>* Table;
        public int SelfIndex;
    }
}
