using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;


namespace LinearAlgebra
{
    /// <summary>
    /// Arena-owned, pointer-stable allocation record backing an shortN (mirrors fProxyVecRecord,
    /// see fProxyRecords.fProxy.cs). Lives inside one of ArenaCore's
    /// <see cref="ChunkedRecordTable{TRecord}"/> tables (persistent or temp pool -- see
    /// <see cref="Table"/>, declared on Arena.short.cs's ArenaCore partial) and is addressed by
    /// shortN's private <c>shortVecRecord*</c> field. Never copied by user code: a struct copy
    /// of shortN just copies the pointer to this SAME record, so every copy resolves to the one
    /// authoritative <see cref="Data"/>.
    /// </summary>
    internal unsafe struct shortVecRecord
    {
        /// <summary>The vector's authoritative backing storage.</summary>
        public UnsafeList<short> Data;

        /// <summary>
        /// The arena this record belongs to -- lets shortN.Copy()/TempCopy() (and the cross-type
        /// allocation shortcuts in shortN.Shortcuts.cs) reach back to the owning arena without the
        /// struct itself storing an <see cref="Arena"/> handle.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;

        /// <summary>
        /// The exact table (persistent or temp pool) this record was carved from -- lets
        /// shortN.Dispose() call <c>Table-&gt;Free(SelfIndex)</c> on the right table without a
        /// separate "which pool" tag.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<shortVecRecord>* Table;

        /// <summary>This record's slot index within <see cref="Table"/>.</summary>
        public int SelfIndex;
    }

    /// <summary>Matrix counterpart of <see cref="shortVecRecord"/> -- backs shortMxN.</summary>
    internal unsafe struct shortMatRecord
    {
        public UnsafeList<short> Data;
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<shortMatRecord>* Table;
        public int SelfIndex;
    }
}
