using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    /// <summary>
    /// Arena-owned, pointer-stable allocation record backing a floatN
    /// (docs/dev/rfc-memory-model.md §4 Option A, §7 step 4). Lives inside one of ArenaCore's
    /// <see cref="ChunkedRecordTable{TRecord}"/> tables (persistent or temp pool -- see
    /// <see cref="Table"/>) and is addressed by floatN's private <c>floatVecRecord*</c> field.
    /// Never copied by user code: a struct copy of floatN just copies the pointer to this SAME
    /// record, so every copy resolves to the one authoritative <see cref="Data"/> (this is what
    /// makes both of the RFC's failure modes structurally impossible for this family).
    /// </summary>
    internal unsafe struct floatVecRecord
    {
        /// <summary>The vector's authoritative backing storage.</summary>
        public UnsafeList<float> Data;

        /// <summary>
        /// The arena this record belongs to -- lets floatN.Copy()/TempCopy() (and the cross-type
        /// allocation shortcuts in floatN.Shortcuts.cs) reach back to the owning arena without the
        /// struct itself storing an <see cref="Arena"/> handle.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;

        /// <summary>
        /// The exact table (persistent or temp pool) this record was carved from -- lets
        /// floatN.Dispose() call <c>Table-&gt;Free(SelfIndex)</c> on the right table without a
        /// separate "which pool" tag.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<floatVecRecord>* Table;

        /// <summary>This record's slot index within <see cref="Table"/>.</summary>
        public int SelfIndex;
    }

    /// <summary>Matrix counterpart of <see cref="floatVecRecord"/> -- backs floatMxN.</summary>
    internal unsafe struct floatMatRecord
    {
        public UnsafeList<float> Data;
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<floatMatRecord>* Table;
        public int SelfIndex;
    }
}
