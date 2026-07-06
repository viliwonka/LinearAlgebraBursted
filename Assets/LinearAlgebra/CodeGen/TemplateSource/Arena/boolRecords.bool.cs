using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    /// <summary>
    /// Arena-owned, pointer-stable allocation record backing a boolN (docs/dev/rfc-memory-model.md §4
    /// Option A, §7 step 4 -- mirrors the analogous per-type record struct every other migrated
    /// family carries, e.g. floatVecRecord/intVecRecord). Lives inside one of the
    /// <see cref="ChunkedRecordTable{TRecord}"/> tables declared directly on ArenaCore in
    /// Arena.cs (bool has only one concrete type, so -- unlike the float/double or int/short/long/
    /// uint families, which need one generated file per type -- there is no per-type Arena.bool.cs
    /// field split to mirror; the record tables just live alongside Pivots/IndexBuffers in the
    /// shared, singular Arena.cs) and is addressed by boolN's private <c>boolVecRecord*</c> field.
    /// Never copied by user code: a struct copy of boolN just copies the pointer to this SAME
    /// record, so every copy resolves to the one authoritative <see cref="Data"/>.
    /// </summary>
    internal unsafe struct boolVecRecord
    {
        /// <summary>The vector's authoritative backing storage.</summary>
        public UnsafeList<bool> Data;

        /// <summary>
        /// The arena this record belongs to -- lets boolN.Copy()/TempCopy() (and the cross-type
        /// allocation shortcuts in boolN.Shortcuts.cs) reach back to the owning arena without the
        /// struct itself storing an <see cref="Arena"/> handle.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;

        /// <summary>
        /// The exact table (persistent or temp pool) this record was carved from -- lets
        /// boolN.Dispose() call <c>Table-&gt;Free(SelfIndex)</c> on the right table without a
        /// separate "which pool" tag.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<boolVecRecord>* Table;

        /// <summary>This record's slot index within <see cref="Table"/>.</summary>
        public int SelfIndex;
    }

    /// <summary>Matrix counterpart of <see cref="boolVecRecord"/> -- backs boolMxN.</summary>
    internal unsafe struct boolMatRecord
    {
        public UnsafeList<bool> Data;
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<boolMatRecord>* Table;
        public int SelfIndex;
    }
}
