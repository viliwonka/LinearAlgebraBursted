using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    /// <summary>
    /// Arena-owned, pointer-stable allocation record backing a doubleN. Lives inside one of
    /// ArenaCore's <see cref="ChunkedRecordTable{TRecord}"/> tables (persistent or temp pool --
    /// see <see cref="Table"/>) and is addressed by doubleN's private <c>doubleVecRecord*</c>
    /// field. Never copied by user code: a struct copy of doubleN just copies the pointer to this
    /// SAME record, so every copy resolves to the one authoritative <see cref="Data"/>.
    /// </summary>
    internal unsafe struct doubleVecRecord
    {
        /// <summary>The vector's authoritative backing storage.</summary>
        public UnsafeList<double> Data;

        /// <summary>
        /// The arena this record belongs to -- lets doubleN.Copy()/TempCopy() (and the cross-type
        /// allocation shortcuts in doubleN.Shortcuts.cs) reach back to the owning arena without the
        /// struct itself storing an <see cref="Arena"/> handle.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;

        /// <summary>
        /// The exact table (persistent or temp pool) this record was carved from -- lets
        /// doubleN.Dispose() call <c>Table-&gt;Free(SelfIndex)</c> on the right table without a
        /// separate "which pool" tag.
        /// </summary>
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<doubleVecRecord>* Table;

        /// <summary>This record's slot index within <see cref="Table"/>.</summary>
        public int SelfIndex;
    }

    /// <summary>Matrix counterpart of <see cref="doubleVecRecord"/> -- backs doubleMxN.</summary>
    internal unsafe struct doubleMatRecord
    {
        public UnsafeList<double> Data;
        [NativeDisableUnsafePtrRestriction] public ArenaCore* Owner;
        [NativeDisableUnsafePtrRestriction] public ChunkedRecordTable<doubleMatRecord>* Table;
        public int SelfIndex;
    }
}
