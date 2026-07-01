using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.InteropServices;
using LinearAlgebra.Sparse;
//singularFile//
namespace LinearAlgebra
{
    /// <summary>
    /// Heap-allocated body holding ALL of an arena's mutable tracking state -- every per-type
    /// growable UnsafeList (fProxyVectors/Matrices/temp*, fProxyBSMs/BSMBuilders/BlockJacobis,
    /// iProxy*, Bool*, Pivots, IndexBuffers), plus Allocator/Initialized. This struct is never
    /// copied by user code: it is Malloc'd ONCE per arena and addressed exclusively through the
    /// stable <see cref="Arena"/> handle's <c>_core</c> pointer, which is what gives the arena a
    /// stable identity-by-address (see docs/rfc-memory-model.md, failure mode 2). Field/method
    /// visibility is <c>internal</c> -- only <see cref="Arena"/>'s own partials (same assembly)
    /// reach through <c>_core-&gt;</c>; nothing outside the library touches ArenaCore directly.
    /// </summary>
    internal partial struct ArenaCore
    {
        public int AllocationsCount =>
            //+copyReplaceFill[+]
            fProxyVectors.Length + fProxyMatrices.Length + fProxyBSMs.Length + fProxyBSMBuilders.Length + fProxyBlockJacobis.Length
            //-copyReplaceFill
            +
            //+copyReplaceFill[+]
            iProxyVectors.Length + iProxyMatrices.Length
            //-copyReplaceFill
        ;

        public int TempAllocationsCount =>
            //+copyReplaceFill[+]
            tempfProxyVectors.Length + tempfProxyMatrices.Length
            //-copyReplaceFill
            +
            //+copyReplaceFill[+]
            tempiProxyVectors.Length + tempiProxyMatrices.Length
            //-copyReplaceFill
        ;

        public int AllAllocationsCount => AllocationsCount + TempAllocationsCount;

        public Allocator Allocator;
        public bool Initialized;

        // internal (not private): Arena.bool.cs's factory methods on the sibling Arena type
        // reach these directly via _core->BoolVectors etc., mirroring fProxyVectors/iProxyVectors.
        internal UnsafeList<boolN> BoolVectors;
        internal UnsafeList<boolMxN> BoolMatrices;
        internal UnsafeList<boolN> TempBoolVectors;
        internal UnsafeList<boolMxN> TempBoolMatrices;

        private UnsafeList<Pivot> Pivots;
        private UnsafeList<Indices> IndexBuffers;

        public void Init(Allocator allocator)
        {
            Allocator = allocator;

            //+copyReplace
            fProxyVectors = new UnsafeList<fProxyN>(8, Allocator);
            fProxyMatrices = new UnsafeList<fProxyMxN>(8, Allocator);
            tempfProxyVectors = new UnsafeList<fProxyN>(8, Allocator);
            tempfProxyMatrices = new UnsafeList<fProxyMxN>(8, Allocator);
            fProxyBSMs = new UnsafeList<fProxyBSM>(4, Allocator);
            fProxyBSMBuilders = new UnsafeList<fProxyBSMBuilder>(4, Allocator);
            fProxyBlockJacobis = new UnsafeList<fProxyBlockJacobi>(4, Allocator);
            //-copyReplace

            //+copyReplace
            iProxyVectors = new UnsafeList<iProxyN>(8, Allocator);
            iProxyMatrices = new UnsafeList<iProxyMxN>(8, Allocator);
            tempiProxyVectors = new UnsafeList<iProxyN>(8, Allocator);
            tempiProxyMatrices = new UnsafeList<iProxyMxN>(8, Allocator);
            //-copyReplace

            BoolVectors = new UnsafeList<boolN>(2, Allocator);
            BoolMatrices = new UnsafeList<boolMxN>(2, Allocator);

            TempBoolVectors = new UnsafeList<boolN>(2, Allocator);
            TempBoolMatrices = new UnsafeList<boolMxN>(2, Allocator);

            Pivots = new UnsafeList<Pivot>(2, Allocator);
            IndexBuffers = new UnsafeList<Indices>(4, Allocator);

            // Set LAST, after every tracking list is constructed: a core that threw partway
            // through construction must never read back as Initialized (see Arena's ctor, which
            // frees a partially-constructed core on failure).
            Initialized = true;
        }

        public Pivot Pivot(int size)
        {
            var pivot = new Pivot(size, this.Allocator);
            Pivots.Add(in pivot);
            return pivot;
        }

        /// <summary>
        /// Allocates a new Indices buffer of length n from this arena.
        /// The arena owns disposal — no manual Dispose needed.
        /// </summary>
        public Indices Indices(int n)
        {
            var buf = new Indices(n, this.Allocator);
            IndexBuffers.Add(in buf);
            return buf;
        }

        public void Clear()
        {
            //+copyReplace
            for (int i = 0; i < fProxyVectors.Length; i++)
                fProxyVectors[i].Dispose();
            fProxyVectors.Clear();

            for(int i = 0; i < fProxyMatrices.Length; i++)
                fProxyMatrices[i].Dispose();
            fProxyMatrices.Clear();

            for (int i = 0; i < fProxyBSMs.Length; i++)
                fProxyBSMs[i].Dispose();
            fProxyBSMs.Clear();

            for (int i = 0; i < fProxyBSMBuilders.Length; i++)
                fProxyBSMBuilders[i].Dispose();
            fProxyBSMBuilders.Clear();

            for (int i = 0; i < fProxyBlockJacobis.Length; i++)
                fProxyBlockJacobis[i].Dispose();
            fProxyBlockJacobis.Clear();
            //-copyReplace

            //+copyReplace
            for (int i = 0; i < iProxyVectors.Length; i++)
                iProxyVectors[i].Dispose();
            iProxyVectors.Clear();

            for (int i = 0; i < iProxyMatrices.Length; i++)
                iProxyMatrices[i].Dispose();
            iProxyMatrices.Clear();
            //-copyReplace

            for (int i = 0; i < BoolVectors.Length; i++)
                BoolVectors[i].Dispose();
            BoolVectors.Clear();

            for (int i = 0; i < BoolMatrices.Length; i++)
                BoolMatrices[i].Dispose();
            BoolMatrices.Clear();

            for (int i = 0; i < Pivots.Length; i++)
                Pivots[i].Dispose();
            Pivots.Clear();

            for (int i = 0; i < IndexBuffers.Length; i++)
                IndexBuffers[i].Dispose();
            IndexBuffers.Clear();

            ClearTemp();
        }

        /// <summary>
        /// dispose only temporary allocations, produced from operations
        /// </summary>
        public void ClearTemp()
        {
            //+copyReplace
            for (int i = 0; i < tempfProxyVectors.Length; i++)
                tempfProxyVectors[i].Dispose();
            tempfProxyVectors.Clear();

            for (int i = 0; i < tempfProxyMatrices.Length; i++)
                tempfProxyMatrices[i].Dispose();
            tempfProxyMatrices.Clear();
            //-copyReplace

            //+copyReplace
            for (int i = 0; i < tempiProxyVectors.Length; i++)
                tempiProxyVectors[i].Dispose();
            tempiProxyVectors.Clear();

            for (int i = 0; i < tempiProxyMatrices.Length; i++)
                tempiProxyMatrices[i].Dispose();
            tempiProxyMatrices.Clear();
            //-copyReplace

            for (int i = 0; i < TempBoolVectors.Length; i++)
                TempBoolVectors[i].Dispose();
            TempBoolVectors.Clear();

            for (int i = 0; i < TempBoolMatrices.Length; i++)
                TempBoolMatrices[i].Dispose();
            TempBoolMatrices.Clear();
        }

        /// <summary>
        /// Disposes every tracked element AND the tracking lists themselves, then marks this
        /// core as torn down. Does NOT free the ArenaCore block itself -- that is the owning
        /// <see cref="Arena"/> handle's job (it knows the allocator the block was Malloc'd with).
        /// </summary>
        public void Dispose()
        {
            Clear();

            //+copyReplace
            fProxyVectors.Dispose();
            fProxyMatrices.Dispose();
            tempfProxyMatrices.Dispose();
            tempfProxyVectors.Dispose();
            fProxyBSMs.Dispose();
            fProxyBSMBuilders.Dispose();
            fProxyBlockJacobis.Dispose();
            //-copyReplace

            //+copyReplace
            iProxyVectors.Dispose();
            iProxyMatrices.Dispose();
            tempiProxyMatrices.Dispose();
            tempiProxyVectors.Dispose();
            //-copyReplace

            BoolVectors.Dispose();
            BoolMatrices.Dispose();
            TempBoolMatrices.Dispose();
            TempBoolVectors.Dispose();

            Pivots.Dispose();
            IndexBuffers.Dispose();

            Initialized = false;
            Allocator = Allocator.Invalid;
        }
    }

    /// <summary>
    /// Thin, freely-copyable handle to a heap-allocated <see cref="ArenaCore"/>. All of the
    /// arena's actual mutable state lives in the core; copying an Arena (by value, through
    /// `in`/`ref` parameters, as a struct field on fProxyMxN/fProxyN/etc., ...) only copies the
    /// <c>_core</c> pointer, so every copy shares exactly ONE core. This is what makes arena
    /// identity stable-by-address: a defensive copy the compiler makes for an `in Arena`
    /// parameter can no longer dangle, because the copy still points at the same live core (see
    /// docs/rfc-memory-model.md, failure mode 2 -- this retires the old "must take `ref Arena`,
    /// not `in Arena`" convention). A default(Arena)/standalone handle has <c>_core == null</c>.
    ///
    /// <para><b>Ownership contract:</b> <c>Arena</c> is a value <b>handle</b>, the same shape as a
    /// Unity <c>NativeContainer</c> (e.g. <c>NativeList</c>): every copy is a view onto the SAME
    /// heap-allocated core, not an independent arena. Exactly ONE owner must call
    /// <see cref="Dispose"/>, exactly once, on the handle that is considered authoritative.
    /// Disposing a second, independently-held copy of the same handle -- or disposing any copy
    /// after the original has already been disposed -- is undefined behavior (double-free of the
    /// core block): idempotence (see <see cref="Dispose"/>) holds only for repeated calls on the
    /// exact same handle instance, not across copies.</para>
    ///
    /// <para><b>Not thread-safe:</b> like Unity's native containers, a single <c>Arena</c> must not
    /// be allocated from or disposed from more than one thread concurrently. The core's tracking
    /// lists (<c>UnsafeList.Add</c> + realloc) are not atomic, so concurrent use across threads can
    /// corrupt tracking state or double-free. Use one arena per job/thread and partition work
    /// across arenas for parallelism, rather than sharing one arena.</para>
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe partial struct Arena : System.IDisposable
    {
        [NativeDisableUnsafePtrRestriction]
        private ArenaCore* _core;

        /// <summary>True if this handle points at a live ArenaCore (was constructed via `new Arena(allocator)`, not `default`).</summary>
        internal bool HasCore => _core != null;

        // Null-guarded (not just `_core->...`): a disposed or default/standalone Arena must
        // report empty counts rather than dereferencing a null core -- this is an existing
        // contract (e.g. reading AllocationsCount right after Dispose() is expected to return 0,
        // exercised by fProxyInitTest.InitMatrixVecPass and its per-type siblings).
        public int AllocationsCount => _core != null ? _core->AllocationsCount : 0;
        public int TempAllocationsCount => _core != null ? _core->TempAllocationsCount : 0;
        public int AllAllocationsCount => _core != null ? _core->AllAllocationsCount : 0;

        public Allocator Allocator => _core != null ? _core->Allocator : Allocator.Invalid;
        public bool Initialized => _core != null && _core->Initialized;

        public Arena(Allocator allocator)
        {
            _core = (ArenaCore*)UnsafeUtility.Malloc(UnsafeUtility.SizeOf<ArenaCore>(), UnsafeUtility.AlignOf<ArenaCore>(), allocator);

            // Free the Malloc'd block if Init throws instead of leaking it. NOTE: this is a
            // try/finally, not try/catch -- Burst/HPC# does not support catching exceptions (only
            // throwing them and try/finally for IDisposable-style cleanup; see
            // Burst's csharp-hpc-overview.md). It is fully effective for plain managed callers
            // (e.g. `new Arena(...)` outside a Burst job). Under Burst, the documented exception
            // semantics skip `finally` entirely on the throw path, so this guard degrades to a
            // no-op there -- a pre-existing Burst limitation, not a regression: the alternative
            // (try/catch) does not compile under Burst at all.
            bool ok = false;
            try
            {
                _core->Init(allocator);
                ok = true;
            }
            finally
            {
                if (!ok)
                {
                    UnsafeUtility.Free(_core, allocator);
                    _core = null;
                }
            }
        }

        public Pivot Pivot(int size)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.Pivot/Indices: arena is not initialized (default or disposed).");
            return _core->Pivot(size);
        }

        /// <summary>
        /// Allocates a new Indices buffer of length n from this arena.
        /// The arena owns disposal — no manual Dispose needed.
        /// </summary>
        public Indices Indices(int n)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.Pivot/Indices: arena is not initialized (default or disposed).");
            return _core->Indices(n);
        }

        // No-op (not a null-deref) on a default/disposed Arena: a disposed/empty arena already
        // reports AllocationsCount == 0 (see the guarded accessors above), so "disposed == empty"
        // must also hold for Clear -- there is nothing to clear, so this agrees rather than throws.
        public void Clear() { if (_core != null) _core->Clear(); }

        /// <summary>
        /// dispose only temporary allocations, produced from operations. No-op on a default or
        /// already-disposed Arena (mirrors <see cref="Clear"/>'s null-guard contract).
        /// </summary>
        public void ClearTemp() { if (_core != null) _core->ClearTemp(); }

        /// <summary>
        /// Frees this handle's <see cref="ArenaCore"/> and every allocation still tracked in it.
        /// Calling <see cref="Dispose"/> again on THIS SAME handle instance is a safe no-op (it
        /// nulls <c>_core</c> below, and the null-guarded accessors/<see cref="Clear"/> all agree
        /// that a null-core handle reads as empty). That idempotence does NOT extend to other
        /// copies of the handle: <c>Arena</c> is a value handle over one shared heap core (see the
        /// class-level ownership contract above) -- disposing a second copy of the same original
        /// handle, or disposing after a DIFFERENT copy already disposed the shared core, frees
        /// already-freed memory (double-free / undefined behavior), because that copy's own
        /// <c>_core</c> field is still non-null even though the block it points to is gone. Exactly
        /// one owning copy may call Dispose.
        /// </summary>
        public void Dispose()
        {
            if (_core == null)
                return;

            var allocator = _core->Allocator;
            _core->Dispose();
            UnsafeUtility.Free(_core, allocator);
            _core = null;
        }
    }
}
