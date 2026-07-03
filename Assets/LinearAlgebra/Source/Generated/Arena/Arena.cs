using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.InteropServices;
using LinearAlgebra.Sparse;
//singularFile//
namespace LinearAlgebra
{
    /// <summary>
    /// Heap-allocated body holding ALL of an arena's mutable tracking state -- every per-type
    /// growable UnsafeList (fProxyVectors/Matrices/temp*, fProxyBSRs/BSRBuilders/BlockJacobis,
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
            
            floatVectors.Length + floatMatrices.Length + floatBSRs.Length + floatBSRBuilders.Length + floatBlockJacobis.Length
            +
            doubleVectors.Length + doubleMatrices.Length + doubleBSRs.Length + doubleBSRBuilders.Length + doubleBlockJacobis.Length
            
            +
            
            intVectors.Length + intMatrices.Length
            +
            shortVectors.Length + shortMatrices.Length
            +
            longVectors.Length + longMatrices.Length
            
        ;

        public int TempAllocationsCount =>
            
            floatTempVectors.Length + floatTempMatrices.Length
            +
            doubleTempVectors.Length + doubleTempMatrices.Length
            
            +
            
            intTempVectors.Length + intTempMatrices.Length
            +
            shortTempVectors.Length + shortTempMatrices.Length
            +
            longTempVectors.Length + longTempMatrices.Length
            
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

            
            floatVectors = new UnsafeList<floatN>(8, Allocator);
            floatMatrices = new UnsafeList<floatMxN>(8, Allocator);
            floatTempVectors = new UnsafeList<floatN>(8, Allocator);
            floatTempMatrices = new UnsafeList<floatMxN>(8, Allocator);
            floatBSRs = new UnsafeList<floatBSR>(4, Allocator);
            floatBSRBuilders = new UnsafeList<floatBSRBuilder>(4, Allocator);
            floatBlockJacobis = new UnsafeList<floatBlockJacobi>(4, Allocator);
            
            doubleVectors = new UnsafeList<doubleN>(8, Allocator);
            doubleMatrices = new UnsafeList<doubleMxN>(8, Allocator);
            doubleTempVectors = new UnsafeList<doubleN>(8, Allocator);
            doubleTempMatrices = new UnsafeList<doubleMxN>(8, Allocator);
            doubleBSRs = new UnsafeList<doubleBSR>(4, Allocator);
            doubleBSRBuilders = new UnsafeList<doubleBSRBuilder>(4, Allocator);
            doubleBlockJacobis = new UnsafeList<doubleBlockJacobi>(4, Allocator);
            

            
            intVectors = new UnsafeList<intN>(8, Allocator);
            intMatrices = new UnsafeList<intMxN>(8, Allocator);
            intTempVectors = new UnsafeList<intN>(8, Allocator);
            intTempMatrices = new UnsafeList<intMxN>(8, Allocator);
            
            shortVectors = new UnsafeList<shortN>(8, Allocator);
            shortMatrices = new UnsafeList<shortMxN>(8, Allocator);
            shortTempVectors = new UnsafeList<shortN>(8, Allocator);
            shortTempMatrices = new UnsafeList<shortMxN>(8, Allocator);
            
            longVectors = new UnsafeList<longN>(8, Allocator);
            longMatrices = new UnsafeList<longMxN>(8, Allocator);
            longTempVectors = new UnsafeList<longN>(8, Allocator);
            longTempMatrices = new UnsafeList<longMxN>(8, Allocator);
            

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
            
            for (int i = 0; i < floatVectors.Length; i++)
                floatVectors[i].Dispose();
            floatVectors.Clear();

            for(int i = 0; i < floatMatrices.Length; i++)
                floatMatrices[i].Dispose();
            floatMatrices.Clear();

            for (int i = 0; i < floatBSRs.Length; i++)
                floatBSRs[i].Dispose();
            floatBSRs.Clear();

            for (int i = 0; i < floatBSRBuilders.Length; i++)
                floatBSRBuilders[i].Dispose();
            floatBSRBuilders.Clear();

            for (int i = 0; i < floatBlockJacobis.Length; i++)
                floatBlockJacobis[i].Dispose();
            floatBlockJacobis.Clear();
            
            for (int i = 0; i < doubleVectors.Length; i++)
                doubleVectors[i].Dispose();
            doubleVectors.Clear();

            for(int i = 0; i < doubleMatrices.Length; i++)
                doubleMatrices[i].Dispose();
            doubleMatrices.Clear();

            for (int i = 0; i < doubleBSRs.Length; i++)
                doubleBSRs[i].Dispose();
            doubleBSRs.Clear();

            for (int i = 0; i < doubleBSRBuilders.Length; i++)
                doubleBSRBuilders[i].Dispose();
            doubleBSRBuilders.Clear();

            for (int i = 0; i < doubleBlockJacobis.Length; i++)
                doubleBlockJacobis[i].Dispose();
            doubleBlockJacobis.Clear();
            

            
            for (int i = 0; i < intVectors.Length; i++)
                intVectors[i].Dispose();
            intVectors.Clear();

            for (int i = 0; i < intMatrices.Length; i++)
                intMatrices[i].Dispose();
            intMatrices.Clear();
            
            for (int i = 0; i < shortVectors.Length; i++)
                shortVectors[i].Dispose();
            shortVectors.Clear();

            for (int i = 0; i < shortMatrices.Length; i++)
                shortMatrices[i].Dispose();
            shortMatrices.Clear();
            
            for (int i = 0; i < longVectors.Length; i++)
                longVectors[i].Dispose();
            longVectors.Clear();

            for (int i = 0; i < longMatrices.Length; i++)
                longMatrices[i].Dispose();
            longMatrices.Clear();
            

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
            
            for (int i = 0; i < floatTempVectors.Length; i++)
                floatTempVectors[i].Dispose();
            floatTempVectors.Clear();

            for (int i = 0; i < floatTempMatrices.Length; i++)
                floatTempMatrices[i].Dispose();
            floatTempMatrices.Clear();
            
            for (int i = 0; i < doubleTempVectors.Length; i++)
                doubleTempVectors[i].Dispose();
            doubleTempVectors.Clear();

            for (int i = 0; i < doubleTempMatrices.Length; i++)
                doubleTempMatrices[i].Dispose();
            doubleTempMatrices.Clear();
            

            
            for (int i = 0; i < intTempVectors.Length; i++)
                intTempVectors[i].Dispose();
            intTempVectors.Clear();

            for (int i = 0; i < intTempMatrices.Length; i++)
                intTempMatrices[i].Dispose();
            intTempMatrices.Clear();
            
            for (int i = 0; i < shortTempVectors.Length; i++)
                shortTempVectors[i].Dispose();
            shortTempVectors.Clear();

            for (int i = 0; i < shortTempMatrices.Length; i++)
                shortTempMatrices[i].Dispose();
            shortTempMatrices.Clear();
            
            for (int i = 0; i < longTempVectors.Length; i++)
                longTempVectors[i].Dispose();
            longTempVectors.Clear();

            for (int i = 0; i < longTempMatrices.Length; i++)
                longTempMatrices[i].Dispose();
            longTempMatrices.Clear();
            

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

            
            floatVectors.Dispose();
            floatMatrices.Dispose();
            floatTempMatrices.Dispose();
            floatTempVectors.Dispose();
            floatBSRs.Dispose();
            floatBSRBuilders.Dispose();
            floatBlockJacobis.Dispose();
            
            doubleVectors.Dispose();
            doubleMatrices.Dispose();
            doubleTempMatrices.Dispose();
            doubleTempVectors.Dispose();
            doubleBSRs.Dispose();
            doubleBSRBuilders.Dispose();
            doubleBlockJacobis.Dispose();
            

            
            intVectors.Dispose();
            intMatrices.Dispose();
            intTempMatrices.Dispose();
            intTempVectors.Dispose();
            
            shortVectors.Dispose();
            shortMatrices.Dispose();
            shortTempMatrices.Dispose();
            shortTempVectors.Dispose();
            
            longVectors.Dispose();
            longMatrices.Dispose();
            longTempMatrices.Dispose();
            longTempVectors.Dispose();
            

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
    /// Thin, freely-copyable handle to a heap-allocated <see cref="ArenaCore"/>: copying an Arena
    /// (by value, through `in`/`ref` parameters, as a struct field on fProxyMxN/fProxyN/etc., ...)
    /// only copies the <c>_core</c> pointer, so every copy shares exactly ONE core -- this is what
    /// makes arena identity stable-by-address (a defensive copy the compiler makes for an `in
    /// Arena` parameter can no longer dangle, since it still points at the same live core; see
    /// docs/rfc-memory-model.md, failure mode 2 -- retires the old "must take `ref Arena`, not `in
    /// Arena`" convention). A default(Arena)/standalone handle has <c>_core == null</c>.
    ///
    /// <para><b>Ownership contract:</b> like a Unity <c>NativeContainer</c>, every copy is a view
    /// onto the SAME heap-allocated core. Exactly ONE owner must call <see cref="Dispose"/>,
    /// exactly once, on the authoritative handle -- disposing a second copy, or disposing after the
    /// original has already been disposed, double-frees the core block (undefined behavior).</para>
    ///
    /// <para><b>Not thread-safe:</b> like Unity's native containers, a single <c>Arena</c> must not
    /// be allocated from or disposed from more than one thread concurrently -- the core's tracking
    /// lists (<c>UnsafeList.Add</c> + realloc) are not atomic. Use one arena per job/thread rather
    /// than sharing one arena across threads.</para>
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

            // Free the Malloc'd block if Init throws instead of leaking it. try/finally, not
            // try/catch -- Burst/HPC# only supports throwing + try/finally cleanup, not catching
            // (see Burst's csharp-hpc-overview.md). Fully effective for plain managed callers
            // (e.g. `new Arena(...)` outside a Burst job); under Burst, `finally` is skipped on the
            // throw path too, so this guard degrades to a no-op there -- a pre-existing Burst
            // limitation (try/catch doesn't compile under Burst at all, so there's no better option).
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
        /// Repeated calls on THIS SAME handle instance are a safe no-op (nulls <c>_core</c>; the
        /// null-guarded accessors/<see cref="Clear"/> then read it as empty). That idempotence does
        /// NOT extend to other copies -- see the class-level ownership contract for why disposing a
        /// second copy double-frees the shared core.
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
