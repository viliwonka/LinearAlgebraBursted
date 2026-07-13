using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.InteropServices;
using System.Threading;
using LinearAlgebra.Sparse;
//singularFile//
//alsoExpand[uint]// ArenaCore merges every generated type's pools into this ONE file via the
//copyReplace/copyReplaceFill blocks below (unlike a per-type file, which would emit a separate
//Arena.<type>.cs per type). The uint record tables themselves are declared in Arena.iProxy.cs's
//alsoExpand-widened partial (uintVecRecords/uintMatRecords/uintTempVecRecords/uintTempMatRecords) -
//this flag widens the iProxy-token blocks below the SAME way, so Init/Clear/ClearTemp/Dispose/
//*AllocationsCount actually construct/clear/dispose/count those tables instead of leaving them
//default-constructed garbage.
namespace LinearAlgebra
{
    /// <summary>
    /// Heap-allocated body holding ALL of an arena's mutable tracking state, plus
    /// Allocator/Initialized. Never copied by user code: Malloc'd ONCE per arena and addressed
    /// exclusively through the stable <see cref="Arena"/> handle's <c>_core</c> pointer, which is
    /// what gives the arena a stable identity-by-address. Field/method visibility is
    /// <c>internal</c> -- only <see cref="Arena"/>'s own partials (same assembly) reach through
    /// <c>_core-&gt;</c>; nothing outside the library touches ArenaCore directly.
    ///
    /// <para>Most families (float/double, int/short/long/uint, bool, sparse BSR/BlockJacobi) own
    /// pointer-stable <see cref="ChunkedRecordTable{TRecord}"/> tables: each family's struct holds
    /// a stable record pointer instead of being tracked by a separate value copy, and
    /// Dispose()/Clear()/ClearTemp() free individual slots. <c>fProxyBSRBuilders</c>,
    /// <c>Pivots</c>, and <c>IndexBuffers</c> stay on the older growable-UnsafeList-of-value-copies
    /// model instead: tracked by <c>.Add</c>, bulk-freed by <c>.Clear()</c>/<c>.Dispose()</c> on
    /// the whole list, with no per-instance early-dispose bookkeeping. See
    /// <see cref="AllocationsCount"/>'s doc for the one user-visible consequence.</para>
    /// </summary>
    internal unsafe partial struct ArenaCore
    {
        // ---- concurrency guards: an Arena is single-threaded by contract; these two mechanisms
        // make a violation detectable instead of silently corrupting the record tables. Both live
        // HERE, inside the heap-Malloc'd ArenaCore, rather than as fields on the pointer-sized
        // Arena handle struct -- see Arena/DEVLOG.md for why.
#if ENABLE_UNITY_COLLECTIONS_CHECKS
        /// <summary>Dispose-lifetime safety handle, checked at the top of every guarded mutating
        /// entry point below. Created once in <see cref="Init"/>, released once in
        /// <see cref="Dispose"/>.</summary>
        internal AtomicSafetyHandle Safety;
#endif

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        /// <summary>
        /// Race tripwire. 0 == free, 1 == a guarded mutating body is currently executing. Not a
        /// mutex -- it never blocks or waits, it only detects two mutating calls overlapping in
        /// time and throws on the second one. Guards ONLY the bodies of the ArenaCore-level
        /// factory/Allocate/Clear/ClearTemp/Dispose/Pivot/Indices entry points. Known gap: an
        /// individual buffer's own <c>Dispose()</c> (e.g. <c>fProxyN.Dispose()</c>) also mutates a
        /// record table but is NOT guarded here, so concurrent frees/allocates on the same table
        /// are a real, uncaught race surface. Reentrancy is avoided structurally rather than by
        /// thread-identity or counting: of the factory overloads that forward to another guarded
        /// overload, only the terminal overload that actually touches a record table calls
        /// <see cref="EnterMutation"/>, so forwarding wrappers never nest.
        /// </summary>
        private int _busy;

        /// <summary>
        /// Marks entry into a guarded mutating body: checks <see cref="Safety"/> (throws if this
        /// core was already Dispose()'d), then arms the <see cref="_busy"/> tripwire (throws if
        /// another mutating call is already in flight). Combined into one call so every guarded
        /// call site -- including the per-type factory methods in <c>Arena.fProxy.cs</c>/
        /// <c>Arena.iProxy.cs</c>/<c>Arena.bool.cs</c>/<c>Arena.Sparse.fProxy.cs</c>, which live
        /// outside this file and don't otherwise reference <c>AtomicSafetyHandle</c> -- only needs
        /// <c>_core-&gt;EnterMutation()</c> / <c>_core-&gt;ExitMutation()</c>, not two separate
        /// checks. See <see cref="_busy"/>'s doc for why this is safe against legitimate
        /// same-thread nesting (there isn't any left to trip on -- reentrancy is avoided
        /// structurally, not by counting).
        /// </summary>
        /// <exception cref="System.InvalidOperationException">
        /// Two mutating calls overlapped. Names the single-threaded contract so the exception
        /// itself points at the fix (one arena per concurrent job/thread).
        /// </exception>
        internal void EnterMutation()
        {
            AtomicSafetyHandle.CheckWriteAndThrow(Safety);

            if (Interlocked.CompareExchange(ref _busy, 1, 0) != 0)
                throw new System.InvalidOperationException("Arena: concurrent mutating access detected -- two threads/jobs called into the same Arena at the same time. An Arena is single-threaded by contract (see Arena's class doc): allocate everything before scheduling jobs, and give each concurrent job its own arena instead of sharing one.");
        }

        /// <summary>
        /// Marks exit from a guarded mutating body. Every <see cref="EnterMutation"/> call site
        /// pairs this in a <c>finally</c> block, so it still runs if the guarded body throws for
        /// an unrelated reason (e.g. a bad argument) -- otherwise that thread's own legitimate
        /// exception would permanently wedge the tripwire for every later call.
        /// </summary>
        internal void ExitMutation()
        {
            Interlocked.Exchange(ref _busy, 0);
        }
#endif

        /// <summary>
        /// Live allocation count across every tracked family. Record-table-backed families
        /// (float/double, int/short/long/uint, bool, sparse BSR/BlockJacobi) decrement this the
        /// moment an individual instance is Dispose()'d. fProxyBSRBuilders is the one exception:
        /// it still uses the old value-copy tracking list, whose `.Length` only shrinks in bulk on
        /// the next Clear()/ClearTemp() -- a disposed builder stays counted until then. (Bool
        /// allocations were never included in this count at all -- a separate, pre-existing gap.)
        /// </summary>
        public int AllocationsCount =>
            //+copyReplaceFill[+]
            fProxyVecRecords.AliveCount + fProxyMatRecords.AliveCount + fProxyBSRRecords.AliveCount + fProxyBSRBuilders.Length + fProxyBlockJacobiRecords.AliveCount
            //-copyReplaceFill
            +
            //+copyReplaceFill[+]
            iProxyVecRecords.AliveCount + iProxyMatRecords.AliveCount
            //-copyReplaceFill
        ;

        public int TempAllocationsCount =>
            //+copyReplaceFill[+]
            fProxyTempVecRecords.AliveCount + fProxyTempMatRecords.AliveCount
            //-copyReplaceFill
            +
            //+copyReplaceFill[+]
            iProxyTempVecRecords.AliveCount + iProxyTempMatRecords.AliveCount
            //-copyReplaceFill
        ;

        public int AllAllocationsCount => AllocationsCount + TempAllocationsCount;

        public Allocator Allocator;
        public bool Initialized;

        // Pointer-stable allocation-record tables, same
        // design as fProxyVecRecords/iProxyVecRecords above -- just declared directly here (rather
        // than in a per-type Arena.<type>.cs, the way fProxy/iProxy split theirs out) since bool has
        // only one concrete type, so there is no per-type file to split them into (see
        // boolRecords.bool.cs for the record struct definitions themselves). internal (not private):
        // Arena.bool.cs's factory methods on the sibling Arena type reach these directly via
        // _core->BoolVecRecords etc.
        internal ChunkedRecordTable<boolVecRecord> BoolVecRecords;
        internal ChunkedRecordTable<boolMatRecord> BoolMatRecords;
        internal ChunkedRecordTable<boolVecRecord> TempBoolVecRecords;
        internal ChunkedRecordTable<boolMatRecord> TempBoolMatRecords;

        private UnsafeList<Pivot> Pivots;
        private UnsafeList<Indices> IndexBuffers;

        public void Init(Allocator allocator)
        {
            Allocator = allocator;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
            Safety = AtomicSafetyHandle.Create();
#endif

            //+copyReplace
            fProxyVecRecords.Init(Allocator);
            fProxyMatRecords.Init(Allocator);
            fProxyTempVecRecords.Init(Allocator);
            fProxyTempMatRecords.Init(Allocator);
            fProxyBSRRecords.Init(Allocator);
            fProxyBSRBuilders = new UnsafeList<fProxyBSRBuilder>(4, Allocator);
            fProxyBlockJacobiRecords.Init(Allocator);
            //-copyReplace

            //+copyReplace
            iProxyVecRecords.Init(Allocator);
            iProxyMatRecords.Init(Allocator);
            iProxyTempVecRecords.Init(Allocator);
            iProxyTempMatRecords.Init(Allocator);
            //-copyReplace

            BoolVecRecords.Init(Allocator);
            BoolMatRecords.Init(Allocator);

            TempBoolVecRecords.Init(Allocator);
            TempBoolMatRecords.Init(Allocator);

            Pivots = new UnsafeList<Pivot>(2, Allocator);
            IndexBuffers = new UnsafeList<Indices>(4, Allocator);

            // Set LAST, after every tracking list is constructed: a core that threw partway
            // through construction must never read back as Initialized (see Arena's ctor, which
            // frees a partially-constructed core on failure).
            Initialized = true;
        }

        public Pivot Pivot(int size)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            EnterMutation();
            try
            {
#endif
                var pivot = new Pivot(size, this.Allocator);
                Pivots.Add(in pivot);
                return pivot;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { ExitMutation(); }
#endif
        }

        /// <summary>
        /// Allocates a new Indices buffer of length n from this arena.
        /// The arena owns disposal — no manual Dispose needed.
        /// </summary>
        public Indices Indices(int n)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            EnterMutation();
            try
            {
#endif
                var buf = new Indices(n, this.Allocator);
                IndexBuffers.Add(in buf);
                return buf;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { ExitMutation(); }
#endif
        }

        /// <summary>
        /// Disposes every tracked (persistent AND temp) allocation. Public, guarded entry point --
        /// see <see cref="ClearCore"/>/<see cref="ClearTempCore"/> for the actual unguarded work.
        /// </summary>
        public void Clear()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            EnterMutation();
            try
            {
#endif
                ClearCore();
                ClearTempCore();
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { ExitMutation(); }
#endif
        }

        /// <summary>
        /// Disposes every PERSISTENT-pool allocation, then also disposes the temp pool via
        /// <see cref="ClearTempCore"/>. Split out of <see cref="Clear"/> as an UNGUARDED core so
        /// that <see cref="Clear"/>, <see cref="Dispose"/> can each acquire the concurrency guard
        /// exactly ONCE and then call this directly, instead of one guarded method calling
        /// another and tripping the tripwire on itself (see <c>_busy</c>'s doc for why reentrancy
        /// is avoided structurally rather than by counting).
        /// </summary>
        private void ClearCore()
        {
            // Dispose-then-Free here (opposite of fProxyN/fProxyMxN.Dispose()'s Free-then-dispose):
            // safe because this loop is the sole owner walking its own record tables sequentially,
            // with no aliased struct copy that could race in and double-Free the same slot mid-loop.
            //+copyReplace
            for (int i = 0; i < fProxyVecRecords.Count; i++)
                if (fProxyVecRecords.IsAlive(i))
                {
                    fProxyVecRecords.Resolve(i)->Data.Dispose();
                    fProxyVecRecords.Free(i);
                }

            for (int i = 0; i < fProxyMatRecords.Count; i++)
                if (fProxyMatRecords.IsAlive(i))
                {
                    fProxyMatRecords.Resolve(i)->Data.Dispose();
                    fProxyMatRecords.Free(i);
                }

            for (int i = 0; i < fProxyBSRRecords.Count; i++)
                if (fProxyBSRRecords.IsAlive(i))
                {
                    var rec = fProxyBSRRecords.Resolve(i);
                    rec->RowPtr.Dispose();
                    rec->ColInd.Dispose();
                    rec->Values.Dispose();
                    fProxyBSRRecords.Free(i);
                }

            for (int i = 0; i < fProxyBSRBuilders.Length; i++)
                fProxyBSRBuilders[i].Dispose();
            fProxyBSRBuilders.Clear();

            for (int i = 0; i < fProxyBlockJacobiRecords.Count; i++)
                if (fProxyBlockJacobiRecords.IsAlive(i))
                {
                    fProxyBlockJacobiRecords.Resolve(i)->DInv.Dispose();
                    fProxyBlockJacobiRecords.Free(i);
                }
            //-copyReplace

            //+copyReplace
            for (int i = 0; i < iProxyVecRecords.Count; i++)
                if (iProxyVecRecords.IsAlive(i))
                {
                    iProxyVecRecords.Resolve(i)->Data.Dispose();
                    iProxyVecRecords.Free(i);
                }

            for (int i = 0; i < iProxyMatRecords.Count; i++)
                if (iProxyMatRecords.IsAlive(i))
                {
                    iProxyMatRecords.Resolve(i)->Data.Dispose();
                    iProxyMatRecords.Free(i);
                }
            //-copyReplace

            for (int i = 0; i < BoolVecRecords.Count; i++)
                if (BoolVecRecords.IsAlive(i))
                {
                    BoolVecRecords.Resolve(i)->Data.Dispose();
                    BoolVecRecords.Free(i);
                }

            for (int i = 0; i < BoolMatRecords.Count; i++)
                if (BoolMatRecords.IsAlive(i))
                {
                    BoolMatRecords.Resolve(i)->Data.Dispose();
                    BoolMatRecords.Free(i);
                }

            for (int i = 0; i < Pivots.Length; i++)
                Pivots[i].Dispose();
            Pivots.Clear();

            for (int i = 0; i < IndexBuffers.Length; i++)
                IndexBuffers[i].Dispose();
            IndexBuffers.Clear();

            // Calls the TEMP pool's unguarded core directly, NOT the public guarded ClearTemp() --
            // this method is itself only ever reached from inside an already-guarded Clear()/
            // Dispose() body, so calling back into a second EnterMutation() here would trip the
            // tripwire on ourselves (see _busy's doc).
            ClearTempCore();
        }

        /// <summary>
        /// Disposes only temporary allocations, produced from operations. Public, guarded entry
        /// point -- see <see cref="ClearTempCore"/> for the actual unguarded work, which
        /// <see cref="ClearCore"/> also calls directly to avoid nested guard acquisition.
        /// </summary>
        public void ClearTemp()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            EnterMutation();
            try
            {
#endif
                ClearTempCore();
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { ExitMutation(); }
#endif
        }

        /// <summary>
        /// Unguarded temp-pool disposal core -- see <see cref="ClearCore"/>'s doc for why this is
        /// split out (so <see cref="ClearCore"/> and <see cref="ClearTemp"/> can each acquire the
        /// concurrency guard exactly once, instead of nesting it).
        /// </summary>
        private void ClearTempCore()
        {
            // Same dispose-then-Free ordering as Clear() above, and safe for the identical reason:
            // this loop is the sole owner sequentially walking its OWN temp record table, gated by
            // IsAlive(i) -- no aliased struct copy can race in here. See Clear()'s comment (and
            // fProxyN/fProxyMxN.Dispose()'s comment, which needs the OPPOSITE order because it has
            // to guard against exactly that aliasing).
            //+copyReplace
            for (int i = 0; i < fProxyTempVecRecords.Count; i++)
                if (fProxyTempVecRecords.IsAlive(i))
                {
                    fProxyTempVecRecords.Resolve(i)->Data.Dispose();
                    fProxyTempVecRecords.Free(i);
                }

            for (int i = 0; i < fProxyTempMatRecords.Count; i++)
                if (fProxyTempMatRecords.IsAlive(i))
                {
                    fProxyTempMatRecords.Resolve(i)->Data.Dispose();
                    fProxyTempMatRecords.Free(i);
                }
            //-copyReplace

            //+copyReplace
            for (int i = 0; i < iProxyTempVecRecords.Count; i++)
                if (iProxyTempVecRecords.IsAlive(i))
                {
                    iProxyTempVecRecords.Resolve(i)->Data.Dispose();
                    iProxyTempVecRecords.Free(i);
                }

            for (int i = 0; i < iProxyTempMatRecords.Count; i++)
                if (iProxyTempMatRecords.IsAlive(i))
                {
                    iProxyTempMatRecords.Resolve(i)->Data.Dispose();
                    iProxyTempMatRecords.Free(i);
                }
            //-copyReplace

            for (int i = 0; i < TempBoolVecRecords.Count; i++)
                if (TempBoolVecRecords.IsAlive(i))
                {
                    TempBoolVecRecords.Resolve(i)->Data.Dispose();
                    TempBoolVecRecords.Free(i);
                }

            for (int i = 0; i < TempBoolMatRecords.Count; i++)
                if (TempBoolMatRecords.IsAlive(i))
                {
                    TempBoolMatRecords.Resolve(i)->Data.Dispose();
                    TempBoolMatRecords.Free(i);
                }
        }

        /// <summary>
        /// Frees every tracked allocation and the tracking lists themselves, then marks this core
        /// as torn down. Does NOT free the ArenaCore block itself -- that is the owning
        /// <see cref="Arena"/> handle's job. Double-dispose is guarded (best-effort): an upfront
        /// existence check turns disposing the same core twice into a clear exception rather than
        /// silent memory corruption.
        /// </summary>
        public void Dispose()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            if (!AtomicSafetyHandle.IsDefaultValue(Safety))
                AtomicSafetyHandle.CheckExistsAndThrow(Safety);
            EnterMutation();
            try
            {
#endif
                // Calls the unguarded cores directly, NOT the public Clear()/ClearTemp() -- this
                // body already holds the guard (EnterMutation above), so routing back through the
                // guarded public entry points would nest and trip the tripwire on itself.
                ClearCore();
                ClearTempCore();

                //+copyReplace
                fProxyVecRecords.Dispose();
                fProxyMatRecords.Dispose();
                fProxyTempMatRecords.Dispose();
                fProxyTempVecRecords.Dispose();
                fProxyBSRRecords.Dispose();
                fProxyBSRBuilders.Dispose();
                fProxyBlockJacobiRecords.Dispose();
                //-copyReplace

                //+copyReplace
                iProxyVecRecords.Dispose();
                iProxyMatRecords.Dispose();
                iProxyTempMatRecords.Dispose();
                iProxyTempVecRecords.Dispose();
                //-copyReplace

                BoolVecRecords.Dispose();
                BoolMatRecords.Dispose();
                TempBoolMatRecords.Dispose();
                TempBoolVecRecords.Dispose();

                Pivots.Dispose();
                IndexBuffers.Dispose();

                Initialized = false;
                Allocator = Allocator.Invalid;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { ExitMutation(); }

            AtomicSafetyHandle.Release(Safety);
#endif
        }
    }

    /// <summary>
    /// Thin, freely-copyable handle to a heap-allocated <see cref="ArenaCore"/>: every copy shares
    /// exactly ONE core, so arena identity is stable by address. A default(Arena)/standalone handle
    /// has <c>_core == null</c>.
    ///
    /// Exactly one owner must call <see cref="Dispose"/>, exactly once, on the authoritative
    /// handle -- disposing a second copy, or disposing after the original has already been
    /// disposed, double-frees the core block. An Arena is single-threaded by contract: do all
    /// persistent allocation (including <c>Pivot</c>/<c>Indices</c> buffers) before scheduling
    /// jobs; give each concurrently-running job/thread its own Arena; wait for jobs to finish
    /// before calling Clear()/ClearTemp()/Dispose(). Inside a job, use only in-place/ref-destination
    /// APIs on pre-allocated buffers plus <c>Allocator.Temp</c> scratch -- never arena-allocating
    /// operators/Copy()/TempCopy() (<c>var c = a + b;</c> inside a job is an arena mutation from a
    /// worker thread, exactly the race this contract forbids). Under
    /// <c>ENABLE_UNITY_COLLECTIONS_CHECKS</c>, mutating entry points detect (not prevent)
    /// overlapping calls and use-after-Dispose. See Arena/DEVLOG.md for the full design writeup.
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

            // UnsafeUtility.Malloc does not clear memory. MemClear so every ChunkedRecordTable's
            // IsCreated starts false (its Init() guards against double-init by checking that flag,
            // and garbage bytes could otherwise spuriously read back as "already created").
            UnsafeUtility.MemClear(_core, UnsafeUtility.SizeOf<ArenaCore>());

            // Free the Malloc'd block if Init throws instead of leaking it. try/finally, not
            // try/catch, since Burst doesn't support try/catch; under Burst this guard degrades to
            // a no-op on the throw path, but is fully effective for plain managed callers.
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

        /// <summary>
        /// Wraps an EXISTING (already-initialized) ArenaCore -- used internally to reconstruct a
        /// live Arena handle from a record's <c>Owner</c> back-pointer, e.g. fProxyN/fProxyMxN's
        /// <c>Copy()</c>/<c>TempCopy()</c> and their cross-type allocation shortcuts. Does NOT
        /// allocate or own the core: disposing a handle built this way is exactly as safe/unsafe as
        /// disposing any other copy of the SAME live Arena (see the class-level ownership contract
        /// above).
        /// </summary>
        internal Arena(ArenaCore* core)
        {
            _core = core;
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
