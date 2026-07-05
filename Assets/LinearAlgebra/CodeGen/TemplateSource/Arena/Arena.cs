using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.InteropServices;
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
    /// Allocator/Initialized. This struct is never copied by user code: it is Malloc'd ONCE per
    /// arena and addressed exclusively through the stable <see cref="Arena"/> handle's <c>_core</c>
    /// pointer, which is what gives the arena a stable identity-by-address (see
    /// docs/rfc-memory-model.md, failure mode 2). Field/method visibility is <c>internal</c> --
    /// only <see cref="Arena"/>'s own partials (same assembly) reach through <c>_core-&gt;</c>;
    /// nothing outside the library touches ArenaCore directly.
    ///
    /// <para><b>Migrated families (float/double, int/short/long/uint, bool, sparse BSR/BlockJacobi)</b>
    /// -- docs/rfc-memory-model.md §4 Option A -- own pointer-stable
    /// <see cref="ChunkedRecordTable{TRecord}"/> tables (<c>fProxyVecRecords</c>/
    /// <c>fProxyMatRecords</c>/temp* -- see <c>fProxyRecords.fProxy.cs</c>, <c>Arena.fProxy.cs</c>;
    /// <c>iProxyVecRecords</c>/<c>iProxyMatRecords</c>/temp* -- see <c>iProxyRecords.iProxy.cs</c>,
    /// <c>Arena.iProxy.cs</c>; <c>BoolVecRecords</c>/<c>BoolMatRecords</c>/temp* -- see
    /// <c>boolRecords.bool.cs</c>, <c>Arena.bool.cs</c>; <c>fProxyBSRRecords</c>/
    /// <c>fProxyBlockJacobiRecords</c> (no temp* -- BSR has no temp-pool variant) -- see
    /// <c>fProxyBSRRecords.fProxy.cs</c>, <c>Arena.Sparse.fProxy.cs</c>): each family's struct holds
    /// a stable record pointer instead of being tracked by a separate value copy, and
    /// Dispose()/Clear()/ClearTemp() free individual slots.
    /// <b>Not-yet-migrated</b>: <c>fProxyBSRBuilders</c>, <c>Pivots</c>, <c>IndexBuffers</c> --
    /// still use the original growable-UnsafeList-of-value-copies model: tracked by <c>.Add</c>,
    /// bulk-freed by <c>.Clear()</c>/<c>.Dispose()</c> on the whole list, with no per-instance
    /// early-dispose bookkeeping. All three stay this way DELIBERATELY, not as an unfinished
    /// migration step: Pivot/Indices have no arena identity to protect and never grow once
    /// allocated (out of scope for this RFC); fProxyBSRBuilder's only mutable-relevant field is
    /// its own heap-Malloc'd <c>State*</c> (see <c>fProxyBSRBuilder.cs</c>), which is already
    /// pointer-stable and identical across every value-copy, so there is no divergence risk (RFC
    /// failure mode 1) left for a record table to fix. See <see cref="AllocationsCount"/>'s doc for
    /// the one user-visible consequence of this mixed model.</para>
    /// </summary>
    internal unsafe partial struct ArenaCore
    {
        /// <summary>
        /// Live allocation count across every tracked family. PERMANENT (not transient) asymmetry
        /// for one deliberately-unmigrated family (docs/rfc-memory-model.md §4 Option A): every
        /// record-table-backed family (float/double, int/short/long/uint, bool, sparse
        /// BSR/BlockJacobi) decrements this THE MOMENT an individual instance is Dispose()'d
        /// (their AliveCount reflects live records exactly); fProxyBSRBuilders alone still uses
        /// the old value-copy tracking list, whose `.Length` only shrinks in bulk on the next
        /// Clear()/ClearTemp() -- an individual disposed builder stays counted until then. This
        /// stays permanent by design: see ArenaCore's class doc for why the builder was left off
        /// the record-table model. (Bool allocations were never included in this count at all,
        /// even before its migration -- a pre-existing gap, unrelated to this asymmetry.)
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

        // Pointer-stable allocation-record tables (docs/rfc-memory-model.md §4 Option A), same
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
            // Ordering note (dispose-then-Free, the OPPOSITE of fProxyN/fProxyMxN.Dispose()'s
            // Free-then-dispose): this loop is the sole owner walking its OWN record tables
            // sequentially, one index at a time, gated by IsAlive(i) -- there is no aliased struct
            // copy that could race in and double-Free the same slot mid-loop, so reading
            // Resolve(i)->Data before marking the slot dead is safe here. fProxyN/fProxyMxN.Dispose()
            // instead has to guard against exactly that aliasing (two struct copies sharing one
            // record), which is why IT frees first and disposes a cached copy of Data second -- see
            // the comment there.
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

            ClearTemp();
        }

        /// <summary>
        /// dispose only temporary allocations, produced from operations
        /// </summary>
        public void ClearTemp()
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
        /// Disposes every tracked element AND the tracking lists themselves, then marks this
        /// core as torn down. Does NOT free the ArenaCore block itself -- that is the owning
        /// <see cref="Arena"/> handle's job (it knows the allocator the block was Malloc'd with).
        /// </summary>
        public void Dispose()
        {
            Clear();

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

            // UnsafeUtility.Malloc does NOT clear memory -- the block starts as garbage bytes.
            // That used to be harmless (every ArenaCore field was unconditionally REASSIGNED in
            // Init() below, so nobody ever read the garbage). It stopped being harmless once
            // ChunkedRecordTable<T> joined the field set (docs/rfc-memory-model.md §4 Option A):
            // its Init() GUARDS against double-init by checking `_chunks.IsCreated`, and garbage
            // bytes can spuriously read back as "already created" -- so the very FIRST Init() call
            // on a freshly Malloc'd core could throw "Init called twice". Zeroing the block first
            // makes every field (old-style lists AND the new tables) start from a clean, honestly
            // "never initialized" state.
            UnsafeUtility.MemClear(_core, UnsafeUtility.SizeOf<ArenaCore>());

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

        /// <summary>
        /// Wraps an EXISTING (already-initialized) ArenaCore -- used internally to reconstruct a
        /// live Arena handle from a record's <c>Owner</c> back-pointer
        /// (docs/rfc-memory-model.md §4 Option A), e.g. fProxyN/fProxyMxN's <c>Copy()</c>/
        /// <c>TempCopy()</c> and their cross-type allocation shortcuts -- replaces the old private
        /// <c>Arena _arena</c> field those used to read directly. Does NOT allocate or own the
        /// core: disposing a handle built this way is exactly as safe/unsafe as disposing any other
        /// copy of the SAME live Arena (see the class-level ownership contract above).
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
