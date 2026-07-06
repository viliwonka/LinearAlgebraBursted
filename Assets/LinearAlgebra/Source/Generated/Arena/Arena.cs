using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.InteropServices;
using System.Threading;
using LinearAlgebra.Sparse;
//singularFile//
namespace LinearAlgebra
{
    /// <summary>
    /// Heap-allocated body holding ALL of an arena's mutable tracking state, plus
    /// Allocator/Initialized. This struct is never copied by user code: it is Malloc'd ONCE per
    /// arena and addressed exclusively through the stable <see cref="Arena"/> handle's <c>_core</c>
    /// pointer, which is what gives the arena a stable identity-by-address (see
    /// docs/dev/rfc-memory-model.md, failure mode 2). Field/method visibility is <c>internal</c> --
    /// only <see cref="Arena"/>'s own partials (same assembly) reach through <c>_core-&gt;</c>;
    /// nothing outside the library touches ArenaCore directly.
    ///
    /// <para><b>Migrated families (float/double, int/short/long/uint, bool, sparse BSR/BlockJacobi)</b>
    /// -- docs/dev/rfc-memory-model.md §4 Option A -- own pointer-stable
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
        // ---- concurrency guards (docs/dev/rfc-memory-model.md's single-threaded-by-contract Arena
        // has never had anything enforcing that contract -- these two mechanisms make a violation
        // detectable instead of silently corrupting the record tables). Both live HERE, inside the
        // heap-Malloc'd ArenaCore, rather than as fields on the pointer-sized Arena handle struct --
        // see Arena's own class doc below ("Threading contract" paragraph) for why, and for the
        // [NativeContainer]/AtomicSafetyHandle protocol conflict that decision resolves.
#if ENABLE_UNITY_COLLECTIONS_CHECKS
        /// <summary>
        /// Dispose-lifetime safety handle. Created once in <see cref="Init"/>, released once in
        /// <see cref="Dispose"/>. Manually checked (via <c>AtomicSafetyHandle.CheckWriteAndThrow</c>/
        /// <c>CheckExistsAndThrow</c>) at the top of every guarded mutating entry point below --
        /// NOT wired into Unity's [NativeContainer] job-reflection protocol, because that protocol
        /// requires the handle to be a field directly on the struct a job captures BY VALUE (see
        /// e.g. Unity.Collections' own <c>NativeList&lt;T&gt;.m_Safety</c>), and the struct jobs
        /// capture here is <see cref="Arena"/> (a bare <c>ArenaCore*</c>), not this heap-resident
        /// <see cref="ArenaCore"/> -- putting the handle on <c>Arena</c> instead would grow
        /// <c>sizeof(Arena)</c> under ENABLE_UNITY_COLLECTIONS_CHECKS and break
        /// ArenaLayoutTests.Arena_IsPointerSized's unconditional pointer-size pin. Chose to keep
        /// Arena pointer-sized rather than chase automatic schedule-time rejection; see Arena's
        /// class doc for the full tradeoff writeup. What this still buys us: a live handle used
        /// (from any thread) after <see cref="Dispose"/> released it throws a clear exception
        /// instead of reading/writing through torn-down state.
        /// </summary>
        internal AtomicSafetyHandle Safety;
#endif

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        /// <summary>
        /// Race tripwire. 0 == free, 1 == a guarded mutating body is currently executing. Not a
        /// mutex -- it never blocks or waits, it only detects two mutating calls overlapping in
        /// time and throws on the second one. Guards ONLY the bodies of the ArenaCore-level
        /// factory/Allocate/Clear/ClearTemp/Dispose/Pivot/Indices entry points enumerated in
        /// docs/features/dense-types.md's threading-contract section.
        ///
        /// <para><b>Known gap, called out deliberately rather than silently:</b> an individual
        /// buffer's OWN <c>Dispose()</c> (e.g. <c>fProxyN.Dispose()</c>/<c>fProxyMxN.Dispose()</c>
        /// and their iProxy/bool/BSR/BlockJacobi counterparts) also mutates a record table --
        /// it calls <c>_rec-&gt;Table-&gt;Free(_rec-&gt;SelfIndex)</c> directly -- but is NOT
        /// guarded here. Two threads freeing different (or the same) allocations from the same
        /// arena concurrently, or one thread freeing while another allocates from the SAME table,
        /// is therefore a real, un-caught race surface, arguably a more common one in practice
        /// than a factory racing another factory. It was scoped out because it sits at a different
        /// altitude (every math struct's own per-instance lifecycle method, not an Arena/ArenaCore
        /// entry point) and because gating it would mean threading a guard through every
        /// numeric/bool/sparse family's Dispose() -- treated here the same as "element reads/
        /// writes on the math structs", which this sweep also left ungated. Flagged as a follow-up
        /// candidate, not implemented.</para>
        ///
        /// <para>Deliberately does NOT attempt same-thread-reentrancy detection via thread
        /// identity (Burst has no <c>Thread.CurrentThread</c>, and the job system's per-worker
        /// <c>JobsUtility.ThreadIndex</c> is only meaningful inside a scheduled job, not from
        /// ordinary managed callers). Instead, reentrancy is avoided STRUCTURALLY: of the handful
        /// of factory overloads that forward to another guarded overload (e.g.
        /// <c>fProxyMat(int,bool)</c> -&gt; <c>fProxyMat(int,int,bool)</c>), only the terminal
        /// overload that actually touches a record table is guarded -- the forwarding wrapper
        /// itself does not call <see cref="EnterMutation"/>, so nesting never happens. Audited
        /// (2026-07-05) across every <c>Arena.*.cs</c>/<c>Arena.Sparse.*.cs</c> factory and every
        /// <c>ArenaExtensions.*.cs</c> convenience method built on top of them: extension methods
        /// only ever call base factories SEQUENTIALLY (each one fully enters and exits its own
        /// guard before the next starts), never while still inside another guarded call.</para>
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
        /// Live allocation count across every tracked family. PERMANENT (not transient) asymmetry
        /// for one deliberately-unmigrated family (docs/dev/rfc-memory-model.md §4 Option A): every
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
            
            floatVecRecords.AliveCount + floatMatRecords.AliveCount + floatBSRRecords.AliveCount + floatBSRBuilders.Length + floatBlockJacobiRecords.AliveCount
            +
            doubleVecRecords.AliveCount + doubleMatRecords.AliveCount + doubleBSRRecords.AliveCount + doubleBSRBuilders.Length + doubleBlockJacobiRecords.AliveCount
            
            +
            
            intVecRecords.AliveCount + intMatRecords.AliveCount
            +
            shortVecRecords.AliveCount + shortMatRecords.AliveCount
            +
            longVecRecords.AliveCount + longMatRecords.AliveCount
            +
            uintVecRecords.AliveCount + uintMatRecords.AliveCount
            
        ;

        public int TempAllocationsCount =>
            
            floatTempVecRecords.AliveCount + floatTempMatRecords.AliveCount
            +
            doubleTempVecRecords.AliveCount + doubleTempMatRecords.AliveCount
            
            +
            
            intTempVecRecords.AliveCount + intTempMatRecords.AliveCount
            +
            shortTempVecRecords.AliveCount + shortTempMatRecords.AliveCount
            +
            longTempVecRecords.AliveCount + longTempMatRecords.AliveCount
            +
            uintTempVecRecords.AliveCount + uintTempMatRecords.AliveCount
            
        ;

        public int AllAllocationsCount => AllocationsCount + TempAllocationsCount;

        public Allocator Allocator;
        public bool Initialized;

        // Pointer-stable allocation-record tables (docs/dev/rfc-memory-model.md §4 Option A), same
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

            
            floatVecRecords.Init(Allocator);
            floatMatRecords.Init(Allocator);
            floatTempVecRecords.Init(Allocator);
            floatTempMatRecords.Init(Allocator);
            floatBSRRecords.Init(Allocator);
            floatBSRBuilders = new UnsafeList<floatBSRBuilder>(4, Allocator);
            floatBlockJacobiRecords.Init(Allocator);
            
            doubleVecRecords.Init(Allocator);
            doubleMatRecords.Init(Allocator);
            doubleTempVecRecords.Init(Allocator);
            doubleTempMatRecords.Init(Allocator);
            doubleBSRRecords.Init(Allocator);
            doubleBSRBuilders = new UnsafeList<doubleBSRBuilder>(4, Allocator);
            doubleBlockJacobiRecords.Init(Allocator);
            

            
            intVecRecords.Init(Allocator);
            intMatRecords.Init(Allocator);
            intTempVecRecords.Init(Allocator);
            intTempMatRecords.Init(Allocator);
            
            shortVecRecords.Init(Allocator);
            shortMatRecords.Init(Allocator);
            shortTempVecRecords.Init(Allocator);
            shortTempMatRecords.Init(Allocator);
            
            longVecRecords.Init(Allocator);
            longMatRecords.Init(Allocator);
            longTempVecRecords.Init(Allocator);
            longTempMatRecords.Init(Allocator);
            
            uintVecRecords.Init(Allocator);
            uintMatRecords.Init(Allocator);
            uintTempVecRecords.Init(Allocator);
            uintTempMatRecords.Init(Allocator);
            

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
        /// Disposes every PERSISTENT-pool allocation (leaves temp pools alone -- see
        /// <see cref="ClearTempCore"/>). Split out of <see cref="Clear"/> as an UNGUARDED core so
        /// that <see cref="Clear"/>, <see cref="Dispose"/> can each acquire the concurrency guard
        /// exactly ONCE and then call this directly, instead of one guarded method calling
        /// another and tripping the tripwire on itself (see <c>_busy</c>'s doc for why reentrancy
        /// is avoided structurally rather than by counting).
        /// </summary>
        private void ClearCore()
        {
            // Ordering note (dispose-then-Free, the OPPOSITE of fProxyN/fProxyMxN.Dispose()'s
            // Free-then-dispose): this loop is the sole owner walking its OWN record tables
            // sequentially, one index at a time, gated by IsAlive(i) -- there is no aliased struct
            // copy that could race in and double-Free the same slot mid-loop, so reading
            // Resolve(i)->Data before marking the slot dead is safe here. fProxyN/fProxyMxN.Dispose()
            // instead has to guard against exactly that aliasing (two struct copies sharing one
            // record), which is why IT frees first and disposes a cached copy of Data second -- see
            // the comment there.
            
            for (int i = 0; i < floatVecRecords.Count; i++)
                if (floatVecRecords.IsAlive(i))
                {
                    floatVecRecords.Resolve(i)->Data.Dispose();
                    floatVecRecords.Free(i);
                }

            for (int i = 0; i < floatMatRecords.Count; i++)
                if (floatMatRecords.IsAlive(i))
                {
                    floatMatRecords.Resolve(i)->Data.Dispose();
                    floatMatRecords.Free(i);
                }

            for (int i = 0; i < floatBSRRecords.Count; i++)
                if (floatBSRRecords.IsAlive(i))
                {
                    var rec = floatBSRRecords.Resolve(i);
                    rec->RowPtr.Dispose();
                    rec->ColInd.Dispose();
                    rec->Values.Dispose();
                    floatBSRRecords.Free(i);
                }

            for (int i = 0; i < floatBSRBuilders.Length; i++)
                floatBSRBuilders[i].Dispose();
            floatBSRBuilders.Clear();

            for (int i = 0; i < floatBlockJacobiRecords.Count; i++)
                if (floatBlockJacobiRecords.IsAlive(i))
                {
                    floatBlockJacobiRecords.Resolve(i)->DInv.Dispose();
                    floatBlockJacobiRecords.Free(i);
                }
            
            for (int i = 0; i < doubleVecRecords.Count; i++)
                if (doubleVecRecords.IsAlive(i))
                {
                    doubleVecRecords.Resolve(i)->Data.Dispose();
                    doubleVecRecords.Free(i);
                }

            for (int i = 0; i < doubleMatRecords.Count; i++)
                if (doubleMatRecords.IsAlive(i))
                {
                    doubleMatRecords.Resolve(i)->Data.Dispose();
                    doubleMatRecords.Free(i);
                }

            for (int i = 0; i < doubleBSRRecords.Count; i++)
                if (doubleBSRRecords.IsAlive(i))
                {
                    var rec = doubleBSRRecords.Resolve(i);
                    rec->RowPtr.Dispose();
                    rec->ColInd.Dispose();
                    rec->Values.Dispose();
                    doubleBSRRecords.Free(i);
                }

            for (int i = 0; i < doubleBSRBuilders.Length; i++)
                doubleBSRBuilders[i].Dispose();
            doubleBSRBuilders.Clear();

            for (int i = 0; i < doubleBlockJacobiRecords.Count; i++)
                if (doubleBlockJacobiRecords.IsAlive(i))
                {
                    doubleBlockJacobiRecords.Resolve(i)->DInv.Dispose();
                    doubleBlockJacobiRecords.Free(i);
                }
            

            
            for (int i = 0; i < intVecRecords.Count; i++)
                if (intVecRecords.IsAlive(i))
                {
                    intVecRecords.Resolve(i)->Data.Dispose();
                    intVecRecords.Free(i);
                }

            for (int i = 0; i < intMatRecords.Count; i++)
                if (intMatRecords.IsAlive(i))
                {
                    intMatRecords.Resolve(i)->Data.Dispose();
                    intMatRecords.Free(i);
                }
            
            for (int i = 0; i < shortVecRecords.Count; i++)
                if (shortVecRecords.IsAlive(i))
                {
                    shortVecRecords.Resolve(i)->Data.Dispose();
                    shortVecRecords.Free(i);
                }

            for (int i = 0; i < shortMatRecords.Count; i++)
                if (shortMatRecords.IsAlive(i))
                {
                    shortMatRecords.Resolve(i)->Data.Dispose();
                    shortMatRecords.Free(i);
                }
            
            for (int i = 0; i < longVecRecords.Count; i++)
                if (longVecRecords.IsAlive(i))
                {
                    longVecRecords.Resolve(i)->Data.Dispose();
                    longVecRecords.Free(i);
                }

            for (int i = 0; i < longMatRecords.Count; i++)
                if (longMatRecords.IsAlive(i))
                {
                    longMatRecords.Resolve(i)->Data.Dispose();
                    longMatRecords.Free(i);
                }
            
            for (int i = 0; i < uintVecRecords.Count; i++)
                if (uintVecRecords.IsAlive(i))
                {
                    uintVecRecords.Resolve(i)->Data.Dispose();
                    uintVecRecords.Free(i);
                }

            for (int i = 0; i < uintMatRecords.Count; i++)
                if (uintMatRecords.IsAlive(i))
                {
                    uintMatRecords.Resolve(i)->Data.Dispose();
                    uintMatRecords.Free(i);
                }
            

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
            
            for (int i = 0; i < floatTempVecRecords.Count; i++)
                if (floatTempVecRecords.IsAlive(i))
                {
                    floatTempVecRecords.Resolve(i)->Data.Dispose();
                    floatTempVecRecords.Free(i);
                }

            for (int i = 0; i < floatTempMatRecords.Count; i++)
                if (floatTempMatRecords.IsAlive(i))
                {
                    floatTempMatRecords.Resolve(i)->Data.Dispose();
                    floatTempMatRecords.Free(i);
                }
            
            for (int i = 0; i < doubleTempVecRecords.Count; i++)
                if (doubleTempVecRecords.IsAlive(i))
                {
                    doubleTempVecRecords.Resolve(i)->Data.Dispose();
                    doubleTempVecRecords.Free(i);
                }

            for (int i = 0; i < doubleTempMatRecords.Count; i++)
                if (doubleTempMatRecords.IsAlive(i))
                {
                    doubleTempMatRecords.Resolve(i)->Data.Dispose();
                    doubleTempMatRecords.Free(i);
                }
            

            
            for (int i = 0; i < intTempVecRecords.Count; i++)
                if (intTempVecRecords.IsAlive(i))
                {
                    intTempVecRecords.Resolve(i)->Data.Dispose();
                    intTempVecRecords.Free(i);
                }

            for (int i = 0; i < intTempMatRecords.Count; i++)
                if (intTempMatRecords.IsAlive(i))
                {
                    intTempMatRecords.Resolve(i)->Data.Dispose();
                    intTempMatRecords.Free(i);
                }
            
            for (int i = 0; i < shortTempVecRecords.Count; i++)
                if (shortTempVecRecords.IsAlive(i))
                {
                    shortTempVecRecords.Resolve(i)->Data.Dispose();
                    shortTempVecRecords.Free(i);
                }

            for (int i = 0; i < shortTempMatRecords.Count; i++)
                if (shortTempMatRecords.IsAlive(i))
                {
                    shortTempMatRecords.Resolve(i)->Data.Dispose();
                    shortTempMatRecords.Free(i);
                }
            
            for (int i = 0; i < longTempVecRecords.Count; i++)
                if (longTempVecRecords.IsAlive(i))
                {
                    longTempVecRecords.Resolve(i)->Data.Dispose();
                    longTempVecRecords.Free(i);
                }

            for (int i = 0; i < longTempMatRecords.Count; i++)
                if (longTempMatRecords.IsAlive(i))
                {
                    longTempMatRecords.Resolve(i)->Data.Dispose();
                    longTempMatRecords.Free(i);
                }
            
            for (int i = 0; i < uintTempVecRecords.Count; i++)
                if (uintTempVecRecords.IsAlive(i))
                {
                    uintTempVecRecords.Resolve(i)->Data.Dispose();
                    uintTempVecRecords.Free(i);
                }

            for (int i = 0; i < uintTempMatRecords.Count; i++)
                if (uintTempMatRecords.IsAlive(i))
                {
                    uintTempMatRecords.Resolve(i)->Data.Dispose();
                    uintTempMatRecords.Free(i);
                }
            

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
        ///
        /// <para>Guarded like every other mutating entry point (see <c>_busy</c>/<c>Safety</c>'s
        /// docs), PLUS an upfront existence check that fires if THIS SAME core was already
        /// Dispose()'d once before (the documented "disposing a second copy... double-frees the
        /// core block" footgun on <see cref="Arena"/>'s own class doc) -- turning what used to be
        /// silent memory corruption into a clear exception, best-effort only: it depends on the
        /// already-freed block's bytes not yet having been reused (reading <c>Safety</c> through a
        /// freed <c>_core</c> pointer is technically undefined behavior, same as any other field
        /// read on a disposed core -- this is a diagnostic aid, not a guarantee).</para>
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

                
                floatVecRecords.Dispose();
                floatMatRecords.Dispose();
                floatTempMatRecords.Dispose();
                floatTempVecRecords.Dispose();
                floatBSRRecords.Dispose();
                floatBSRBuilders.Dispose();
                floatBlockJacobiRecords.Dispose();
                
                doubleVecRecords.Dispose();
                doubleMatRecords.Dispose();
                doubleTempMatRecords.Dispose();
                doubleTempVecRecords.Dispose();
                doubleBSRRecords.Dispose();
                doubleBSRBuilders.Dispose();
                doubleBlockJacobiRecords.Dispose();
                

                
                intVecRecords.Dispose();
                intMatRecords.Dispose();
                intTempMatRecords.Dispose();
                intTempVecRecords.Dispose();
                
                shortVecRecords.Dispose();
                shortMatRecords.Dispose();
                shortTempMatRecords.Dispose();
                shortTempVecRecords.Dispose();
                
                longVecRecords.Dispose();
                longMatRecords.Dispose();
                longTempMatRecords.Dispose();
                longTempVecRecords.Dispose();
                
                uintVecRecords.Dispose();
                uintMatRecords.Dispose();
                uintTempMatRecords.Dispose();
                uintTempVecRecords.Dispose();
                

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
    /// Thin, freely-copyable handle to a heap-allocated <see cref="ArenaCore"/>: copying an Arena
    /// (by value, through `in`/`ref` parameters, as a struct field on fProxyMxN/fProxyN/etc., ...)
    /// only copies the <c>_core</c> pointer, so every copy shares exactly ONE core -- this is what
    /// makes arena identity stable-by-address (a defensive copy the compiler makes for an `in
    /// Arena` parameter can no longer dangle, since it still points at the same live core; see
    /// docs/dev/rfc-memory-model.md, failure mode 2 -- retires the old "must take `ref Arena`, not `in
    /// Arena`" convention). A default(Arena)/standalone handle has <c>_core == null</c>.
    ///
    /// <para><b>Ownership contract:</b> like a Unity <c>NativeContainer</c>, every copy is a view
    /// onto the SAME heap-allocated core. Exactly ONE owner must call <see cref="Dispose"/>,
    /// exactly once, on the authoritative handle -- disposing a second copy, or disposing after the
    /// original has already been disposed, double-frees the core block (undefined behavior).</para>
    ///
    /// <para><b>Threading contract (authoritative statement):</b> an <c>Arena</c> is
    /// single-threaded BY CONTRACT -- exactly like Unity's own native containers, nothing about it
    /// is safe to touch from two threads/jobs at once. Concretely:
    /// <list type="bullet">
    /// <item><description><b>Allocate before you schedule.</b> Do all of an arena's persistent
    /// allocations (and any <c>Pivot</c>/<c>Indices</c> buffers a factorization needs) on the
    /// scheduling thread BEFORE handing data derived from it to a job. A job's <c>Execute()</c>
    /// CAN call arena factories/Clear/ClearTemp (they are Burst-compilable), but only if no OTHER
    /// thread is touching the SAME arena at the same time -- see the next point.</description></item>
    /// <item><description><b>One arena per concurrently-running job/thread.</b> If two jobs may run
    /// at the same time (no dependency between them), each needs its OWN <c>Arena</c> instance.
    /// Sharing one arena across concurrent jobs is exactly the bug this contract exists to name --
    /// two <c>ChunkedRecordTable&lt;T&gt;.Allocate</c>/<c>Free</c> calls (or a factory racing a
    /// <c>Clear</c>) interleaving their chunk-directory/free-list mutations corrupts the arena
    /// silently, no exception, just wrong answers or a crash later.</description></item>
    /// <item><description><b>Complete jobs before Clear()/Dispose().</b> Never call
    /// <c>Clear()</c>/<c>ClearTemp()</c>/<c>Dispose()</c> while a job that still holds (or might
    /// still schedule work against) this arena is in flight. Wait on the job's <c>JobHandle</c>
    /// first.</description></item>
    /// </list>
    /// Detection, not prevention: under <c>ENABLE_UNITY_COLLECTIONS_CHECKS</c>, every mutating
    /// entry point (the factory methods, <c>Clear</c>, <c>ClearTemp</c>, <c>Dispose</c>,
    /// <c>Pivot</c>, <c>Indices</c>) is wrapped by an <see cref="ArenaCore"/>-resident interlocked
    /// tripwire that throws <see cref="System.InvalidOperationException"/> the instant two such
    /// calls overlap in time, plus an <c>AtomicSafetyHandle</c> that throws if the arena is used
    /// after <see cref="Dispose"/> already released it. Neither mechanism makes concurrent access
    /// SAFE -- both only turn an otherwise-silent race into a loud, deterministic-ish exception (see
    /// <c>ArenaCore</c>'s own field docs, and docs/features/dense-types.md, for the full writeup
    /// including why this is NOT wired into Unity's automatic [NativeContainer] job-scheduling
    /// enforcement). NOT gated by either mechanism, out of scope for this pass: element
    /// reads/writes on the math structs themselves (fProxyN/fProxyMxN/etc. indexers -- same as the
    /// rest of this library's per-buffer safety story), AND an individual buffer's own
    /// <c>Dispose()</c> (e.g. <c>fProxyN.Dispose()</c>), which also mutates a record table but sits
    /// at a different altitude than Arena/ArenaCore's own entry points -- see <c>_busy</c>'s field
    /// doc (ArenaCore, above) for that known gap.</para>
    ///
    /// <para><b>The two-tier model (which API belongs on which thread):</b> the arena is the
    /// AUTHORING tier -- operators (<c>a + b</c>), <c>Copy()</c>/<c>TempCopy()</c>, cross-type
    /// shortcuts, and the temp pool, all main-thread. Easy arithmetic structurally REQUIRES the
    /// arena: a C# operator receives only its operands, so the result's allocator must ride inside
    /// them (that is what each struct's internal owner reference is). The COMPUTE tier -- in-place/
    /// ref-destination APIs on pre-allocated buffers, plus standalone <c>Allocator.Temp</c> scratch
    /// -- is the job-safe tier; every kernel in this library lives there. The trap: the arena rides
    /// invisibly inside every arena-tracked struct, so <c>var c = a + b;</c> INSIDE a job is an
    /// arena mutation from a worker thread -- exactly the race the contract above forbids, reached
    /// without ever passing the arena anywhere. Full writeup: docs/features/dense-types.md,
    /// "The two-tier model".</para>
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
            // ChunkedRecordTable<T> joined the field set (docs/dev/rfc-memory-model.md §4 Option A):
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
        /// (docs/dev/rfc-memory-model.md §4 Option A), e.g. fProxyN/fProxyMxN's <c>Copy()</c>/
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
