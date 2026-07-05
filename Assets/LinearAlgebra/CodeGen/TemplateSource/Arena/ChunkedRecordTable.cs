using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.InteropServices;

namespace LinearAlgebra
{
    /// <summary>
    /// A pointer-stable, chunked slot table for arena-owned allocation records
    /// (docs/rfc-memory-model.md §4 Option A / A1, §6.1, §7 step 2). A family-specific record struct
    /// is carved out of a table like this one and addressed by a raw <c>TRecord*</c> that never moves
    /// for the record's lifetime -- so a copy of a math struct holding that pointer can never diverge
    /// from the one source of truth (the RFC's failure modes 1 and 2).
    ///
    /// <para><b>Live, family-by-family.</b> <c>ArenaCore</c> owns one table per migrated family/pool
    /// (float/double's <c>fProxyVecRecords</c>/<c>fProxyMatRecords</c>/temp* -- see
    /// <c>Arena.fProxy.cs</c>, <c>fProxyRecords.fProxy.cs</c>; the int-family and bool
    /// equivalents live in their own sibling Arena partials; the sparse
    /// <c>fProxyBSRRecords</c>/<c>fProxyBlockJacobiRecords</c> -- see <c>Arena.Sparse.fProxy.cs</c>,
    /// <c>fProxyBSRRecords.fProxy.cs</c>): fProxyN/fProxyMxN (and the other migrated types) hold a
    /// stable <c>fProxyVecRecord*</c>/<c>fProxyMatRecord*</c> into one of these tables instead of
    /// being tracked by a separate value copy. <c>Arena.Clear()</c>/<c>ClearTemp()</c> walk a table's
    /// <c>Count</c>/<c>IsAlive</c>/<c>Resolve</c> surface, dispose each alive record's payload, and
    /// <see cref="Free"/> the slot; <c>fProxyN</c>/<c>fProxyMxN.Dispose()</c> does the same for a
    /// single record (see those types' Dispose() for the ordering rationale). Not-yet-migrated:
    /// <c>fProxyBSRBuilder</c> (deliberately -- its own <c>State*</c> indirection already makes a
    /// value-copy tracking list safe), <c>Pivot</c>/<c>Indices</c> (deliberately out of scope --
    /// no arena identity, never grow). Both still use the original growable-UnsafeList-of-value-
    /// copies model and don't touch this table -- see the migration's per-family status in
    /// <c>ArenaCore</c>'s own class doc (<c>Arena.cs</c>). Exercised both end-to-end
    /// (<c>ArenaWiringTests.fProxy.cs</c>) and directly against its own primitives
    /// (<c>ChunkedRecordTableTests</c>).</para>
    ///
    /// <para><b>Storage shape.</b> Records live in fixed-capacity <c>Chunk</c>s, each a single
    /// <c>UnsafeUtility.Malloc</c> block that is <b>never reallocated or moved</b> -- this is exactly
    /// what makes a <c>TRecord*</c> handed out by <see cref="Allocate"/> pointer-stable for the
    /// record's entire lifetime, even while the table keeps growing (the bumpalo "chain of chunks"
    /// pattern, RFC §3.7 / §4 A1). Only the chunk <b>directory</b> (an <c>UnsafeList&lt;Chunk&gt;</c>)
    /// grows, and growing it only copies small <c>Chunk</c> headers (a pointer + two ints) around --
    /// never the chunk memory those headers point at -- so a directory reallocation can never
    /// invalidate a record address.</para>
    ///
    /// <para><b>Chunk sizing.</b> The first chunk holds 8 slots; each subsequent chunk doubles the
    /// previous chunk's capacity (8, 16, 32, 64, ...). This keeps a small arena (README-demo scale --
    /// a handful of allocations) down to one tiny 8-slot Malloc, while a large arena still only needs
    /// a handful of chunks (10 chunks already covers 8*(2^10-1) = 8184 slots) rather than hundreds of
    /// separately-Malloc'd ones. A fixed 8-slot-per-chunk scheme was considered and rejected: it
    /// wastes nothing per chunk either, but a large arena would pay proportionally many more Malloc
    /// calls (one per 8 records) for no benefit, since chunk lookup cost is already independent of
    /// chunk size (see <see cref="SlotPtr"/>) -- so there is no offsetting win to justify the extra
    /// allocation churn.</para>
    ///
    /// <para><b>Free list.</b> <see cref="Free"/> pushes the freed slot's global index onto a free
    /// list; <see cref="Allocate"/> pops from it before ever carving a brand-new slot. This keeps the
    /// table's steady-state size (chunk count) constant across alloc/free churn -- e.g. a per-frame
    /// <c>ClearTemp</c>-shaped loop reuses the same handful of slots forever instead of growing a new
    /// chunk on every frame.</para>
    ///
    /// <para><b>Bookkeeping.</b> Each slot carries <c>Alive</c> + <c>Generation</c> (bumped on
    /// <see cref="Free"/>) alongside its <typeparamref name="TRecord"/> payload. <c>Alive</c> is live
    /// release-code state, consumed via <see cref="IsAlive"/> by <c>Arena.Clear()</c>/<c>ClearTemp()</c>
    /// (skip an already-Dispose()'d record instead of double-freeing it) and guarded by
    /// <see cref="Free"/> itself (rejects a double-Free). <c>Generation</c>, read via
    /// <see cref="GetGeneration"/>, backs the <c>ENABLE_UNITY_COLLECTIONS_CHECKS</c>-only
    /// generational-validation overlay (RFC §6.2, Option B): fProxyMxN/iProxyMxN/boolMxN/fProxyBSR
    /// each carry a small stamp of their own (packed into an existing padding hole -- see those
    /// types' own <c>_gen</c> doc comments) captured at allocation time and compared against this
    /// table's current generation on every read, to catch a stale handle into a since-recycled
    /// slot. fProxyN/iProxyN/boolN/fProxyBlockJacobi have no spare bits for a stamp, so they check
    /// <see cref="IsAlive"/> alone (catches use-after-dispose, not use-after-recycle).</para>
    ///
    /// <para><b>Burst.</b> Unmanaged generic (<c>where TRecord : unmanaged</c>), the one raw pointer
    /// held in a field (<c>Chunk.Slots</c>) is <c>[NativeDisableUnsafePtrRestriction]</c>, no managed
    /// types anywhere -- fully Burst-compilable (exercised inside a <c>[BurstCompile] IJob</c> by
    /// <c>ChunkedRecordTableTests</c>).</para>
    /// </summary>
    internal unsafe struct ChunkedRecordTable<TRecord> : System.IDisposable where TRecord : unmanaged
    {
        // Table-owned bookkeeping alongside the record payload. Kept OUT of TRecord itself so a
        // future family record struct doesn't need to know anything about the table's slot
        // machinery.
        //
        // [StructLayout(Sequential)] + Record FIRST is a HARD CONTRACT, not incidental: a
        // TRecord* handed out by Allocate/Resolve (i.e. the fProxyVecRecord*/fProxyMatRecord*/etc.
        // that fProxyN/fProxyMxN/etc. hold as `_rec`) points at THIS SAME ADDRESS as the Slot that
        // contains it, precisely because Record is the struct's first field at offset 0. That is
        // what makes IsAliveFast/GenerationFast below a valid "container-of" pointer cast
        // (TRecord* -> Slot*) instead of undefined behavior. If Record ever stops being the first
        // field (or a field is inserted before it), that cast reads garbage instead of
        // Alive/Generation -- do not reorder without updating IsAliveFast/GenerationFast too.
        [StructLayout(LayoutKind.Sequential)]
        private struct Slot
        {
            public TRecord Record;
            public int Generation;
            public bool Alive;
        }

        // One Malloc'd, NEVER-moved block of slots. The directory (UnsafeList<Chunk>) may reallocate
        // and move CHUNK HEADERS around, but a header is just {pointer, int, int} -- the Slots block
        // itself, and every TRecord*/Slot* address inside it, is untouched by that.
        private struct Chunk
        {
            [NativeDisableUnsafePtrRestriction]
            public Slot* Slots;
            public int Capacity;
            public int StartIndex; // global slot index of Slots[0]
        }

        private const int FirstChunkCapacity = 8;

        private UnsafeList<Chunk> _chunks;
        private UnsafeList<int> _freeList;
        private Allocator _allocator;

        /// <summary>
        /// Total slots ever carved (alive + freed-but-recyclable) -- the high-water mark for a full
        /// disposal walk: <c>for (int i = 0; i &lt; table.Count; i++) if (table.IsAlive(i)) ...</c>.
        /// This, together with <see cref="IsAlive"/> and <see cref="Resolve"/>, IS the table's
        /// iteration surface -- there is no separate `ForEachAlive` callback API (Burst has no
        /// managed delegates to hang one off), just these three exposed primitives.
        /// </summary>
        public int Count { get; private set; }

        /// <summary>Slots currently allocated (not yet freed).</summary>
        public int AliveCount { get; private set; }

        /// <summary>Number of chunks in the directory. Mostly useful for tests confirming free-list
        /// recycling does not grow the directory.</summary>
        public int ChunkCount => _chunks.Length;

        public void Init(Allocator allocator)
        {
            // Calling Init twice on the same instance without a Dispose in between would overwrite
            // _chunks/_freeList with fresh lists and leak the previous ones (their Malloc'd chunk
            // blocks become unreachable). IsCreated is false both before the first Init and again
            // after a proper Dispose, so this only rejects the genuine double-Init case.
            if (_chunks.IsCreated)
                throw new System.InvalidOperationException("ChunkedRecordTable: Init called twice (or called again without Dispose) -- this would leak the previous chunks.");

            _allocator = allocator;
            _chunks = new UnsafeList<Chunk>(4, allocator);
            _freeList = new UnsafeList<int>(8, allocator);
            Count = 0;
            AliveCount = 0;
        }

        // Cheap (single field check) guard shared by every entry point below: rejects use of a
        // table that was never Init'd, or that has already been Dispose'd -- without it, e.g.
        // Allocate on a disposed table would silently re-Malloc a chunk using the stale _allocator
        // field instead of failing loudly.
        private void EnsureInitialized()
        {
            if (!_chunks.IsCreated)
                throw new System.InvalidOperationException("ChunkedRecordTable: not initialized (Init was never called, or the table was already Disposed).");
        }

        /// <summary>
        /// Carves (or recycles, via the free list) a slot and returns a stable pointer to its
        /// <typeparamref name="TRecord"/> payload, plus the slot index to hand back later to
        /// <see cref="Free"/> / <see cref="Resolve"/>. The returned pointer is valid until the
        /// matching <see cref="Free"/> call and never relocates before then. A freshly-carved (never
        /// before recycled) slot's <typeparamref name="TRecord"/> reads as all-zero; a recycled slot
        /// retains whatever its previous occupant left behind (no poisoning at this stage).
        /// </summary>
        public TRecord* Allocate(out int slotIndex)
        {
            EnsureInitialized();

            int idx;
            if (_freeList.Length > 0)
            {
                idx = _freeList[_freeList.Length - 1];
                _freeList.RemoveAt(_freeList.Length - 1);
            }
            else
            {
                if (LastChunkIsFull())
                    GrowChunk();
                idx = Count;
                Count++;
            }

            Slot* slot = SlotPtr(idx);
            slot->Alive = true;
            slotIndex = idx;
            AliveCount++;
            return &slot->Record;
        }

        /// <summary>
        /// Marks a slot dead and pushes it onto the free list for reuse by a later
        /// <see cref="Allocate"/>. Bumps the slot's <c>Generation</c> -- read back later via
        /// <see cref="GetGeneration"/>/<see cref="GenerationFast"/> by the
        /// <c>ENABLE_UNITY_COLLECTIONS_CHECKS</c> generational-validation overlay (see this class's
        /// own doc, "Bookkeeping" paragraph) to detect a stale handle into a since-recycled slot.
        /// </summary>
        /// <exception cref="System.InvalidOperationException">
        /// The slot is already dead. Unconditional (not DEBUG-gated): a double-Free would otherwise
        /// push the same slot index onto the free list twice, so two LATER <see cref="Allocate"/>
        /// calls would hand out the SAME <typeparamref name="TRecord"/>* to two different callers --
        /// exactly the aliasing/use-after-free bug this table exists to make impossible. This is one
        /// branch on a cold path (Free is called once per allocation, never per element), so there is
        /// no perf case for gating it behind DEBUG.
        /// </exception>
        public void Free(int slotIndex)
        {
            EnsureInitialized();

            Slot* slot = SlotPtr(slotIndex);
            if (!slot->Alive)
                throw new System.InvalidOperationException($"ChunkedRecordTable: double-Free of slot {slotIndex} (already dead).");

            slot->Alive = false;
            slot->Generation++;
            _freeList.Add(slotIndex);
            AliveCount--;
        }

        /// <summary>Resolves a slot index to its stable record pointer without allocating.</summary>
        public TRecord* Resolve(int slotIndex)
        {
            EnsureInitialized();
            return &SlotPtr(slotIndex)->Record;
        }

        /// <summary>True if the slot is currently allocated (not freed).</summary>
        public bool IsAlive(int slotIndex)
        {
            EnsureInitialized();
            return SlotPtr(slotIndex)->Alive;
        }

        /// <summary>Current generation stamp of the slot (starts at 0; bumped by each <see cref="Free"/>).</summary>
        public int GetGeneration(int slotIndex)
        {
            EnsureInitialized();
            return SlotPtr(slotIndex)->Generation;
        }

        /// <summary>
        /// Fast O(1) Alive check, bypassing <see cref="EnsureInitialized"/> and <see cref="SlotPtr"/>'s
        /// chunk scan entirely: <paramref name="rec"/> -- the very <typeparamref name="TRecord"/>*
        /// a math struct's <c>_rec</c> field already holds -- IS a valid <c>Slot*</c> at the exact
        /// same address, because <see cref="Slot"/> is <c>[StructLayout(Sequential)]</c> with
        /// <c>Record</c> as its guaranteed-first field (see that field's own comment). One pointer
        /// read, no index, no lookup.
        ///
        /// <para>Exists for the <c>ENABLE_UNITY_COLLECTIONS_CHECKS</c> generational-overlay guards
        /// (<c>AssertRecordAlive</c>/<c>AssertRecordValid</c> across fProxyN/fProxyMxN/fProxyBSR/
        /// etc.), which run on EVERY guarded getter/property read -- i.e. per element, since an
        /// indexer routes through <c>Data</c>. The index-based <see cref="IsAlive"/> would re-walk
        /// the chunk directory on every one of those reads; this does not.</para>
        /// </summary>
        internal static unsafe bool IsAliveFast(TRecord* rec) => ((Slot*)rec)->Alive;

        /// <summary>Fast O(1) generation read -- see <see cref="IsAliveFast"/> for the container-of
        /// rationale and why the guards use this instead of the index-based <see cref="GetGeneration"/>.</summary>
        internal static unsafe int GenerationFast(TRecord* rec) => ((Slot*)rec)->Generation;

        /// <summary>
        /// Frees every chunk block plus the directory/free-list themselves. Does NOT itself walk or
        /// dispose individual records' payloads -- <typeparamref name="TRecord"/> has no Dispose
        /// contract of its own (it's a plain unmanaged struct, e.g. <c>fProxyVecRecord</c>'s
        /// <c>UnsafeList&lt;fProxy&gt; Data</c>). That is the CALLER's job, done BEFORE this runs:
        /// <c>ArenaCore.Clear()</c>/<c>ClearTemp()</c> walk <c>Count</c>/<c>IsAlive</c>/<c>Resolve</c>,
        /// dispose each alive record's payload and <see cref="Free"/> its slot, and only THEN is this
        /// table itself (now empty of live records) disposed alongside the others in
        /// <c>ArenaCore.Dispose()</c>.
        ///
        /// <para>Idempotent: a second call on an already-disposed (or never-initialized) table is a
        /// safe no-op rather than a double-free of the directory/free-list, mirroring
        /// <see cref="Arena"/>'s own Dispose contract.</para>
        /// </summary>
        public void Dispose()
        {
            if (!_chunks.IsCreated)
                return;

            for (int c = 0; c < _chunks.Length; c++)
                UnsafeUtility.Free(_chunks[c].Slots, _allocator);

            _chunks.Dispose();
            _freeList.Dispose();
            Count = 0;
            AliveCount = 0;
        }

        private bool LastChunkIsFull()
        {
            if (_chunks.Length == 0)
                return true;

            Chunk last = _chunks[_chunks.Length - 1];
            return Count >= last.StartIndex + last.Capacity;
        }

        private void GrowChunk()
        {
            int capacity = FirstChunkCapacity << _chunks.Length; // 8, 16, 32, 64, ...
            // long, not int: capacity * sizeof(Slot) overflows a 32-bit int long before capacity
            // itself (also an int) would, since sizeof(Slot) is >= several bytes per slot.
            long bytes = (long)UnsafeUtility.SizeOf<Slot>() * capacity;
            Slot* slots = (Slot*)UnsafeUtility.Malloc(bytes, UnsafeUtility.AlignOf<Slot>(), _allocator);
            UnsafeUtility.MemClear(slots, bytes); // Alive = false, Generation = 0, Record = default

            var chunk = new Chunk { Slots = slots, Capacity = capacity, StartIndex = Count };
            _chunks.Add(in chunk);
        }

        // Reverse scan: chunks are ordered by ascending StartIndex, so the first chunk (scanning from
        // the END) whose StartIndex <= idx is guaranteed to be the one containing idx -- idx is
        // always a previously-issued slot index, so idx < Count <= (that chunk's StartIndex +
        // Capacity). Chunk counts stay small for the arena sizes this library targets (see the
        // chunk-sizing doc above), so this is effectively O(1) in practice, and it is not a hot path
        // regardless -- it resolves once per Allocate/Free/Resolve call, never per element. This
        // "never per element" claim depends on callers NOT routing the ENABLE_UNITY_COLLECTIONS_
        // CHECKS generational-overlay guards through IsAlive(int)/GetGeneration(int) (both of which
        // call this): those guards fire on every guarded getter/property read (i.e. per element),
        // so they use IsAliveFast/GenerationFast instead -- a direct TRecord*->Slot* cast that
        // bypasses this scan (and EnsureInitialized) entirely. See those methods' doc comments.
        private Slot* SlotPtr(int idx)
        {
            // Single unsigned comparison covers idx < 0 AND idx >= Count in one branch: casting a
            // negative idx to uint wraps to a huge value, which is always >= (uint)Count. Without
            // this, idx >= Count used to fall through the loop below silently and return a pointer
            // into uncarved (or entirely past-the-last-chunk) memory -- a heap-corruption footgun,
            // not just a logic bug, since the caller would then read/write through a bogus pointer.
            if ((uint)idx >= (uint)Count)
                throw new System.ArgumentOutOfRangeException(nameof(idx), $"ChunkedRecordTable: slot index {idx} is out of range (valid range is [0, {Count})).");

            for (int c = _chunks.Length - 1; c >= 0; c--)
            {
                Chunk chunk = _chunks[c];
                if (idx >= chunk.StartIndex)
                    return chunk.Slots + (idx - chunk.StartIndex);
            }

            // Unreachable given the bounds check above (idx < Count guarantees SOME chunk's
            // StartIndex <= idx, by the proof in the comment above) -- this would only fire if the
            // table's own chunk bookkeeping were corrupted, not for any caller-supplied idx.
            throw new System.InvalidOperationException($"ChunkedRecordTable: internal error, no chunk found for in-range index {idx}.");
        }
    }
}
