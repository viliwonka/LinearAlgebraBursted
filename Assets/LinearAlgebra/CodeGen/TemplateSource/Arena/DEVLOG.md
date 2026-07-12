# DEVLOG — Arena
Code comments state contracts only; history lives here (see CLAUDE.md).

## Arena.cs — ArenaCore.Safety field
- 2026-07-11 | Why the AtomicSafetyHandle lives on ArenaCore, not on the Arena handle struct:
  Unity's [NativeContainer] job-reflection protocol requires the safety handle to be a field
  directly on the struct a job captures BY VALUE (see e.g. Unity.Collections' own
  `NativeList<T>.m_Safety`). The struct jobs capture here is `Arena` (a bare `ArenaCore*`), not
  the heap-resident `ArenaCore` -- putting the handle on `Arena` instead would grow
  `sizeof(Arena)` under ENABLE_UNITY_COLLECTIONS_CHECKS and break
  `ArenaLayoutTests.Arena_IsPointerSized`'s unconditional pointer-size pin. Chose to keep Arena
  pointer-sized rather than chase automatic schedule-time rejection; the handle is instead
  manually checked (`AtomicSafetyHandle.CheckWriteAndThrow`/`CheckExistsAndThrow`) at the top of
  every guarded mutating entry point. (was Arena.cs:56-71)

## Arena.cs — Arena struct doc (ownership/threading/two-tier model)
- 2026-07-11 | Full design essay behind the condensed class doc:
  - **Ownership contract:** like a Unity `NativeContainer`, every Arena copy is a view onto the
    same heap-allocated core. Exactly one owner must call Dispose exactly once on the
    authoritative handle -- disposing a second copy, or disposing after the original was already
    disposed, double-frees the core block (undefined behavior).
  - **Threading contract:** an Arena is single-threaded by contract, exactly like Unity's own
    native containers. Allocate before you schedule (do all persistent allocation, including
    Pivot/Indices buffers, on the scheduling thread before handing data derived from it to a job;
    a job's Execute() CAN call arena factories/Clear/ClearTemp since they're Burst-compilable, but
    only if no other thread touches the same arena concurrently). One arena per
    concurrently-running job/thread -- sharing one arena across concurrent jobs lets two
    `ChunkedRecordTable<T>.Allocate`/`Free` calls (or a factory racing a Clear) interleave their
    chunk-directory/free-list mutations and corrupt the arena silently (no exception, just wrong
    answers or a later crash). Complete jobs before Clear()/Dispose() -- never call those while a
    job that still holds this arena is in flight; wait on its JobHandle first.
  - **Detection, not prevention:** under ENABLE_UNITY_COLLECTIONS_CHECKS, every mutating entry
    point is wrapped by an ArenaCore-resident interlocked tripwire that throws
    InvalidOperationException the instant two such calls overlap in time, plus an
    AtomicSafetyHandle that throws if the arena is used after Dispose already released it. Neither
    mechanism makes concurrent access safe -- both only turn an otherwise-silent race into a loud,
    deterministic-ish exception. NOT gated by either mechanism: element reads/writes on the math
    structs themselves, and an individual buffer's own Dispose() (e.g. fProxyN.Dispose()), which
    also mutates a record table but sits at a different altitude than Arena/ArenaCore's own entry
    points (see ArenaCore's `_busy` field doc for that known gap).
  - **The two-tier model:** the arena is the AUTHORING tier -- operators (`a + b`),
    Copy()/TempCopy(), cross-type shortcuts, and the temp pool, all main-thread. Easy arithmetic
    structurally requires the arena: a C# operator receives only its operands, so the result's
    allocator must ride inside them (each struct's internal owner reference). The COMPUTE tier --
    in-place/ref-destination APIs on pre-allocated buffers, plus standalone Allocator.Temp scratch
    -- is the job-safe tier; every kernel in this library lives there. The trap: the arena rides
    invisibly inside every arena-tracked struct, so `var c = a + b;` INSIDE a job is an arena
    mutation from a worker thread -- exactly the race the contract forbids, reached without ever
    passing the arena anywhere.
  (was Arena.cs:538-599)

## Arena.fProxy.cs / Arena.iProxy.cs
- 2026-07-12 | fProxyMat(dim)/iProxyMat(dim) forward to the (rows, cols) overload so the matrix is
  tracked in fProxyMatRecords/iProxyMatRecords: an earlier direct `new fProxyMxN(...)` here was
  untracked and leaked on Dispose. (was Arena.fProxy.cs:131-132, Arena.iProxy.cs:137-138)

## ChunkedRecordTable.cs
- 2026-07-11 | Chunk-sizing rationale: the first chunk holds 8 slots, each subsequent chunk
  doubles the previous chunk's capacity (8, 16, 32, 64, ...). This keeps a small arena
  (README-demo scale) down to one tiny 8-slot Malloc, while a large arena still only needs a
  handful of chunks (10 chunks already covers 8*(2^10-1) = 8184 slots) rather than hundreds of
  separately-Malloc'd ones. A fixed 8-slot-per-chunk scheme was considered and rejected: it wastes
  nothing per chunk either, but a large arena would pay proportionally many more Malloc calls (one
  per 8 records) for no benefit, since chunk lookup cost is already independent of chunk size --
  so there was no offsetting win to justify the extra allocation churn. (was
  ChunkedRecordTable.cs:7-75)
