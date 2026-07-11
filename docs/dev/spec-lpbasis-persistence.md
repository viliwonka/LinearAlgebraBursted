# Spec: LP warm-solve factor persistence (fProxyLPCache)

## Motivation (measured)

MIP profiling (2026-07-10, stein15/p0033): >98% of wall time is inside per-node LP.solve
calls, and roughly half of each warm call is FIXED SETUP, not pivoting — a full
BuildComputationalForm M rebuild (O(m·N) copy) plus a from-scratch Refactorize (LU of the
m×m basis matrix) whose result we already had in hand when the previous solve returned.
At stein15 a node costs ~230µs of which pivots are ~120µs (16.5 pivots × 7-8µs). The
zero-pivot fast path (222e43f) removed the redundant SECOND refactorize for the
"warm basis already optimal" case (−44% on those calls); this feature generalizes: skip
the FIRST one too whenever the previous solve's factors still describe the current basis
and matrix.

Second payoff, same struct: DSE weight persistence. DualSimplexCore currently seeds
steepest-edge weights w=1 on EVERY call — exact only at the all-logical basis. At a warm
start from a parent basis this silently degrades pricing toward Dantzig for the first
pivots of every node (Sonnet fidelity review, Major). HiGHS reuses weights carried on the
HEkk instance between solves (`status.has_dual_steepest_edge_weights`, HEkkDual.cpp solve
setup); our carrier of between-solve state is exactly this cache.

Target: ~1.5-2× on stein15/p0033 MIPBenchmark wall time. Strong-branch trials (38-57% of
node time) are the best case: each trial changes ONE bound-row rhs of an already-active
row, so a whole node's 2×trials reuse one factorization.

## Design

### New per-dtype struct `fProxyLPCache` (LP.Info stays type-agnostic)

`LPBasis` (LP.Info.cs) is deliberately dtype-free (byte/int buffers, shared by both
generated builds) — do NOT add fProxy fields to it. Instead a new template struct, house
pattern = fProxyCHOPCache / fProxyPCAModel (buffer-carrying, explicit allocator +
Dispose, IsCreated):

```
public struct fProxyLPCache {
    // computational form, persisted across calls
    fProxyMxN M;            // m x N   (N = n + m)
    fProxyN   rhs;          // m       (re-copied from caller's b every call, cheap)
    fProxyN   lower, upper; // N       computational-form bounds
    fProxyN   cost;         // N
    // basis factorization state (exactly DualSimplexCore's locals today)
    fProxyMxN B;            // m x m   LU factors in place
    Pivot     P;
    fProxyMxN etaAlpha;     // REFACTOR_INTERVAL x m
    NativeArray<int> etaRow;
    int       etaCount;
    // DSE weights
    fProxyN   weight;       // m
    // validity
    int  matrixVersion;     // stamp of caller's structure the cached M/factors were built at
    bool factorsValid;      // B/P/eta describe (M, LPBasis.basis) as of matrixVersion
    bool weightsValid;      // weight[] is the terminal state of a solve that ended at LPBasis.basis
}
```

Construction: `new fProxyLPCache(n, m, allocator)` + Dispose. `default` /
`!IsCreated` = "no cache": behavior must be byte-identical to today's path.

### New LP.solve overload

`LP.solve(in A, in b, in c, in senses, ref x, out obj, ref LPBasis basis,
ref fProxyLPCache cache, int maxIter = 0)` — same routing as the existing ref-LPBasis
overload (dual simplex warm path). Existing overloads unchanged and keep today's
behavior exactly (they run with a `default` cache internally or via the current code
path — coder's choice, but NO behavior change without a cache).

### Invalidation contract (caller-stamped, checked where affordable)

- `cache.matrixVersion` is bumped BY THE CALLER whenever A's coefficients or senses
  change (MIP: on inert-bound-row activation/deactivation, i.e. inside
  PushBoundChange/UndoToMarker/ApplyNodeBounds coefficient writes). rhs-only changes
  (bAug writes) do NOT bump it.
- On entry with a valid cache: if the stamp matches and `factorsValid` and
  `LPBasis.basis` is unchanged since the factors were stored (see below), skip
  BuildComputationalForm (patch `rhs` from b only — O(m)) and skip Refactorize (resume
  B/P + eta file). Otherwise: rebuild/refactorize exactly as today, then store.
- Basis-unchanged detection: the solver owns LPBasis mutation during a solve, so it can
  record a monotonically increasing `basisStamp` in both LPBasis-adjacent state and the
  cache at solve end; an external caller that hand-edits basis[] must bump
  matrixVersion (documented). Alternative mechanisms allowed if simpler — the CONTRACT
  is what's fixed: stale state must never be silently trusted; any doubt → cold path.
- Under ENABLE_UNITY_COLLECTIONS_CHECKS (i.e. in the test suite), a full O(m·N)
  verification compare of cached M vs freshly-built M runs on every cache hit and
  throws on mismatch — contract violations fail loudly in tests, cost nothing in
  release.
- eta-file at capacity on entry (etaCount >= REFACTOR_INTERVAL) → refactorize, as the
  in-solve interval logic already does.

### DSE weight seeding (HiGHS parity)

At solve entry, in priority order (mirrors HEkkDual.cpp solve setup):
1. cache.weightsValid AND basis unchanged since stored → seed weight[] from cache
   (HiGHS: reuse carried weights).
2. basis is all-logical → w=1 (exact, both we and HiGHS).
3. otherwise → w=1 approximation, TODAY's documented simplification (HiGHS computes
   exact weights via m BTRANs here; measured too expensive at our sizes — keep the
   documented deviation, taxonomy (a), unless the coder can show a cheap win).
At solve end (any terminal status that leaves basis meaningful): store weight[] +
mark weightsValid.

### xB and rhs

xB must be rebuilt every call (rhs changes between calls) — RebuildXB via the existing
Mmul GEMV path. rhs re-copied from the caller's b unconditionally (O(m)).

### Zero-pivot fast path

Unchanged and composes: with a cache hit the entry refactorize is skipped AND the
zero-pivot exit skip still applies — the best case becomes O(m) setup + pricing pass.

### Optional (coder judgment, only if clean): in-call factor handoff

LP.solve's dual→primal-cleanup composition refactorizes again inside the primal core.
The same cache seam could hand the dual's terminal factors to the cleanup. Do it only
if it falls out naturally; NOT required for acceptance.

## MIP integration

- SearchCore allocates one fProxyLPCache (Temp, per solve; sized mAug×NAug) next to its
  persistent LPBasis and passes it to every node/trial LP.solve.
- Bump matrixVersion at every Aaug coefficient write (bound-row activation/deactivation
  sites). rhs-only bound updates (the common plunge/strong-branch case) leave it alone.
- Expected regime: plunge nodes + strong-branch trials = cache hits; queue jumps that
  flip row activation = refactorize. Both correct; only speed differs.

## Acceptance

1. Full suite green. LP filter + MIP filter green.
2. Correctness A/B: every MIPLIB oracle (stein9/15, p0033) and every MIPTests instance
   returns the SAME objective/status as before the change. Node/iteration counts MAY
   change (better pricing from persisted weights changes trajectories) — any changed
   node-count regression anchor must be re-anchored with before/after values reported
   and justified in the coder report.
3. New tests (test-writer): (a) warm re-solve with cache == without cache: identical
   status/objective on a bound-perturbed re-solve chain; (b) coefficient mutation
   WITHOUT version bump is caught by the checks-build verification compare (throws);
   with the bump it solves correctly; (c) determinism ×2 with cache in the loop;
   (d) an LPBasis+cache pair reused across a shape-compatible different problem falls
   back cold and solves correctly.
4. MIPBenchmark A/B (before/after, same machine): report all cells; expect wall-time
   reduction on stein15/p0033; no cell regresses by more than noise.
5. LPBenchmark warm-resolve section (if present) or a small added section: single warm
   re-solve with cache vs without.

## Constraints (standing)

- Templates are source of truth; regen via Tools/regen.ps1; no literal float outside
  //+choose; no double-only kernels (double only as local scalar control math).
- Fidelity rule (docs + memory): HiGHS behavior is the reference for anything touching
  pricing/factor reuse semantics; deviations documented with taxonomy (a)/(b)/(c).
- Comment style: short factual descriptions, no design essays.
- ONE Unity at a time; every Unity invocation synchronous foreground with a 10-min
  timeout — never background+monitor.
- Agents never commit; the main session commits after gates.
