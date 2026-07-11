# Doc/comment audit — OP/ S through Z

Files scanned: 31 (SVD.* ×12, SelectOP.* ×3, SimdMath, SolveInfo, SolveStatus, SwapOP, UnsafeBitsOP.iProxy,
UnsafeBoolOP.* ×3, UnsafeMathOP.* ×2, UnsafeOP.* ×3, UtilityOP, Wave.fProxy, WindowType).
Files with findings: 8 (SVD.fProxy.cs, SVD.Solvers.fProxy.cs, SVD.LowRank.fProxy.cs, SolveInfo.cs, SwapOP.cs,
SimdMath.cs, UnsafeOP.fProxy.cs, UnsafeOP.iProxy.cs). Clean: 23.
Counts: WRONG 1, TOO-LONG 2, JARGON 3, HISTORY 9 (some multi-line), NOISE 1.
Worst offenders: (1) `SolveInfo.cs` `LstsqInfo` doc comment — 30-line essay citing an internal spec file and
a ticket code ("Krylov R6a"); (2) `SVD.fProxy.cs` `thin()`'s workspace-size doc is factually incomplete
(WRONG); (3) `UnsafeOP.fProxy.cs` — several kernel comments narrate benchmark history ("measured ~2x",
"measured NO further gain") instead of just stating the design.

## SVD.fProxy.cs

- `SVD.fProxy.cs:177-178` — WRONG — `"Allocates an n x n + 2*n Temp workspace (plus whatever Bidiag.decomp uses)."` — the method body also allocates `Ut` (n×m) and `Vt` (n×n) Temp buffers (lines 221-222) that aren't in this tally, understating the real footprint (the workspace-cache overload's `fProxySVDThinCache` correctly lists `Ut`/`Vt`, so this allocating-overload doc just wasn't updated to match) — reword to `"Allocates an n x n + 2*n + (n x m) + (n x n) Temp workspace (plus whatever Bidiag.decomp uses)."` or simplify to "Allocates several Temp buffers sized to A's shape."

## SVD.Solvers.fProxy.cs

- `SVD.Solvers.fProxy.cs:29-30` — JARGON — `"Hoist these out of a hot loop solving many same-shape systems to avoid per-call allocs."` — "Hoist" is compiler jargon → `"Move these out of a hot loop..."`

## SVD.LowRank.fProxy.cs

- `SVD.LowRank.fProxy.cs:369` — HISTORY — `"// (byte-identical to the pre-change code path)"` — references a prior version of the code that no longer exists in this file — delete.

## SolveInfo.cs

- `SolveInfo.cs:6-35` — TOO-LONG + HISTORY — the `LstsqInfo` summary is a ~30-line essay: a full code sample, a bulleted per-solver formula list (phibar, Fong-Saunders recurrence, ζ̄), and a citation of an internal spec file and ticket code (see next finding). Far beyond "what it's for + its contract" — trim to: what the struct carries, the implicit-bool contract, and a one-line pointer to `Krylov.lstsqResidual` for an exact residual; drop the per-solver derivation list.
- `SolveInfo.cs:21-22` — HISTORY — `"...EXCEPT cgls's Converged exit (Krylov R6a, docs/draft-spec-krylov-optimization.md): one fresh Apply + ApplyT verifies..."` — internal ticket code + spec file path in a public XML doc comment — reword to `"...except cgls's Converged exit, which reruns Apply + ApplyT once to verify the claimed convergence before trusting it."`
- `SolveInfo.cs:88` — HISTORY — `"cg/pcg/cgne verify a claimed Converged exit with one fresh r = b-Ax first (Krylov R6a);"` — same internal ticket code — delete the `(Krylov R6a)` parenthetical.
- `SolveInfo.cs:63,159,226` — HISTORY — three near-identical sentences: `"...keep compiling after the return type changed from bool to this struct."` — narrates the historical API migration rather than stating the current contract — reword to `"...so `if (solve(...))` still reads as a success test."`

## SwapOP.cs

- `SwapOP.cs:28-29,50-51,73-74` — NOISE — three occurrences of `// do nothing` directly above a bare `return;` inside `if (i == j) { ... }` — adds nothing beyond the code itself — delete.

## SimdMath.cs

- `SimdMath.cs:14` — JARGON (minor) — `"Determinism is unaffected: these are lane-wise abs/max, no reassociation."` — "reassociation" is precise here and the line is short, but per the audit's jargon list it should read in plain words → `"Determinism is unaffected: these are simple lane-wise abs/max, with no reordering of floating-point operations."`

## UnsafeOP.fProxy.cs

- `UnsafeOP.fProxy.cs:16-25` — HISTORY + JARGON — header comment shared by `sumAbs`/`sum`/`maxAbs`/`vecDot`: `"...one 4-lane accumulator left them ~half idle in-cache; the 2nd accumulator measured ~2x..."` plus two uses of "reassociation" — benchmark-campaign narration that belongs in a perf doc, not inline — trim to the determinism contract ("two independent width-4 accumulators; the summation order is fixed for bit-reproducibility") and drop the "measured ~2x" clause.
- `UnsafeOP.fProxy.cs:154-165` — HISTORY — `matVecDot`'s comment: `"TWO fProxy4 accumulators (8 lane-chains): ~2x over a single 4-lane accumulator in-cache... 4 accumulators measured NO further gain (memory/port-bound)."` — same benchmark-narration pattern ("gave Nx over the old X") — trim to stating the chosen width and why (determinism), not the tuning history.
- `UnsafeOP.fProxy.cs:230` — JARGON — `"Tiling only interleaves independent accumulators (ILP across the MR*NR chains)..."` — reword to `"...independent accumulator chains..."`
- `UnsafeOP.fProxy.cs:237` — HISTORY — `"See docs/dev/level3-blocking-guide.md for the blocking background and GemmBenchmark for the sweep."` — internal dev-doc and benchmark-class pointer — delete or move to the dev doc itself.
- `UnsafeOP.fProxy.cs:1218-1234` — TOO-LONG + HISTORY — `sortByKeyAscending`'s comment narrates the routine it replaced: `"...that scan used to repeatedly linear-scan the REMAINING candidates for the current minimum ratio, removing the winner by swap-with-last each round..."` — 17 lines of replaced-algorithm history for a private helper — trim to current behavior + complexity + the non-stable-sort caveat (which is a real contract note worth keeping).

## UnsafeOP.iProxy.cs

- `UnsafeOP.iProxy.cs:216-220` — HISTORY — `"...it used to implement \"v - s\" as \"v + (-s)\" for signed types only... so unifying on the direct kernel avoids needing two branches there."` — describes a prior implementation that no longer exists — trim to `"target[i] -= s. Forward-order twin of the (s, target, n) overload above; used uniformly by subInPlace<T>(T, iProxy) for every generated type."`

## Clean files (no findings)

SVD.FullWorkspace.fProxy.cs, SVD.Metrics.fProxy.cs, SVD.Randomized.fProxy.cs, SVD.RandomizedWorkspace.fProxy.cs, SVD.Subspace.fProxy.cs, SVD.ThinWorkspace.fProxy.cs, SVD.TruncatedWorkspace.fProxy.cs, SVD.ValuesWorkspace.fProxy.cs, SVD.Workspace.fProxy.cs, SelectOP.bool.cs, SelectOP.fProxy.cs, SelectOP.iProxy.cs, SolveStatus.cs, UnsafeBitsOP.iProxy.cs, UnsafeBoolOP.bool.cs, UnsafeBoolOP.fProxy.cs, UnsafeBoolOP.iProxy.cs, UnsafeMathOP.fProxy.cs, UnsafeMathOP.iProxy.cs, UnsafeOP.bool.cs, UtilityOP.cs, Wave.fProxy.cs, WindowType.cs.
