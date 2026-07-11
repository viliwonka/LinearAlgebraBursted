# Doc/comment quality audit — TemplateSource/OP, files A–L (2026-07-11)

**Summary**
- Files scanned: 39 (.cs templates in `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/`, names A–L); files with findings: 22, clean: 17.
- Counts: WRONG/STALE 1 · TOO-LONG 43 · JARGON 6 · HISTORY 91 · NOISE 0 — 141 findings total.
- Worst offenders: Krylov.fProxy.cs (23), LP.DualSimplex.fProxy.cs (15), LP.RevisedSimplex.fProxy.cs (13), LOBPCG.fProxy.cs (11), Control.fProxy.cs (11).
- Dominant pattern: HISTORY — internal spec/dev-doc pointers (`docs/spec-*.md`, `docs/dev/*.md`, ~30 occurrences), ticket codes (R2/R3/R6a, "Krylov R1", "stage 1/2"), bug/benchmark post-mortems ("was observed to produce a false Infeasible", dated benchmark numbers), and "an earlier version..." narration.
- One factual error found (FFT.Workspace twiddle-memory figure); two internal-agent-workflow leaks ("test-writer's ... check", "see coder report") that must not ship.

Legend: WRONG = WRONG/STALE, TL = TOO-LONG, J = JARGON, H = HISTORY/DEV-SPEAK, N = NOISE.

---

## Blas.Fused.fProxy.cs
- Blas.Fused.fProxy.cs:9 — H — "Krylov R1 fused vector kernels (see docs/draft-spec-krylov-optimization.md)" — replace with "Fused vector kernels for Krylov solvers:".

## CHO.fProxy.cs
- CHO.fProxy.cs:43 — TL — 23-line inline essay (43–65) on right-looking vs left-looking Cholesky and DPOTRF/DPOTF2/DTRSM naming — trim to 2-3 lines on the algorithm choice.
- CHO.fProxy.cs:57 — H — "see docs/dev/level3-blocking-guide.md recipe B" — delete internal doc pointer.
- CHO.fProxy.cs:88 — H — "see docs/dev/level3-blocking-guide.md \"size gate\"" — delete internal doc pointer.
- CHO.fProxy.cs:135 — TL — numbered (1)/(2)/(3) tutorial (135–188) re-deriving the blocked panel/TRSM/SYRK algorithm — condense to a short "blocked panel update (DPOTF2+DTRSM+SYRK)" note.

## CHOP.fProxy.cs
- CHOP.fProxy.cs:72 — H — "that mirror was the cache cliff" — delete the old-perf-problem clause; keep the storage-layout explanation.
- CHOP.fProxy.cs:109 — H — "see docs/dev/level3-blocking-guide.md \"size gate\" ... for the rationale" — delete internal doc pointer.
- CHOP.fProxy.cs:200 — TL — 48-line blocked-path essay (200–247) incl. full two-tier Lucas/Higham walkthrough — keep the one-line dpstrf.f attribution + the (a)/(b) trick in ~10 lines.
- CHOP.fProxy.cs:231 — H — "DEVIATIONS from a literal dpstrf.f port (both under the proven-equivalence / missing-subsystem taxonomy, not invented shortcuts)" and "predating this change" (242) — internal porting-policy language; state the two behavioral differences plainly or delete.

## Control.fProxy.cs
- Control.fProxy.cs:10 — TL — 30-line module banner (10–40) re-deriving the DARE/SDA recurrences and storage policy — condense to algorithm + the never-NaN/last-good-iterate contract.
- Control.fProxy.cs:11 — H — "Discrete-time LQR (docs/spec-lqr.md)." — drop the spec path.
- Control.fProxy.cs:24 — H — "SEE THE ADDENDUM in docs/spec-lqr.md for the fetched source" — replace with one-line attribution "port of Chiang-Fan-Lin Algorithm 2.1".
- Control.fProxy.cs:169 — H — "doubling recursion per the spec addendum" — delete the spec reference.
- Control.fProxy.cs:174 — H — "(see spec-lqr.md test discussion)" — delete.
- Control.fProxy.cs:283 — H — "shared by the plain cold entry point and entry 2's cold fallback" — name the actual overload instead of "entry 2".
- Control.fProxy.cs:305 — H — "test-writer's SDA-vs-oracle check calls this directly" — internal agent/workflow name in source; replace with "exposed for direct testing".
- Control.fProxy.cs:306 — H — "per docs/spec-lqr.md's Tests section" — delete.
- Control.fProxy.cs:365 — H — "see docs/spec-lqr.md's addendum for the exact recurrences and source" (public XML doc) — replace with short attribution.
- Control.fProxy.cs:444 — H — "this is the recursion's test oracle, and the reason it exists as a secondary entry point" — state the finite-horizon feature's purpose plainly.
- Control.fProxy.cs:498 — TL — fProxyLQRState doc (498–508) compares design philosophy to fProxyLPCache/LPBasis — trim to construct-before-use + warm-start-S contract.

## Eigen.fProxy.cs
- Eigen.fProxy.cs:16 — TL — 35-line power-iteration summary mixing contract, "Notes:" list and forwarder-architecture narration — condense to purpose, seeding, convergence criterion, complex-pair limitation.
- Eigen.fProxy.cs:215 — TL — ~65-line inverse-power-iteration essay (design rationale, workspace-reuse discussion, tol/cgTol interplay) — trim to purpose, precondition, scratch layout, and the tol-vs-cgTol gotcha.
- Eigen.fProxy.cs:226 — H — "this is the roadmap's lambda_min capability" — state the use case directly.
- Eigen.fProxy.cs:533 — TL — ~65-line Lanczos doc incl. full Gershgorin-padding rationale — keep purpose, symmetric-only precondition, output convention; move the rationale inline near the code.
- Eigen.fProxy.cs:1140 — H — "(see docs/dev/spec-svd-eigen-convergence.md)" — delete from public doc.

## Eigen.Info.cs
- Eigen.Info.cs:8 — H — "Every converted eigensolver RETURNS this by value... the old success-test call shapes still compile unchanged" — replace with plain "implicitly converts to bool for use in `if (...)`".
- Eigen.Info.cs:81 — H — same "Every converted entry point..." phrasing — same fix.
- Eigen.Info.cs:97 — H — "NO residual field (that is what the test oracles are for, not this struct)" — just state there is no residual field.
- Eigen.Info.cs:103 — H — "Twin of SVDInfo... house pattern for this file is one Info struct per family; see SolveInfo.cs" — drop the file-organization note.
- Eigen.Info.cs:154 — H — same "Every converted overload..." phrasing — same fix.

## FFT.Workspace.fProxy.cs
- FFT.Workspace.fProxy.cs:12 — WRONG — "full table uses ~2× twiddle memory (~16 MB at N=1M for float)" — twRe+twIm at N=1M floats is ~8 MB, not 16 (consistent with the stated ~2× over the half-table's ~4 MB) — fix the number or drop the parenthetical.
- FFT.Workspace.fProxy.cs:133 — TL — `fft` doc restates the full internal dispatch (IsPowerOf4 → radix-4, else pow2 → mixed, else throw) — shorten to "any power-of-two length via the precomputed twiddle table; throws otherwise (use dft)."
- FFT.Workspace.fProxy.cs:170 — TL — `ifft` doc says "same as fft" then re-explains the whole dispatch anyway — delete the restatement; note only the 1/N scaling.

## Krylov.Guards.cs
- Krylov.Guards.cs:11 — H — "See docs/dev/codegen-refactor-lessons.md." — drop the doc-file pointer.
- Krylov.Guards.cs:14 — TL — full RequireDistinctBuffers essay (cross-refs, stackalloc design, `long*` cast) — trim to "Throws if any two of the first `count` pointers are equal (aliasing guard for solver scratch buffers)."

## Krylov.fProxy.cs
- Krylov.fProxy.cs:14 — H — "EXCEPTION (R6a, docs/draft-spec-krylov-optimization.md)" — drop ticket code + spec path; keep the verify-at-exit contract sentence.
- Krylov.fProxy.cs:104 — H — "Krylov R2's ApplyDot... a fused version was tried and measured slower" — drop ticket tag + trial narrative.
- Krylov.fProxy.cs:121 — H — "R6a verify-at-exit: r is recursively updated..." — drop the "R6a" tag.
- Krylov.fProxy.cs:304 — H — repeated "Krylov R2's ApplyDot... measured slower" in pcg — same fix as :104.
- Krylov.fProxy.cs:319 — H — "R6a verify-at-exit -- see cg<TOp>'s matching block" — drop tag.
- Krylov.fProxy.cs:410 — H — "(Krylov R3, docs/draft-spec-krylov-optimization.md)" — drop ticket + spec path.
- Krylov.fProxy.cs:444 — H — "// Phase 3: MINRES..., BiCGSTAB..." — drop the project-roadmap "Phase 3" label.
- Krylov.fProxy.cs:452 — TL — minres doc: 4 paragraphs walking Lanczos recurrence/Givens QR/naming history — trim to purpose + symmetric-only + warm-start + breakdown contract.
- Krylov.fProxy.cs:543 — H — "rounding-only vs the original CopyFrom+divInPlace" — drop the prior-implementation comparison.
- Krylov.fProxy.cs:554 — H — "contract-clean (see spec's buffer-rotation rationale)" — drop spec pointer.
- Krylov.fProxy.cs:583 — H — "vs the original copy+axpy+axpy+divInPlace chain" — drop.
- Krylov.fProxy.cs:667 — TL — biCGStab doc: 3 paragraphs on naming and two-half-step presentation — trim to purpose + contract.
- Krylov.fProxy.cs:905 — TL — cgls doc: 4 paragraphs re-deriving normal-equation math, CGLS-vs-CGNR comparison — trim to purpose + damp contract + breakdown condition.
- Krylov.fProxy.cs:1023 — H — "R6a verify-at-exit..." — drop tag.
- Krylov.fProxy.cs:1140 — TL — cgls(BSR) allocating-overload doc spelling out three sibling overloads' full signatures — trim to one cross-reference.
- Krylov.fProxy.cs:1186 — TL — lsqr doc: 5 paragraphs — trim to purpose + damp/warm-start contract + breakdown condition.
- Krylov.fProxy.cs:1357 — TL — LstsqInfoTracked doc: dense algebraic derivation with embedded CAVEAT paragraph — trim to "recovers plain ‖b−Ax‖ from the damped residual; exact only for damp=0 or cold start".
- Krylov.fProxy.cs:1475 — TL — lsqr(BSR) allocating-overload doc, same cross-reference chain as :1140 — same fix.
- Krylov.fProxy.cs:1522 — TL — lsmr doc: 5 paragraphs — trim as :1186.
- Krylov.fProxy.cs:2021 — TL — cgne doc: 3 paragraphs incl. breakdown-condition proof — trim to purpose + contract.
- Krylov.fProxy.cs:2101 — H — "reordered from the original x-then-Apply-then-r sequence" — drop the prior-shape comparison; keep the aliasing justification.
- Krylov.fProxy.cs:2110 — H — "R6a verify-at-exit -- see cg<TOp>'s matching block" — drop tag.

## LOBPCG.fProxy.cs
- LOBPCG.fProxy.cs:10 — TL — class doc spans 10–83 (~74 lines, eight titled sections + worked code sample) — condense to summary + bullet contract (B must be SPD, ascending order, locking, zero-alloc); move the buckling example to user docs.
- LOBPCG.fProxy.cs:139 — H — "an earlier version used `(i + c*3 + 1) & 3`, which repeats with period 4... EXACTLY rank-deficient for any k > 4" — delete bug history; keep "fixed-seed Random fill avoids periodic degeneracy".
- LOBPCG.fProxy.cs:157 — TL — 13-line inline essay ("harmless busywork on not-yet-meaningful data... DELIBERATELY EUCLIDEAN ONLY") — condense to two sentences.
- LOBPCG.fProxy.cs:282 — H — "...the buckling smoke test hit exactly this in float" — drop the test-narration clause.
- LOBPCG.fProxy.cs:384 — H — "exactly the observed failure mode (residual shrinks nicely for ~15-20 iterations, then stalls...)" — delete; keep the recompute-fresh contract.
- LOBPCG.fProxy.cs:396 — H — "this is what actually produced Ritz values below lambda_min, even wildly negative..." — delete debugging narrative.
- LOBPCG.fProxy.cs:477 — TL — 15-line NOTE justifying a missing overload via CS0111 — shorten to two sentences.
- LOBPCG.fProxy.cs:826 — TL — restates the full row-by-row Cholesky-QR update formula — shorten to one-line contract.
- LOBPCG.fProxy.cs:1042 — J — "hoist their tuning constants into method scope" — "declare".
- LOBPCG.fProxy.cs:1092 — H — "observed: Ritz values... down to -1E13 and beyond... exceeded this envelope by 1E5-1E30x" — delete observed-bug narrative; keep the safeguard rationale.
- LOBPCG.fProxy.cs:1225 — H — "an earlier version did, for AX/AP... was pure wasted work" — delete; state current contract only.

## LOBPCG.Cache.fProxy.cs
- LOBPCG.Cache.fProxy.cs:89 — H — "an earlier version maintained this purely via linearity, which compounded rounding error into a slow convergence stall" — delete; keep "recomputed fresh via A.Apply every iteration".
- LOBPCG.Cache.fProxy.cs:117 — H — "an earlier version mirror-combined AX/AP the same way, but that work was always immediately discarded... dead weight" — keep only "AXnext/APnext are allocated but unused; do not rely on their contents".

## LP.BarrodaleRoberts.fProxy.cs
- LP.BarrodaleRoberts.fProxy.cs:15 — TL — 83-line file banner (15–97) with Source+verification / Deviations / Algorithm-shape sections — cut to ~15 lines: one-line attribution (Barrodale-Roberts 1973 / Koenker-d'Orey rqbr), the two-stage shape, and the numbered deviations compressed to one line each.
- LP.BarrodaleRoberts.fProxy.cs:22 — H — "Transcribed line-by-line from... fetched from https://cdn.jsdelivr.net/gh/cran/quantreg@master/src/rqbr.f (the same mirror pattern that worked for LP.FrischNewton...)" — replace with one-line attribution; drop the fetch narrative.
- LP.BarrodaleRoberts.fProxy.cs:20 — H — "see docs/spec-lad-barrodale-roberts.md" (also :69) — delete spec pointers.
- LP.BarrodaleRoberts.fProxy.cs:106 — H — "(see docs/spec-lad-barrodale-roberts.md test 4)" inside a public XML doc — delete.
- LP.BarrodaleRoberts.fProxy.cs:316 — H — 21-line comment (316–336) narrating the measured quadratic-cost investigation ("measured, not merely suspected... LPTests.fProxy.cs's LadBRvsOracleM192... LPBenchmark's Section 2b (1024-16384), which is exactly where the quadratic behavior was measured") — keep two lines: what the threshold does and why sorting is used above it.
- LP.BarrodaleRoberts.fProxy.cs:464 — TL — 15-line BRPivot paragraph justifying the vectorized split ("BIT-IDENTICAL to the original branchy loop... the right place to spend the vectorization effort") — keep 2-3 lines: the enter-column exception and that the split ranges are exact.

## LP.Cache.fProxy.cs
- LP.Cache.fProxy.cs:7 — TL — 39-line struct doc (7–45) with five titled sections — keep the invalidation contract + lifecycle; cut the rest to a line each.
- LP.Cache.fProxy.cs:9 — H — "(docs/spec-lpbasis-persistence.md)" — delete spec pointer.
- LP.Cache.fProxy.cs:26 — H — "(Two ints rather than the spec sketch's single field... Documented deviation from the sketch...)" — delete the spec-negotiation parenthetical.
- LP.Cache.fProxy.cs:31 — H — "no separate snapshot of LPBasis.basis is kept (spec allows \"alternative mechanisms if simpler\")" — drop the spec-permission clause.

## LP.DualSimplex.fProxy.cs
- LP.DualSimplex.fProxy.cs:10 — TL — ~52-line file banner (10–61), five stacked paragraphs — trim to ~5 lines: what the file implements + one-line HiGHS attribution.
- LP.DualSimplex.fProxy.cs:12 — H — "HiGHS-style dense revised-simplex port (docs/spec-revised-simplex.md)" — drop the spec path.
- LP.DualSimplex.fProxy.cs:36 — H — "verified line-by-line against HiGHS source... HEkk.cpp::updateDualSteepestEdgeWeights... HEkkDual.cpp::updatePrimal" — replace with "DSE update follows HiGHS's Forrest-Goldfarb formula."
- LP.DualSimplex.fProxy.cs:45 — H — "WARM-START (docs/draft-spec-mip.md stage 1, ...)" — drop spec reference.
- LP.DualSimplex.fProxy.cs:56 — H — "FACTOR/WEIGHT PERSISTENCE (docs/spec-lpbasis-persistence.md, ...)" — drop spec reference.
- LP.DualSimplex.fProxy.cs:95 — H — "(was a per-column loop reading M[i,j] with stride N -- the worst pattern for a row-major matrix)" — drop prior-implementation narration.
- LP.DualSimplex.fProxy.cs:124 — TL — 35-line derivation essay on DualRatioTest — condense to a short contract note.
- LP.DualSimplex.fProxy.cs:145 — H — "An earlier version of this method allowed flips to fully resolve a row with no pivot; it passed every test at n<=24 but produced a false Infeasible on a 48-variable random instance" — delete.
- LP.DualSimplex.fProxy.cs:245 — TL — 22-line, 4-paragraph warm-start-overload essay — shorten to the actual contract.
- LP.DualSimplex.fProxy.cs:287 — TL — 19-line cache-aware-overload essay (repeats spec-file ref) — shorten to hit/miss contract.
- LP.DualSimplex.fProxy.cs:357 — H — "...swamps feasTol (~3.45e-4) outright and was observed to produce a false Infeasible within the first few dual iterations" — drop the bug observation; keep the scaling rationale as one line.
- LP.DualSimplex.fProxy.cs:392 — H — "an earlier dualTol-scaled float variant made float B&B trees explode -- benchmark-verified" — delete.
- LP.DualSimplex.fProxy.cs:471 — H — "Benchmark-caught (MIPBenchmark float branchy12: Optimal/216 nodes/10.6ms regressed to NodeLimit/20000 nodes/122.7ms ... see coder report for the isolation test)" — delete; "coder report" is an internal-workflow reference that must not ship. Keep only the seeding-priority rule.
- LP.DualSimplex.fProxy.cs:493 — H — "Using perturbedCost here was an actual bug ... corrupted the warm-started basis... false Unbounded" — delete bug narrative; keep the "use original cost, not perturbedCost" contract line.
- LP.DualSimplex.fProxy.cs:718 — H — "measured ~0.12ms/call -> ~0.06ms/call at mAug~80..., MIP perf investigation 2026-07-10" — delete the benchmark/date narrative; keep the soundness reasoning.

## LP.FrischNewton.fProxy.cs
- LP.FrischNewton.fProxy.cs:16 — TL — 62-line file banner (16–78) — trim to ~15 lines: attribution, the sign-convention warning (genuinely load-bearing), CHOP-not-CHO contract, job-safety.
- LP.FrischNewton.fProxy.cs:23 — H — "Ported and verified line-by-line against... fetched from https://github.com/karenamckinnon/... -- see the derivation trail in docs/spec-lad-frisch-newton.md's authoring history" — replace with one-line attribution.
- LP.FrischNewton.fProxy.cs:43 — H — "verified against LadStackloss's published coefficients in testing" — delete.
- LP.FrischNewton.fProxy.cs:45 — H — "---- Kernel: reuse, per the standing rule ----" — drop the internal-policy phrase.
- LP.FrischNewton.fProxy.cs:52 — H — "CHOP..., NOT plain CHO, per this feature's review -- ...(docs/spec-lad-frisch-newton.md test 4)" — keep the CHOP rationale, drop "per this feature's review" and the spec/test pointer.
- LP.FrischNewton.fProxy.cs:87 — H — "See docs/spec-lad-frisch-newton.md." in public XML doc — delete.
- LP.FrischNewton.fProxy.cs:251 — H — "Fused into ONE pass over m (was two separate loops, each re-reading z[i]/w[i])" — state what it does, not what it was.
- LP.FrischNewton.fProxy.cs:376 — H — BuildATQA paragraph: "BIT-IDENTICAL... just without the aliasing ambiguity that kept Burst from vectorizing the indexer form" — compress to one line; drop before/after framing.
- LP.FrischNewton.fProxy.cs:411 — H — "instead of the two separate loops the reference port used... (see docs/dev/perf-vectorization-lessons.md point 2). BIT-IDENTICAL to the two original loops" — drop history + doc pointer.

## LP.fProxy.cs
- LP.fProxy.cs:72 — TL — 31-line warm-start solve doc (72–103), three dense paragraphs — keep the three-way lifecycle as short bullets; cut the rest.
- LP.fProxy.cs:160 — H — "FACTOR/WEIGHT PERSISTENCE (docs/spec-lpbasis-persistence.md)" in public XML doc — drop the spec path.
- LP.fProxy.cs:266 — H — "Checks-build-only contract verification (docs/spec-lpbasis-persistence.md)" — drop the spec path.
- LP.fProxy.cs:325 — H — "MEASURED, RE-TUNABLE, PER-DTYPE crossover (LPBenchmark Section 2b, 2026-07-09, AFTER the BR sort-path + FN SIMD optimization round): double -- BR wins through m=4096 (2.49ms vs FN 2.71ms)... this is benchmark data, not theory" — replace with one line: "measured per-dtype crossover; re-tune if either engine's per-iteration cost changes."

## LP.Info.cs
- LP.Info.cs:43 — H — "(HiGHS-lineage, stage 1 of docs/spec-revised-simplex.md)" in RevisedSimplex's public enum doc — drop the spec-stage reference.
- LP.Info.cs:50 — H — "(HiGHS-lineage, stage 2 of docs/spec-revised-simplex.md)" in DualSimplex's public enum doc — same fix.
- LP.Info.cs:162 — TL — ~50-line LPBasis doc (162–213) with four titled sections + three-bullet lifecycle — keep the lifecycle bullets (real contract), compress the rest.
- LP.Info.cs:181 — H — "the same mechanism that makes the dual simplex branch-and-bound's workhorse...; see docs/draft-spec-mip.md" — drop the spec pointer.
- LP.Info.cs:313 — H — "after the 2026-07-09 optimization round the measured crossover became per-dtype..." NOTE — delete the dated campaign narration; one line saying the threshold lives inline in LP.fProxy.cs suffices.
- LP.Info.cs:322 — H — "Set comfortably above every m this library's test suite exercises for BR (<=192)...; comfortably below the sizes (1024-16384) where the quadratic behavior was measured" — drop the test/benchmark internals; keep what the gate does.

## LP.InteriorPoint.fProxy.cs
- LP.InteriorPoint.fProxy.cs:268 — H — "(two fProxy4 SIMD accumulators...; see docs/dev/perf-vectorization-lessons.md and the SIMD reduction campaign in git log)... matching every other kernel this campaign has touched" — keep the routed-through-matVecDot fact + the zeroed-first contract; delete campaign/doc references.
- LP.InteriorPoint.fProxy.cs:285 — H — "This loop was ALREADY exactly UnsafeOP.vecMatDot's computation... BIT-IDENTICAL, not a reordering" — state what it does now, drop the before/after framing.

## LP.RevisedSimplex.fProxy.cs
- LP.RevisedSimplex.fProxy.cs:13 — TL — 46-line, 4-paragraph file banner (13–58) — trim to ~5 lines (purpose, computational form, HiGHS attribution).
- LP.RevisedSimplex.fProxy.cs:15 — H — "HiGHS-style dense revised-simplex port (docs/spec-revised-simplex.md)" — drop the spec path.
- LP.RevisedSimplex.fProxy.cs:27 — TL — paragraph re-deriving why tolerance helpers must be inlined (C# overload rules) — cut to one line.
- LP.RevisedSimplex.fProxy.cs:46 — H — "this file used to carry its own hand-written SolveTranspose; it is now just a call to the library primitive" — delete "used to" narration.
- LP.RevisedSimplex.fProxy.cs:56 — J — "...would collide (CS0102) -- see that file's comment" — plain "would collide as a duplicate member".
- LP.RevisedSimplex.fProxy.cs:125 — H — "used to be a per-column loop (j outer, i inner)... matching every other kernel this campaign has touched (see docs/perf-vectorization-lessons.md)" — delete before/after + campaign/doc references.
- LP.RevisedSimplex.fProxy.cs:152 — TL — 11-line cache-locality essay justifying loop order — one line ("i-outer/k-inner avoids column-strided access").
- LP.RevisedSimplex.fProxy.cs:199 — J — "bit-identical to the branchy scalar loop for every i != row -- no reduction, no reassociation" — "matches the scalar loop's result exactly".
- LP.RevisedSimplex.fProxy.cs:217 — H — "same idiom as every other dot product this campaign has touched" — delete.
- LP.RevisedSimplex.fProxy.cs:285 — H — "The nonbasic contribution was originally a per-column scatter... Reshaped into one dense GEMV" (+ spec ref at 299) — drop before/after narration and spec path.
- LP.RevisedSimplex.fProxy.cs:320 — TL — 42-line essay on HarrisRatioTest's far-bound fallback — condense to a short contract note.
- LP.RevisedSimplex.fProxy.cs:338 — H — "Caught by the LP benchmark: ... Reproduced in LPTests.fProxy.cs as RevisedDenseCovering (failed before this fix, passes after)" — delete.
- LP.RevisedSimplex.fProxy.cs:507 — TL — re-explains the same CS0111 rationale already in the file header — delete; point at the header.

## LQ.fProxy.cs
- LQ.fProxy.cs:51 — H — "(see docs/dev/perf-vectorization-lessons.md and the fProxy4 reductions in UnsafeOP)" — delete the doc pointer; keep "dot4 routes through vecDot".
- LQ.fProxy.cs:326 — J — "see dot4's doc comment for why the per-row reduction itself only gets ILP, not full SIMD, in this row-major layout" — in a PUBLIC doc comment; uses "ILP" and appears stale (dot4 now routes through the SIMD vecDot) — delete the parenthetical.
- LQ.fProxy.cs:509 — H — "skipping the ~half-of-runtime Q reconstruction" — unverified profiling factoid in a public API doc — replace with "Q is never materialized; Qᵀ is applied from the stored reflectors."

## LQRP.fProxy.cs
- LQRP.fProxy.cs:22 — TL — class remarks (22–47): multi-paragraph design-rationale essay (no-transpose choice, downdate-vs-recompute, why LEVEL-3 is not mirrored, "primary consumer (rank-deficient IK Jacobians)") — condense to purpose + the basic-vs-min-norm contract (the one part a user needs).
- LQRP.fProxy.cs:25 — H — "same no-transpose choice LQ itself made... (see docs/dev/perf-vectorization-lessons.md)" — delete the doc pointer and comparison.
- LQRP.fProxy.cs:29 — H — "removes the second O(m²n) pass the original exact-recompute cut spent re-summing candidate norms" — drop prior-implementation narration.
- LQRP.fProxy.cs:51 — J — "Four independent accumulators for ILP, mirroring LQ.dot4's rationale (a single running-sum reduction can't be auto-vectorised under strict FloatMode)" — plain wording: "four accumulators so the loop pipelines".
- LQRP.fProxy.cs:139 — J — "Hoisted (n is fixed for the whole call)" — "computed once".

## LU.fProxy.cs
- LU.fProxy.cs:118 — H — "measured crossover, not the naive 4*LU_BLOCK... (see docs/dev/level3-blocking-guide.md \"size gate\")" — delete internal doc pointer.
- LU.fProxy.cs:186 — TL — 26-line blocked-path essay (186–211) with the "WHY THE PIVOT SEQUENCE STAYS IDENTICAL" proof and LAST COLUMN analysis (+ doc pointer at 189) — compress to ~8 lines; the pivot-identity invariant deserves one sentence, not a proof.
- LU.fProxy.cs:414 — TL — 26-line essay (414–439): "The one structural difference is deliberate, not incidental... floating-point summation doesn't know or care whether a row's address came from..." — compress to ~6 lines.
- LU.fProxy.cs:738 — TL — 20-line transposed-solve section header (738–758) incl. revised-simplex BTRAN cross-reference and vectorization rationale — keep the A = Pᵀ L U derivation line + gather/scatter note; cut the rest.

## Clean files
Bidiag.Workspace.fProxy.cs, Bidiag.fProxy.cs, Blas.ColumnScaling.fProxy.cs, Blas.Triangular.fProxy.cs, BoolOP.cs, CHOP.Workspace.fProxy.cs, Control.Info.cs, Easing.fProxy.cs, Eigen.LanczosWorkspace.fProxy.cs, Eigen.SymWorkspace.fProxy.cs, FFT.fProxy.cs, GenOP.fProxy.cs, LOBPCG.Info.cs, LP.Sparse.fProxy.cs, LQ.MinNormWorkspace.fProxy.cs, LQ.Workspace.fProxy.cs, LQRP.Workspace.fProxy.cs
