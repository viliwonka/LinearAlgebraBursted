# Doc/comment quality audit — TemplateSource/OP, files M–R

- Files scanned: 35 (every .cs in `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/` starting M–R)
- Files with findings: 16; clean: 19
- Counts: WRONG 0, TOO-LONG 25, HISTORY 27, JARGON 4, NOISE 2 — 58 findings total (plus ~35 recurring `STAGE n (docs/draft-spec-qp.md)` tags in QP.fProxy.cs counted once)
- Worst offenders: QP.fProxy.cs (9 findings + pervasive stage/spec tagging), QRCP.fProxy.cs (10), OP.Component.fProxy.cs (6), MIP.fProxy.cs (5), MIP.Domain.fProxy.cs (5)
- Dominant patterns: internal spec-file citations (`docs/draft-spec-*.md`, `OQ-7`, `P2/P3`, `STAGE n`) leaking into public docs; multi-paragraph derivation essays as comments; dev-process narration ("third-review finding", "measured worse, reverted", "fetched and read 2026-07-09")

## MIP.Domain.fProxy.cs
- `MIP.Domain.fProxy.cs:20` — HISTORY — "reproduced on the Gomory/Wolsey instance: sentinel=1e30 -> false Infeasible after 63 pivots" — keep the invariant ("a large sentinel rhs can corrupt dual-simplex scaling"), drop the repro narrative.
- `MIP.Domain.fProxy.cs:49` — HISTORY — "(docs/spec-lpbasis-persistence.md: rhs-only bound updates, the common case, leave it alone)" — inline the one-sentence rule; drop the spec citation.
- `MIP.Domain.fProxy.cs:114` — TOO-LONG — `ApplyNodeBounds` multi-paragraph doc duplicating `PushBoundChange`/`UndoToMarker` rationale — trim to what it overwrites + the "always forces cold rebuild" contract.
- `MIP.Domain.fProxy.cs:154` — TOO-LONG/HISTORY — `PropagateFixpoint` doc line-maps to HiGHS internals ("mip/HighsDomain.cpp's propagate/propagateRowUpper... ninfmin/ninfmax...") — condense to 2-3 lines: worklist propagation, returns false on proven infeasibility.
- `MIP.Domain.fProxy.cs:330` — TOO-LONG — "shared with the branching/strong-branch call sites (out of this stage's scope to change), so the same check is done here..." — state what the O(n) sweep catches; drop the scope narration.

## MIP.Pseudocost.fProxy.cs
- `MIP.Pseudocost.fProxy.cs:50` — TOO-LONG/JARGON — full HiGHS `getScore` formula plus "OMITTED (fidelity taxonomy -- subsystems this stage does not have)" listing conflictScore/cutoffScore/... — condense to "product-rule branching score (port of HiGHS getScore); some HiGHS terms not implemented".

## MIP.fProxy.cs
- `MIP.fProxy.cs:11` — TOO-LONG — 30+ line file-header architecture essay citing docs/draft-spec-mip.md repeatedly — trim to a short paragraph + the finite-xl contract.
- `MIP.fProxy.cs:93` — HISTORY — "(docs/draft-spec-mip.md: pseudocost + reliability branching, ...)" inside the public `MIP.solve` XML doc — remove spec-file citation; state the features plainly.
- `MIP.fProxy.cs:275` — HISTORY — "Factor/weight persistence cache (docs/spec-lpbasis-persistence.md)" — state the cache's purpose inline; drop citation.
- `MIP.fProxy.cs:322` — HISTORY — "(docs/draft-spec-mip.md open question 6)" — replace with "fixed internal seed keeps solves deterministic".
- `MIP.fProxy.cs:541` — TOO-LONG/HISTORY — 19-line `TryRoundingHeuristic` essay: "the mini-spec calls for...", "open question 6 in the spec", "(not the root bounds -- third-review finding)" — literal review-process narration; cut to the rounding rule + bound-clamping contract.

## OP.Component.fProxy.cs
- `OP.Component.fProxy.cs:53` — HISTORY — "(compAdd is (target, from) — a prior reversed call mutated `from` instead.)" — delete the bug-history parenthetical.
- `OP.Component.fProxy.cs:154` — TOO-LONG/HISTORY — CS8338 receiver explanation + "Existing callers that wrote the old static-style clampInPlace(in v, ...) just drop the now-illegal `in`" — trim remarks to the contract ("lo must be <= hi"); drop migration history.
- `OP.Component.fProxy.cs:170` — HISTORY — "forwarding to UnsafeMathOP (mathUnsafe's former home)" — drop the rename aside.
- `OP.Component.fProxy.cs:306` — HISTORY — "Not in the original exposure list but componentwise like every other kernel here, so exposed for consistency" — drop the internal-planning reference.
- `OP.Component.fProxy.cs:314` — HISTORY — "Not in the original exposure list, but iProxyComp's analogous reluInPlace was explicitly requested" — delete.
- `OP.Component.fProxy.cs:422` — HISTORY/TOO-LONG — "Not in the original exposure list (which only spelled out clamp/lerp/smoothstep/step...) ... excluded by none of the stated exclusion rules" — delete design-review narration.

## OP.Component.iProxy.cs
- `OP.Component.iProxy.cs:57` — HISTORY — "(was passing the operands to compAdd reversed → mutated `from` instead.)" — delete.
- `OP.Component.iProxy.cs:116` — TOO-LONG — 6-line rationale for direct subInPlace kernel — compress to "direct kernel; unsigned types can't negate s".
- `OP.Component.iProxy.cs:152` — TOO-LONG/HISTORY — same CS8338/migration remarks block as the fProxy file — trim to the contract.
- `OP.Component.iProxy.cs:259` — HISTORY — "forwarding to UnsafeMathOP (mathUnsafe's former home)" — drop aside.
- `OP.Component.iProxy.cs:299` — TOO-LONG — 6-line section-header essay on bit-manipulation sign-agnosticism — compress to one line.

## OP.Dot.fProxy.cs
- `OP.Dot.fProxy.cs:95` — HISTORY — "Krylov R2's ApplyDot (docs/draft-spec-krylov-optimization.md...) MEASURED WORSE on the BSR analogue... Reverted here too on the same architectural basis" — delete; internal benchmark/revert narration with ticket codes and spec files.
- `OP.Dot.fProxy.cs:278` — NOISE — "// Inline dot product calculation" — delete.
- `OP.Dot.fProxy.cs:292` — NOISE — "// Apply directly to matrix" — delete.

## OpHelpers.Shared.cs
- `OpHelpers.Shared.cs:6` — JARGON — "helpers hoisted out of the per-type FFT / Resample templates" — "shared by the FFT / Resample templates".

## Optimize.fProxy.cs
- `Optimize.fProxy.cs:224` — TOO-LONG — `ladIRLS` 4-paragraph summary (algorithm restatement, delta tuning advice, tutorial) — keep complexity + rank-deficiency caveat + job-safety; drop the weighting-formula walkthrough.

## Query.Shared.cs
- `Query.Shared.cs:6` — JARGON — "Query helper hoisted out of the per-type QueryOP templates" — "shared by the per-type QueryOP templates".

## QP.Info.cs
- `QP.Info.cs:13` — HISTORY — "STAGE 1 (docs/draft-spec-qp.md) -- the fixed-working-set equality QP kernel" — drop the stage/spec tag.
- `QP.Info.cs:19` — HISTORY — "the enum is defined complete now so that loop does not need a breaking status-enum change later" — replace with "reserved for future active-set use".
- `QP.Info.cs:67` — HISTORY — "STAGE 2 (docs/draft-spec-qp.md): per-row state of the active-set working set W" — drop tag.
- `QP.Info.cs:94` — TOO-LONG — multi-paragraph `QPInfo` doc (usage sample, precision rationale, "stage 2-3 will extend this struct's diagnostics") — cut to 2-3 sentences: contents, objective is always double, Optimal/Unbounded contract.

## QP.fProxy.cs
(Note: `STAGE n` / `docs/draft-spec-qp.md` tags recur ~35 times across this file; items below are the worst instances. One sweep removing stage numbers and spec-file pointers fixes most of the file.)
- `QP.fProxy.cs:12` — TOO-LONG — 70-line file-header tutorial derivation of the null-space method (Nocedal & Wright eq. 16.16-16.19, step-by-step walkthrough) — compress to "null-space active-set QP; Q must be PSD; no dense null-space basis is formed"; move the derivation to docs/ if kept.
- `QP.fProxy.cs:85` — HISTORY — "STAGE 3 (docs/draft-spec-qp.md): the PUBLIC FACADE -- QP.solve, mirroring LP.solve's doc voice..." — keep only the validation contract.
- `QP.fProxy.cs:205` — TOO-LONG/HISTORY — "Two alternatives were considered and rejected: (1)..." — design-review writeup; state only what `PhaseOneFeasibleStart` does.
- `QP.fProxy.cs:358` — TOO-LONG — half of `eqpNullSpaceStep` doc is "STAGE 2-3 (future): the active-set loop will call this once PER ITERATION... expect that seam to require splitting" — roadmap speculation; document current behavior only.
- `QP.fProxy.cs:545` — HISTORY — "v1 scope (draft-spec-qp.md \"Judgment\"): this ALWAYS re-factors A_Wᵀ from scratch. HiGHS instead maintains..." — reduce to "always re-factors from scratch (no incremental update)".
- `QP.fProxy.cs:698` — TOO-LONG — ~85-line unbounded-detection derivation with a 4-condition proof and textbook section citations — keep a short summary of the 4 conditions; cut the proof.
- `QP.fProxy.cs:762` — HISTORY — "(\"Active-Set Methods for Indefinite QP\") -- fetched and read 2026-07-09: with Z the null-space..." — delete the date-stamped research note; keep at most the citation.
- `QP.fProxy.cs:1112` — TOO-LONG — cross-file rationale essay quoting LP.DualSimplexCore's header — shrink to "one exact Newton step against the true bounds removes perturbation drift".
- `QP.fProxy.cs:1223` — TOO-LONG — `BuildPerturbedBounds` multi-sentence justification (MurmurHash3 mix, magnitude proofs) for a 10-line function — keep the one-line contract ("widens L/U deterministically to break ratio-test ties").

## QueryOP.iProxy.cs
- `QueryOP.iProxy.cs:10` — HISTORY — "This is the P2 subset from spec-query.md" — drop the spec-file/phase reference; just state which metrics are integer-exact.
- `QueryOP.iProxy.cs:18` — HISTORY — "P3 overflow note: ALL integer metrics require..." (label recurs at line 391 and is referenced from the XML doc at line 398) — keep the overflow contract, drop the internal "P3" phase label.

## RandomMatrixOP.fProxy.cs
- `RandomMatrixOP.fProxy.cs:141` — TOO-LONG — `orthogonalInPlace` doc restates the algorithm as a numbered tutorial ("1. Fill G with N(0,1)... 2. QR-decompose... 3. Haar sign fix... 4. Copy corrected Q") plus a "WHY" bias essay duplicated by inline comments at line 179 — condense to what it produces, the Mezzadri 2007 citation, and the throw contract.

## QR.fProxy.cs
- `QR.fProxy.cs:283` — HISTORY — "column-tiling was tried and measured SLOWER (added MemClear/call overhead...), so it is deliberately not done here" — delete the rejected-experiment narration.
- `QR.fProxy.cs:112` — JARGON (also line 605) — "Hoist both out of a hot loop to skip the per-call Allocator.Temp allocs" — "move both out of the hot loop".

## QRCP.Workspace.fProxy.cs
- `QRCP.Workspace.fProxy.cs:21` — HISTORY — "the guarded norm-downdating state from docs/dev/spec-qrcp-downdate.md" — drop the spec-file citation from this public struct doc.
- `QRCP.Workspace.fProxy.cs:26` — TOO-LONG/HISTORY — "Deliberately holds ONLY vn1/vn2... (revisiting OQ-7 of docs/dev/spec-solver-api-rework.md: QRCP earns a cache purely for the downdating state). Promoting the blocked buffers... is a candidate follow-up." — public API doc containing an internal ticket reference and a roadmap note; cut to "Holds the two n-length downdating vectors. Allocate once via Arena.fProxyQRCPCache(n) and reuse across same-shape calls."

## QRCP.fProxy.cs
- `QRCP.fProxy.cs:21` — TOO-LONG — class `<remarks>` is a full paragraph on downdating mechanics, the blocked panel core, and Temp-buffer allocation strategy — keep 2 lines ("column norms are downdated LAPACK dgeqp3-style; wide matrices use a blocked level-3 panel core") and drop the internals.
- `QRCP.fProxy.cs:114` — HISTORY — "transcribed unsquared, per docs/dev/spec-qrcp-downdate.md" — drop the spec citation; "LAPACK dgeqp3/dlaqps-style" suffices.
- `QRCP.fProxy.cs:124` — TOO-LONG — "tol3z is Consts.fProxySqrtEps directly: Consts.cs already defines it... every other caller in this codebase (Eigen/LOBPCG/Krylov/SVD.LowRank) references it the same way rather than recomputing" — codebase-consistency essay; delete (one line "tol3z = sqrt(eps)" is enough).
- `QRCP.fProxy.cs:129` — HISTORY — "the old exact-recompute-every-step buffer is fully retired. This is a deliberate widening from LAPACK's own per-column selective recompute... the ORIGINAL always-exact QRCP avoided..." — narrates the pre-optimization implementation; describe current behavior only ("guard trip re-sums all trailing columns in one batched row-major sweep").
- `QRCP.fProxy.cs:188` — HISTORY — "Same relative tie tolerance the exact-recompute kernel used... the old test compared squared norms via maxNorm2 > diagNorm2*(1+pivotRelTol)... see docs/dev/spec-qrcp-downdate.md OQ-D1" — delete old-code comparison and spec/ticket reference; state the tolerance and why ties are left in place.
- `QRCP.fProxy.cs:193` — JARGON — "m is fixed for the whole call, so this is hoisted out of the per-step loop" — "computed once before the loop".
- `QRCP.fProxy.cs:377` — TOO-LONG/HISTORY — ~35-line blocked-core section header with a numbered 8-step algorithm walkthrough and "Full derivation + range table: docs/dev/spec-qrcp-blocked.md" — condense to the invariant (A_true = A_stale − V·Fᵀ) plus one line per unusual choice; drop the spec pointer.
- `QRCP.fProxy.cs:429` — HISTORY — "32 is the measured sweep optimum — the same width QR settled on" — benchmark narration; "panel width 32 (matches QR)".
- `QRCP.fProxy.cs:807` — TOO-LONG — solveInPlace public doc's third paragraph: "never forms or reconstructs Q — the whole point, since reconstruction is ~⅓ of the runtime" — keep the DESTROYS-A-and-b contract (must stay); drop the runtime-fraction sales pitch. The "~⅓ of runtime" claim recurs at lines 411, 925, 1318 and 1478 — remove everywhere.
- `QRCP.fProxy.cs:1067` — TOO-LONG — COD section header is a ~30-line mathematical derivation (full R-block algebra, LQ compress proof, LQRP transpose-dual discussion) — trim to "min-norm solve via complete orthogonal decomposition (xGELSY): QRCP + LQ of the top r×n block"; move the derivation to docs/.

CLEAN FILES: MIP.Info.cs, NormsOP.fProxy.cs, NormsOP.iProxy.cs, OP.Dot.iProxy.cs, QR.Workspace.fProxy.cs, QueryCore.Metric.fProxy.cs, QueryCore.Metric.iProxy.cs, QueryCore.Predicate.fProxy.cs, QueryCore.Predicate.iProxy.cs, QueryEnums.cs, QueryOP.Predicate.fProxy.cs, QueryOP.Predicate.iProxy.cs, QueryOP.fProxy.cs, RandomOP.bool.cs, RandomOP.cs, RandomOP.fProxy.cs, RandomOP.iProxy.cs, ResampleEnums.cs, ResampleOP.fProxy.cs
