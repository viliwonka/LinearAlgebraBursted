# Release scan 2026-07-12 — area: new-demos (post-scan code)

{"total":4,"confirmed":2,"uncertain":0,"unverified":0,"refuted":2,"high":1,"medium":0,"low":1}

## Scope

- Assets/Demos/09_TrussModal/TrussModalDemo.cs
- Assets/Demos/10_Loadout/LoadoutMIPDemo.cs
- Assets/Demos/11_HoverTank/HoverTankDemo.cs
- Assets/Demos/Tests/TrussModalSmokeTests.cs
- Assets/Demos/Tests/LoadoutSmokeTests.cs
- Assets/Demos/Tests/HoverTankSmokeTests.cs

## Findings

### 1. [high/pointer/CONFIRMED] Assets/Demos/11_HoverTank/HoverTankDemo.cs:210 — NullReferenceException in the auto-orbit block when an assigned target is destroyed at runtime

**Evidence**

```
BuildScene only creates autoTargetGO when target starts null: line 191
`if (target == null) { autoTargetGO = GameObject.CreatePrimitive(...) }`.
But FixedUpdate line 207-211 does
`if (target == null && autoOrbitTarget) { ... autoTargetGO.transform.position = ... }`
with no null-check on autoTargetGO. If the user assigns `target` in the
inspector (the primary purpose of the public field, default
autoOrbitTarget=true) and that target is later destroyed, `target == null`
becomes true while autoTargetGO was never created (real C# null) -> NRE
every FixedUpdate. This is exactly the scenario the class explicitly claims
to handle (line 224 comment: 'a user-assigned target destroyed at runtime'),
and the aim path at line 225 and gizmo path at line 309 DO guard it -- only
the orbit block does not.
```

**Verifier** — Verified in Assets/Demos/11_HoverTank/HoverTankDemo.cs. BuildScene (line 191-199) creates autoTargetGO only inside `if (target == null)`, so when the user assigns `target` in the inspector, `autoTargetGO` stays as real C# null. The FixedUpdate orbit block (line 207-212) enters on `target == null && autoOrbitTarget`, where `autoOrbitTarget` defaults to true (line 63). If the assigned `target` is destroyed at runtime, Unity's overloaded `==` makes `target == null` return true, the block runs, and dereferences `autoTargetGO.transform` at line 210 — real NullReferenceException every FixedUpdate. The sibling aim path at line 225 and gizmo path at line 309 both use the correct guarded pattern `autoTargetGO != null ? autoTargetGO.transform : null`, and the line 223-224 comment explicitly names "a user-assigned target destroyed at runtime" as an intended-handled case. No DEVLOG exists for the Demos folder documenting the orbit-block omission as intentional. The severity rating (high) is appropriate for a runtime NRE in an on-by-default code path. Suggested fix (`&& autoTargetGO != null`) is minimal and consistent with the existing guards.

**Suggested fix** — Guard the orbit branch with `&& autoTargetGO != null`, or lazily create autoTargetGO when needed.

### 2. [low/logical/CONFIRMED] Assets/Demos/10_Loadout/LoadoutMIPDemo.cs:71 — (int)-cast slider makes the last selectable index nearly unreachable (mandatory item, max weapons)

**Evidence**

```
Line 71 `mandatoryItemIndex = Mathf.Clamp((int)LabeledSlider(...,
mandatoryItemIndex, 0, ItemCount - 1), ...)` and line 69
`maxWeapons = (int)LabeledSlider(..., 0, 6)` cast a float slider over
[0, max] to int. Every integer k<max occupies width 1 but the top value
(item index 15 'Cloaking Device', or maxWeapons 6) is only hit at the exact
right edge, so it is practically unselectable. The sibling TrussModalDemo
deliberately avoids this with `ModeCount - 0.51f` (line 208), so this is an
inconsistent/copy-paste-hazardous pattern.
```

**Verifier** — Verified in Assets/Demos/10_Loadout/LoadoutMIPDemo.cs:69 and :71. `maxWeapons = (int)LabeledSlider(..., 0, 6)` and `mandatoryItemIndex = Mathf.Clamp((int)LabeledSlider(..., mandatoryItemIndex, 0, ItemCount - 1), 0, ItemCount - 1)` both cast a continuous slider value in [0, hi] to int via truncation. Every integer k<hi occupies a full unit of slider width; the top value (maxWeapons=6, or item index 15 "Cloaking Device") is only produced when the slider snaps to exactly hi, i.e. the right-edge single point, so it is practically unselectable via dragging. Confirmed inconsistency: Assets/Demos/09_TrussModal/TrussModalDemo.cs:208 uses `ModeCount - 0.51f` and Assets/Demos/04_TrussStability/TrussStabilityDemo.cs:197 uses `3.49f` as the upper bound precisely to give the top integer a normal-width bucket, so an internal idiom exists that LoadoutMIPDemo does not follow. No DEVLOG.md exists under Assets/Demos/10_Loadout/ documenting a deliberate deviation. Low-severity UX/logical defect; no numerical impact on the MIP solve itself. Suggested fix (hi + 0.49f before the int-cast) matches the local pattern.

**Suggested fix** — Use `hi + 0.49f` as the slider upper bound before the int-cast (as TrussModalDemo does), or round instead of truncate.

## Refuted

| # | File:line | Summary | Why refuted |
|---|-----------|---------|-------------|
| R1 | Assets/Demos/11_HoverTank/HoverTankDemo.cs:121 | OnDestroy disposes native buffers but never destroys the self-created scene GameObjects/materials | Factually accurate observation but by-design for this demo suite: demos are single-component enter-play-mode scenes (Assets/Demos/README.md) where play-mode teardown reclaims scene objects; 9 of 11 demos have no OnDestroy at all and none destroy scene objects — OnDestroy exists only to satisfy Allocator.Persistent leak warnings. The demo DEVLOG (docs/dev/demo-findings.md) catalogs teaching-material issues exhaustively and never flags GameObject/material cleanup. The failing scenario (additive scene unload, runtime Destroy) is hypothetical — no code path in the repo does it. Also outside the review's declared scope (numerical correctness / arena+in-place memory model / codegen / Burst / tests); "pointer" category mislabeled. |
| R2 | Assets/Demos/Tests/TrussModalSmokeTests.cs:96 | Temp allocations phi/Aphi/Bphi never disposed despite the test's '(d) no native leaks' claim | By-design, matches the folder convention: DemoSmokeTests.cs:148-150 uses the identical pattern (Allocator.Temp floatN handles left undisposed; only TempJob outStats + arena disposed at 188-189). Unity Collections' leak detector does not track Allocator.Temp (per-thread frame-rewind allocator) — "can trip the leak detector" is speculation, not a traced failure. The "(d) no native leaks" comment labels teardown of the resources that DO need explicit disposal (TempJob NativeArray, Arena), consistent with docs/dev/rfc-memory-model.md §3.5 and Assets/Demos/README.md:21. floatN.cs:73-86 shows the standalone Temp path uses no arena record (`_rec = null`), so no arena-slot leak either. No concrete failing scenario. |

## Scanner notes

Verified against solver signatures: MIP.solve arg order (maxNodes, maxIter, absGap, relGap) matches the demo call; objective is +Inf on no-incumbent / NaN on infeasible, so LoadoutMIPDemo's hasIncumbent guard (line 165) is correct. floatLOBPCGCache.X is k x n (LOBPCG.Cache.float.cs:84-87), matching modes[mode,dof] indexing in both TrussModalDemo and its smoke test. The HoverTank hover mixer and roll/pitch sign conventions are internally self-consistent (sensing and actuation both use ride-height as the currency; positive tauRoll raises right corners -> raises right ride height -> raises estimated roll -> negative feedback via u=-Kx), and the r x F cross product confirms the CornerDX/CornerDZ lever arms in the mixer match the physical torque. Gravity feedforward sign (Gravity = -Physics.gravity.y = +9.81, fTotal = Mass*(Gravity+u)) gives a correct hover equilibrium. Barrel/turret Euler sign conventions (line 278-279) match the atan2 desired-angle definitions. The three smoke tests are non-vacuous (brute-force 2^16 cross-check for Loadout, independent spMV residual recomputation for TrussModal, linear closed-loop decay for HoverTank). No numerical/formula defects found in the control or eigen math.
