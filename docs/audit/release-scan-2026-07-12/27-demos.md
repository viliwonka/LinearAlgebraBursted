# Release scan 2026-07-12 — area: demos (non-template)

Scanned 9 files (demos). Counts: total 5, confirmed 4, uncertain 0, unverified 0, refuted 1 — high 0, medium 0, low 4.

## Scope

- Assets/Demos/01_LeastSquares/LeastSquaresFitDemo.cs
- Assets/Demos/02_LeastAbsoluteDeviation/LadFitDemo.cs
- Assets/Demos/03_LinearProgram/EconomyLPDemo.cs
- Assets/Demos/04_TrussStability/TrussStabilityDemo.cs
- Assets/Demos/05_PendulumLQR/CartPoleLQRDemo.cs
- Assets/Demos/05_PendulumLQR/DoubleCartPoleLQRDemo.cs
- Assets/Demos/06_DroneLQR/DroneLQRDemo.cs
- Assets/Demos/07_SpringSystem/SpringLatticeDemo.cs
- Assets/Demos/08_Circuit/CircuitDemo.cs

## Findings

### 1. [low/performance/CONFIRMED] Assets/Demos/01_LeastSquares/LeastSquaresFitDemo.cs:71 — Every demo allocates a managed Stopwatch object per frame in Update (GC pressure; teaching anti-pattern)

**Evidence**

```
var sw = Stopwatch.StartNew();
```

`var sw = Stopwatch.StartNew();` inside Update. Stopwatch.StartNew() does `new Stopwatch()` each call, so a heap object is allocated every frame. Repeated verbatim in all nine demos: LadFitDemo.cs:73, EconomyLPDemo.cs:106, TrussStabilityDemo.cs:142, CartPoleLQRDemo.cs:68, DoubleCartPoleLQRDemo.cs:65, DroneLQRDemo.cs:77, SpringLatticeDemo.cs:144, CircuitDemo.cs:117.

**Verifier**

Verified at Assets/Demos/01_LeastSquares/LeastSquaresFitDemo.cs:71. The line `var sw = Stopwatch.StartNew();` sits inside the unconditional per-frame `Update()` method (opened at line 50) with no debug/editor gating. `System.Diagnostics.Stopwatch` is a reference type and `StartNew()` internally does `new Stopwatch()`, so this is a genuine per-frame managed heap allocation. Grep across Assets/Demos confirms the identical pattern at all eight other file:line locations named in the report (CircuitDemo.cs:117, DoubleCartPoleLQRDemo.cs:65, LadFitDemo.cs:73, DroneLQRDemo.cs:77, SpringLatticeDemo.cs:144, CartPoleLQRDemo.cs:68, TrussStabilityDemo.cs:142, EconomyLPDemo.cs:106). Severity "low" is honest — one small gen-0 object per frame won't stall the collector — but the "teaching anti-pattern in copy-paste demos" framing lands: these demos are explicitly reference material for users writing Burst compute loops that must be alloc-free. The suggested fix (one field-cached Stopwatch + `sw.Restart()`) is functionally correct: `Restart()` resets elapsed and starts the existing instance without allocating.

**Suggested fix**

Cache one Stopwatch in a field and call `sw.Restart()` each frame instead of `Stopwatch.StartNew()`; demos are copy-paste teaching material so the zero-alloc pattern is worth showing.

### 2. [low/performance/CONFIRMED] Assets/Demos/05_PendulumLQR/DoubleCartPoleLQRDemo.cs:186 — RK4 loop allocates 5 NativeArrays per substep plus one per Blend and two per Deriv, all Allocator.Temp, where stack float vectors would do

**Evidence**

```
var z = new NativeArray<float>(6, Allocator.Temp);
```

Per substep: `var z = new NativeArray<float>(6, Allocator.Temp);` and k1..k4 (lines 186-190), `Blend` news another `NativeArray<float>(6, Allocator.Temp)` (line 206) three times, and each `Deriv` news `floatMxN(3,3)` + `floatN(3)` (lines 229,232) four times — ~16 Temp allocations/substep × 8 substeps. The single-pole demo (CartPoleLQRDemo) does the equivalent with stack `float4`, so this is a needless inconsistency in the teaching set.

**Verifier**

Verified line-by-line. Lines 186-190 allocate 5 NativeArray<float>(6, Allocator.Temp) per substep (z, k1-k4). Blend (line 206) allocates another NativeArray<float>(6, Temp) — invoked 3× per substep at lines 194-196. Deriv (lines 229, 232) allocates floatMxN(3,3, Temp) + floatN(3, Temp) — invoked 4× per substep at lines 193-196. Total 16 Temp allocations/substep × 8 substeps = 128/frame, exactly as claimed. The sibling CartPoleLQRDemo.cs at lines 185-192 does the RK4 with stack float4 and its Deriv returns float4 (lines 197-205) with zero allocations, so the "single-pole demo does the equivalent with stack float4" comparison is accurate. Note: the single-pole variant can skip a CHO solve entirely because its 4-state dynamics have an analytical closed form; the double-pole must still solve the 3×3 mass matrix per Deriv call — the suggested fix correctly narrows library buffers to that CHO. Allocator.Temp is a bump allocator auto-freed at job end so this is not a leak or correctness bug — genuine performance/teaching-set inconsistency, matching the reported "low performance" severity. Nothing to refute.

**Suggested fix**

Represent the 6-state as a stack struct (e.g. two float3 / float3x2 as the drone demo does) so the hot RK4 path allocates nothing; only the 3x3 CHO solve needs library buffers.

### 3. [low/performance/CONFIRMED] Assets/Demos/04_TrussStability/TrussStabilityDemo.cs:183 — A new GUIStyle is allocated on every OnGUI event (OnGUI fires multiple times per frame)

**Evidence**

```
var style = new GUIStyle(GUI.skin.label);
```

`var style = new GUIStyle(GUI.skin.label);` inside OnGUI, allocated every layout/repaint pass just to set a text color.

**Verifier**

Line 183 of Assets/Demos/04_TrussStability/TrussStabilityDemo.cs contains `var style = new GUIStyle(GUI.skin.label);` inside the OnGUI method (lines 174-198), within the `if (built && lambda.IsCreated)` block that runs in normal steady state after the first Update. Unity dispatches OnGUI at least twice per frame (Layout + Repaint) plus once per input event, and each dispatch allocates a fresh GUIStyle plus its internal GUIStyleState/RectOffset copies from GUI.skin.label just to set one textColor before a single GUILayout.Label call. No caching field, no repaint guard, no `using`/pooling — a genuine per-event managed allocation. Severity "low" is correct (demo code, not a compute path, no correctness impact); the suggested fixes (cache the style in a field lazily, or wrap the label with GUI.contentColor) both eliminate the allocation. Claim is accurate.

**Suggested fix**

Build the GUIStyle once (lazily cached in a field) and only mutate its color, or use GUI.contentColor around the label.

### 4. [low/naming/CONFIRMED] Assets/Demos/05_PendulumLQR/CartPoleLQRDemo.cs:95 — Control-force gizmo hardcodes a /30 scale while maxForce is a 5-100 slider, so the arrow length misrepresents force when maxForce != 30

**Evidence**

```
Gizmos.DrawLine(cart, cart + new Vector3(u / 30f, 0f, 0f));
```

`Gizmos.DrawLine(cart, cart + new Vector3(u / 30f, 0f, 0f));` uses a fixed 30 divisor, but `[Range(5f, 100f)] public float maxForce = 30f;` (line 28) lets the clamp limit change; at maxForce=100 a saturated force draws a arrow 3.3x longer than the intended unit scale.

**Verifier**

Code inspection confirms the claim exactly:

- Line 28: `[Range(5f, 100f)] public float maxForce = 30f;` — slider range 5..100, default 30.
- Line 95: `Gizmos.DrawLine(cart, cart + new Vector3(u / 30f, 0f, 0f));` — hardcoded /30f divisor.
- Line 183 (in `CartPoleStepJob.Execute`): `u = math.clamp(u, -MaxForce, MaxForce);` — `u` (written to `Out[0]` = `outStats[0]` read at line 94) is bounded by ±maxForce.

Therefore at maxForce=30 a saturated force draws a 1-unit arrow; at maxForce=100 it draws ~3.33 units; at maxForce=5 it draws ~0.17 units. The visualization scale does not track the slider — the fact that 30f matches only the default value (with no named constant, no comment justifying a fixed N-per-unit scale) points to the divisor being tied to the default rather than an explicit "1 unit = 30 N" design decision.

No other guard or scaling exists in `OnDrawGizmos`. Severity "low" is appropriate — this is cosmetic demo code, not a numerical/memory correctness issue, but the observation is factually accurate. Suggested fix (`u / MaxForce`) is reasonable, though `u / maxForce` (the MonoBehaviour field, not the job-copy) would produce a full-range arrow at saturation regardless of slider position.

File path (absolute): `C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\Demos\05_PendulumLQR\CartPoleLQRDemo.cs` — line 95 (bug), line 28 (slider bound), line 183 (clamp source).

**Suggested fix**

Scale the arrow by `u / maxForce` (or another explicit visual constant tied to the current clamp) so it stays normalized as the slider moves.

## Refuted

| file:line | claim | why refuted |
|---|---|---|
| Assets/Demos/05_PendulumLQR/DoubleCartPoleLQRDemo.cs:175 | Riccati/LQR result K is used to integrate persistent state regardless of the converged flag; a single failed solve permanently poisons state (NaN K latches State into NaN until Reset). | Every failure path in the warm `ref floatLQRState` overload guarantees K stays finite (verified in Assets/LinearAlgebra/Source/OP/Control.float.cs): RiccatiStep leaves Snext/K untouched on hard CHOP failure (writes gated on `rinfo.Solved`, line 140); RiccatiIterate's finalize step that writes K (line 339) is gated on `status != LQRStatus.Diverged` (line 337) — Diverged keeps last-converged values by design; SDACore keeps S bounded (H0=Q at worst, line 272). The LQRStatus.Diverged XML contract (Control.Info.cs:29-31) guarantees "outputs are the last KNOWN-GOOD iterate (not the exploded one), never NaN". Demo also clamps u to ±MaxForce (DoubleCartPoleLQRDemo.cs:184) and the 3x3 mass matrix stays SPD for slider-legal values, so the inner CHO solves (lines 155, 237) can't NaN either. "info reported but never checked" is factually true but intentional — the warm-start overload exists to keep using the last-good K on transient failures. No concrete failing scenario exists. |

## Scanner notes

Verified positives (not defects): control-model sign consistency between each linearization and its nonlinear Deriv (cart-pole, double-cart-pole, drone all check out); truss BSR symmetric lower-block assembly and pin-penalty DOF indexing (nodes 0->dof 0,1 and 3->dof 6,7) are correct; warm-cache demos (EconomyLP/Truss/CartPole/DoubleCartPole/Drone) correctly use IJobExtensions.RunByRef + copy the LPBasis/LQRState/LOBPCG cache back, while stateless demos correctly use Run(); floatN(Voltages) zero-copy view in CircuitDemo writes solution back in place with no b/x aliasing; no cross-frame native leaks found (persistent arrays disposed in OnDisable, arenas disposed/rebuilt on parameter change, in-job allocations are Allocator.Temp). No arena-allocation-inside-a-job trap in any demo. No high-severity numerical/pointer defects found.
