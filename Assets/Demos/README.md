# Demos

Realtime demos exercising the library from MonoBehaviour + Burst jobs (float API only).
Each demo is a single component: create an empty GameObject, add the component, enter
play mode. Visuals are gizmos (enable Gizmos in the Game view) + an on-screen panel.

| Folder | Component | Library surface |
|---|---|---|
| 01_LeastSquares | `LeastSquaresFitDemo` | `QR.solveInPlace` (plane/quadric fit, per frame) |
| 02_LeastAbsoluteDeviation | `LadFitDemo` | `LP.lad` / `ladBR` / `ladFN` (τ quantile) vs `QR` |
| 03_LinearProgram | `EconomyLPDemo` | warm `LP.solve` + `LPBasis` + `floatLPCache` |
| 04_TrussStability | `TrussStabilityDemo` | symmetric BSR assembly, `Eigen.lobpcg` + `floatBlockJacobi` |
| 05_PendulumLQR | `CartPoleLQRDemo` | `LQR.lqr` warm (`floatLQRState`), cart-pole |
| 06_DroneLQR | `DroneLQRDemo` | `LQR.lqr` warm, 6-state planar quadrotor |
| 07_SpringSystem | `SpringLatticeDemo` | `floatBSRBuilder`, `Krylov.cg` + `floatIC0` |
| 08_Circuit | `CircuitDemo` | MNA (indefinite), `Krylov.pbiCGStab` + `floatILU0` |

Interop pattern used throughout: solver warm-state structs (`LPBasis`, `floatLPCache`,
`floatLQRState`, `floatLOBPCGCache`) carry scalar fields the solvers mutate, so jobs
run via `IJobExtensions.RunByRef(ref job)` and the structs are copied back after the
run. Matrices/vectors built inside jobs use `Allocator.Temp` (zero-initialized).

Dev notes and API findings from writing these: `docs/dev/demo-findings.md`.
