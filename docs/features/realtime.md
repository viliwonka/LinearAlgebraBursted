# Realtime — RollingWindow

`LinearAlgebra.Realtime.floatRollingWindow` — a fixed-capacity ring buffer of feature vectors
(`Capacity` rows × `Features` columns), indexed oldest→newest. Allocated via
`arena.floatRollingWindow(int capacity, int features)`, disposed with the arena.

- `Push(in floatN sample)` — O(Features), overwrites the oldest row once full.
- `Mean(ref floatN dest)` / `Mean() : floatN` — moving average.
- `Covariance(ref floatMxN dest)` / `Covariance() : floatMxN` — requires `Count ≥ 2`; reuses
  [`Stats.covarianceInto`](stats.md).
- `AsMatrix(ref floatMxN dest)` / `AsMatrix() : floatMxN` — materializes the buffer in time order
  (oldest-first); the allocating form pulls from the arena's temp pool.
- `Count`, `Capacity`, `Features`, `IsFull`, `IsEmpty`, indexer `this[int i, int f]`, `GetSample`,
  `Clear()`.

Kalman filtering (linear KF, EKF, UKF) is implemented separately on `Kalman` — see
[control.md](control.md). The rest of the "realtime" design surface — frame-amortized solvers,
resumable iterative state (CG/PCG stepping across frames), online covariance/PCA — is still
unsettled design, not implemented.
