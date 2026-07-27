# Realtime - RollingWindow

`BULA.Realtime.floatRollingWindow` - a fixed-capacity ring buffer of feature vectors
(`Capacity` rows × `Features` columns), indexed oldest→newest. Allocated via
`new floatRollingWindow(int capacity, int features, Allocator allocator)`, disposed via `.Dispose()`.

- `Push(in floatN sample)` - O(Features), overwrites the oldest row once full.
- `Mean(ref floatN dest)` / `Mean() : floatN` - moving average.
- `Covariance(ref floatMxN dest)` / `Covariance() : floatMxN` - requires `Count ≥ 2`; reuses
  [`Stats.covarianceInto`](stats.md).
- `AsMatrix(ref floatMxN dest)` / `AsMatrix() : floatMxN` - materializes the buffer in time order
  (oldest-first); the allocating form uses `Allocator.Temp`.
- `Count`, `Capacity`, `Features`, `IsFull`, `IsEmpty`, indexer `this[int i, int f]`, `GetSample`,
  `Clear()`.

Kalman filtering (linear KF, EKF, UKF) is implemented separately on `Kalman` - see
[control.md](control.md). Frame-amortized solvers, resumable iterative state (CG/PCG stepping across frames), and online covariance/PCA are not yet implemented.
