# DEVLOG — Realtime
Code comments state contracts only; history lives here (see CLAUDE.md).

## fProxyRollingWindow._head/_count: native-backed ring state
- 2026-07-23 | `_head`/`_count` were plain int fields mutated by `Push`/`Clear` -- lost on an `IJob`
  by-value copy, same bug class as `fProxyLQRState.populated`/`fProxyMPCState.populated`/`LPBasis.
  populated` (see OP/DEVLOG.md). A window held as a job field across separate `Run()`/`Schedule()` calls
  would silently forget every push once the job returned. Fixed with a 2-slot `NativeArray<int>`
  (`_ring[0]`=head, `_ring[1]`=count) allocated alongside the ring buffer in both constructors and freed
  in `Dispose`; `_head`/`_count` became properties over those slots with the SAME private names, so every
  existing reader/writer (Push, Clear, OldestRow, RingRow, the indexer, GetSample, AsMatrix, Mean,
  Covariance) needed no further changes. Public API surface unchanged. See
  [[job-struct-copy-warmstate-audit]].
