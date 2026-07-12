# DEVLOG — Statistics
Code comments state contracts only; history lives here (see CLAUDE.md).

## StatsCore.iProxy.cs
- 2026-07-12 | The `long`-accumulator sum contract (int/short always safe, long can wrap) is
  pinned by StatsTests.iProxy.cs's SumAccumulatorOwnOverflow: the same 2-element/MaxValue-filled
  input is correct-and-widened for int/short but silently wraps for long. (was StatsCore.iProxy.cs:27-29)
