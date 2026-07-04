using System.Runtime.CompilerServices;

// Grants the hand-authored SourceTests assembly (BurstLinearAlgebra.Tests) visibility into this
// assembly's `internal` types -- needed so concrete (NOT codegen'd) test files like
// ChunkedRecordTableTests.cs can exercise internal-only building blocks (e.g.
// LinearAlgebra.ChunkedRecordTable<TRecord>, docs/rfc-memory-model.md §4/§6.1/§7 step 2) directly,
// the same way ArenaLayoutTests.cs already exercises Arena's public surface. Does not affect any
// other consumer of this assembly (InternalsVisibleTo grants visibility only to the named assembly).
[assembly: InternalsVisibleTo("BurstLinearAlgebra.Tests")]
