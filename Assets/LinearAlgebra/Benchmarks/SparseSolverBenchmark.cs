using System.Collections.Generic;
using System.Globalization;
using System.Text;

using Unity.Mathematics;

namespace LinearAlgebra.Benchmarks
{
    // Small position record used by the block-pattern choosers (avoids depending on ValueTuple support
    // one way or the other). Public so the code-generated per-dtype builders (in a separate template
    // assembly) can name it as the choosers' element type.
    public readonly struct BlockPos
    {
        public readonly int Bi, Bj;
        public BlockPos(int bi, int bj) { Bi = bi; Bj = bj; }
    }

    // Shared, dtype-agnostic config + table formatters + block-pattern choosers for SparseSolverBenchmark.
    // Public so the code-generated per-dtype jobs/builders/sections (in a separate template assembly) can
    // reach the constants (BR, densities, iteration budgets), the seed helper, the row formatters, and the
    // index-only block choosers. The choosers are dtype-INDEPENDENT (pure block-index/count logic), so they
    // live here once rather than being generated twice.
    public static class SparseSolverFmt
    {
        // Block-aligned sizes (N = nb * BR). All three sizes fit comfortably within a few minutes for this
        // section (see the report's own timings); if a future change makes this section too slow, drop 768
        // or lower Bench.Runs for just this section and note it here.
        public static readonly int[] BlockSizesN = { 192, 384, 768 };
        public const int BR = 3; // block size b=3 (the FEM/cloth/PD workhorse)
        public static readonly float[] Densities = { 0.07f, 0.33f }; // ~7% / ~33% of BLOCKS nonzero

        public const int K_CG = 40;        // CG / MINRES iteration budget (fixed, tol=0)
        public const int K_BICGSTAB = 40;  // BiCGSTAB iteration budget
        public const int K_LS = 24;        // CGLS / LSQR iteration budget
        public const int REPS_MATVEC = 64; // operator microbench: back-to-back matvecs per timed sample

        // Section 1x dedicated b=4/N=1024 case (1024 isn't divisible by the b=3 workhorse size).
        public const int N_B4 = 1024, BR4 = 4, NB_B4 = N_B4 / BR4; // 256 blocks of 4x4

        public static uint Seed(int n, float density, int tag)
        {
            unchecked
            {
                int d = (int)math.round(density * 10000f);
                uint s = (uint)(n * 100003 + d * 131 + tag * 7919 + 12345);
                return s == 0 ? 1u : s;
            }
        }

        // ==== table formatters ==========================================================================

        public static string RowHeader() => string.Format("{0,-7} {1,-6} {2,7} {3,-20} {4,11} {5,11} {6,14}",
            "dtype", "N", "dens%", "path", "med(ms)", "min(ms)", "residual");

        public static string Row(string dtype, int n, float density, string path, Bench.Stat st, double residual) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,7:F1} {3,-20} {4,11:F4} {5,11:F4} {6,14:E3}",
                dtype, n, density * 100f, path, st.Median, st.Min, residual);

        public static string MatvecHeader() => string.Format("{0,-7} {1,-6} {2,7} {3,-12} {4,11} {5,11} {6,9} {7,12}",
            "dtype", "N", "dens%", "path", "med(ms)", "min(ms)", "speedup", "maxAbsDiff");

        public static string MatvecRow(string dtype, int n, float density, string path, Bench.Stat st, double speedup, double? maxAbsDiff)
        {
            string sp = string.Format(CultureInfo.InvariantCulture, "{0:F2}x", speedup);
            string md = maxAbsDiff.HasValue ? maxAbsDiff.Value.ToString("E2", CultureInfo.InvariantCulture) : "-";
            return string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,7:F1} {3,-12} {4,11:F4} {5,11:F4} {6,9} {7,12}",
                dtype, n, density * 100f, path, st.Median, st.Min, sp, md);
        }

        // ==== block-pattern choosers (dtype-independent: index/count logic only) =========================

        // Symmetric off-diagonal pairs (bi<bj); caller mirrors each into (bj,bi) via the transposed block,
        // so nnzb = nb (diagonal) + 2*pairs.Count.
        public static List<BlockPos> ChooseOffDiagPairsSymmetric(int nb, float density, uint seed, out int nnzb)
        {
            int nnzTarget = math.max(nb, (int)math.round(density * nb * nb));
            int offDiagTarget = math.max(0, nnzTarget - nb);
            int totalPairs = nb * (nb - 1) / 2;
            int pairsWanted = math.min(offDiagTarget / 2, totalPairs);

            var rng = new Random(seed);
            var seen = new HashSet<long>();
            var list = new List<BlockPos>(pairsWanted);
            while (list.Count < pairsWanted)
            {
                int bi = rng.NextInt(0, nb);
                int bj = rng.NextInt(0, nb);
                if (bi == bj) continue;
                if (bi > bj) { int t = bi; bi = bj; bj = t; }
                if (seen.Add((long)bi * nb + bj)) list.Add(new BlockPos(bi, bj));
            }

            nnzb = nb + list.Count * 2;
            return list;
        }

        // Ordered off-diagonal pairs, NOT mirrored -- yields a non-symmetric matrix.
        public static List<BlockPos> ChooseOffDiagPairsAsymmetric(int nb, float density, uint seed, out int nnzb)
        {
            int nnzTarget = math.max(nb, (int)math.round(density * nb * nb));
            int offDiagTarget = math.max(0, nnzTarget - nb);
            int totalOffDiag = nb * (nb - 1);
            offDiagTarget = math.min(offDiagTarget, totalOffDiag);

            var rng = new Random(seed);
            var seen = new HashSet<long>();
            var list = new List<BlockPos>(offDiagTarget);
            while (list.Count < offDiagTarget)
            {
                int bi = rng.NextInt(0, nb);
                int bj = rng.NextInt(0, nb);
                if (bi == bj) continue;
                if (seen.Add((long)bi * nb + bj)) list.Add(new BlockPos(bi, bj));
            }

            nnzb = nb + list.Count;
            return list;
        }

        // Rectangular mb x nb block grid; "diagonal" = (i,i) for i in [0, min(mb,nb)).
        public static List<BlockPos> ChooseOffDiagPairsRect(int mb, int nb, float density, uint seed, out int nnzb)
        {
            int diagCount = math.min(mb, nb);
            int nnzTarget = math.max(diagCount, (int)math.round(density * mb * nb));
            int offDiagTarget = math.max(0, nnzTarget - diagCount);
            int totalOffDiag = mb * nb - diagCount;
            offDiagTarget = math.min(offDiagTarget, totalOffDiag);

            var rng = new Random(seed);
            var seen = new HashSet<long>();
            var list = new List<BlockPos>(offDiagTarget);
            while (list.Count < offDiagTarget)
            {
                int bi = rng.NextInt(0, mb);
                int bj = rng.NextInt(0, nb);
                if (bi == bj && bi < diagCount) continue; // part of the guaranteed diagonal set
                if (seen.Add((long)bi * nb + bj)) list.Add(new BlockPos(bi, bj));
            }

            nnzb = diagCount + list.Count;
            return list;
        }
    }

    // ================================================================================================
    // Dense-vs-sparse iterative solver benchmark + numerical cross-check.
    //
    // The core method: for every case, ONE matrix is built with a block-sparsity pattern (block size
    // b=3, the FEM/cloth/PD workhorse) and materialized in BOTH storage forms -- a dense NxN
    // floatMxN/doubleMxN with zeros in the absent blocks, AND a floatBSR/doubleBSR (block-CSR) holding
    // exactly the nonzero blocks. Because both forms encode the IDENTICAL matrix:
    //   (a) dense-vs-sparse solve TIME is a fair, apples-to-apples comparison (same math, only the
    //       storage/traversal differs), and
    //   (b) dense-vs-sparse solve RESULTS must agree numerically -- the residual column is exactly
    //       that cross-check (always computed from the DENSE reference matrix/rhs).
    //
    // maxIterations is FIXED with tolerance=0, so every sample runs exactly K iterations -- deterministic
    // timing, mirroring IterativeBenchmark.cs's convention. Reporting the residual alongside the timing
    // shows both "how fast" and "how converged" (not just one or the other).
    //
    // Block density is at the BLOCK level (nb x nb block grid, b=3 scalar-per-side blocks): ~7% and
    // ~33% of blocks nonzero, always including every diagonal block (needed for conditioning /
    // solvability). Off-diagonal block magnitudes are kept small relative to the (diagonally-boosted)
    // diagonal blocks so the assembled systems stay diagonally dominant -- SPD for section 1, general
    // square for section 2, well-conditioned rectangular for section 3.
    //
    // Hand-written harness half. The timed IJobs, the residual + build helpers, and the per-section
    // build+measure methods are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/SparseSolverBenchmark.fProxy.cs.
    // ================================================================================================
    public static partial class SparseSolverBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-sparse-solvers.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Dense vs Sparse (BSR) iterative solvers: timing + numerical cross-check ===");
            sb.AppendLine("Same matrix, two storage forms: a dense NxN floatMxN/doubleMxN with zeros in the");
            sb.AppendLine("absent blocks, and a floatBSR/doubleBSR (block-CSR) holding exactly the nonzero");
            sb.AppendLine("b=3 blocks. Because both encode the IDENTICAL matrix, (a) dense-vs-sparse time is");
            sb.AppendLine("directly comparable (same math, only storage/traversal differs), and (b) dense-vs-");
            sb.AppendLine("sparse SOLUTIONS must agree numerically -- the residual column is that cross-check,");
            sb.AppendLine("always computed from the DENSE reference matrix. maxIterations is FIXED with");
            sb.AppendLine("tolerance=0, so every sample runs exactly K iterations (deterministic timing,");
            sb.AppendLine("mirroring IterativeBenchmark.cs); residual after K iterations shows how converged");
            sb.AppendLine("(not just how fast) each path is. Block density is at the BLOCK level (nb x nb");
            sb.AppendLine("block grid): ~7% / ~33% of blocks nonzero, always including every diagonal block.");
            sb.AppendLine("Section 0 first isolates the pure per-iteration operator cost (dense GEMV vs sparse");
            sb.AppendLine("spMV) that dominates every solver -- the cleanest dense-vs-sparse signal. Section 0b");
            sb.AppendLine("goes one level deeper: symmetric upper-block storage (Symmetric=true, ToBSRSymmetric)");
            sb.AppendLine("vs full block-CSR storage on the IDENTICAL SPD matrix -- bsrMatVecSym touches half as");
            sb.AppendLine("many stored blocks as the full traversal, so this isolates that ~2x memory/FLOP win.");
            sb.AppendLine("Section 1x is a dedicated N=1024 CG-only case at b=4 (256 blocks of 4x4, an unrolled");
            sb.AppendLine("kernel size) -- 1024 isn't divisible by the b=3 workhorse size Section 1 sweeps.");
            sb.AppendLine();

            Section0Float(sb);
            Section0Double(sb);
            Section0bFloat(sb);
            Section0bDouble(sb);
            Section1Float(sb);
            Section1Double(sb);
            Section1xFloat(sb);
            Section1xDouble(sb);
            Section2Float(sb);
            Section2Double(sb);
            Section3Float(sb);
            Section3Double(sb);
            Section4Float(sb);
            Section4Double(sb);
        }
    }
}
