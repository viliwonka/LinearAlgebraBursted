using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// BSR SpMM -- BSR.spMM / floatBSROperator.
// ApplyBlock now streams the matrix once and applies to k row-vectors together instead of looping
// k scalar BSR.spMV calls through two Allocator.Temp vectors.
//
// (a) Oracle: SpMM output must equal k separate BSR.spMV calls, ROW BY ROW. Every
//     bsrMatMat*/bsrMatMatSym* kernel (UnsafeOP.Sparse.float.cs) is documented to preserve its
//     scalar bsrMatVec*/bsrMatVecSym* counterpart's exact accumulation order per row (same pairing
//     where the scalar kernel pairs, same tail) -- asserted BIT-IDENTICAL (Assert.AreEqual on the
//     double-cast values), not just within tolerance. Swept over b in {1,2,3,4,6} (specialized) +
//     b=5 (general fallback), both storage modes (full + symmetric-upper), a rectangular-block
//     case (always the general fallback), and k in {1,3,8} (single row / mid / LOBPCG-scale).
// (b) LOBPCG results unchanged: since SpMM is row-for-row bit-identical to the OLD per-row-Apply
//     ApplyBlock it replaced, LOBPCG's whole trajectory (iterations, eigenvalues, eigenvectors)
//     must be bit-identical too -- checked directly against an in-test replica of the OLD
//     ApplyBlock (no need to check out pre-change history).
public class floatSparseSpMMTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct SparseSpMMTestJob : IJob
    {
        public enum TestType
        {
            SpMM_B1, SpMM_B2, SpMM_B3, SpMM_B4, SpMM_B6, SpMM_B5Fallback,
            SpMM_Sym_B1, SpMM_Sym_B2, SpMM_Sym_B3, SpMM_Sym_B4, SpMM_Sym_B6, SpMM_Sym_B5Fallback,
            SpMM_Rectangular,
            ApplyBlockForwardsToSpMM,
        }

        public TestType Type;

        // k values: single row-vector, mid-size, LOBPCG-scale multivector. Declared static readonly
        // (not an inline `int[]` in a method body): Burst rejects constructing a managed array at
        // runtime inside a job (BC1028) -- same idiom as floatKrylovFusedKernelTests.Sizes.
        static readonly int[] Ks = { 1, 3, 8 };

        public void Execute()
        {
            switch (Type)
            {
                case TestType.SpMM_B1: CheckSpMM(1, 101000u, false); break;
                case TestType.SpMM_B2: CheckSpMM(2, 102000u, false); break;
                case TestType.SpMM_B3: CheckSpMM(3, 103000u, false); break;
                case TestType.SpMM_B4: CheckSpMM(4, 104000u, false); break;
                case TestType.SpMM_B6: CheckSpMM(6, 106000u, false); break;
                case TestType.SpMM_B5Fallback: CheckSpMM(5, 105000u, false); break;

                case TestType.SpMM_Sym_B1: CheckSpMM(1, 111000u, true); break;
                case TestType.SpMM_Sym_B2: CheckSpMM(2, 112000u, true); break;
                case TestType.SpMM_Sym_B3: CheckSpMM(3, 113000u, true); break;
                case TestType.SpMM_Sym_B4: CheckSpMM(4, 114000u, true); break;
                case TestType.SpMM_Sym_B6: CheckSpMM(6, 116000u, true); break;
                case TestType.SpMM_Sym_B5Fallback: CheckSpMM(5, 115000u, true); break;

                case TestType.SpMM_Rectangular: CheckRectangular(); break;
                case TestType.ApplyBlockForwardsToSpMM: CheckApplyBlockForwards(); break;
            }
        }

        // ---- helpers (mirrors floatSparseUnrollTests' BuildRandomSquare/BuildRandomSymmetric
        // recipe -- same 4x4 block grid shape, distinct seed range) ----------------------------

        static floatBSR BuildRandomSquare(ref Arena arena, int b, uint seedBase)
        {
            var builder = arena.floatBSRBuilder(4, 4, b, b);
            builder.AddBlock(0, 0, arena.floatRandomMat(b, b, (float)(-1f), (float)1f, seedBase + 1u));
            builder.AddBlock(0, 2, arena.floatRandomMat(b, b, (float)(-1f), (float)1f, seedBase + 2u));
            builder.AddBlock(1, 1, arena.floatRandomMat(b, b, (float)(-1f), (float)1f, seedBase + 3u));
            builder.AddBlock(1, 3, arena.floatRandomMat(b, b, (float)(-1f), (float)1f, seedBase + 4u));
            builder.AddBlock(2, 0, arena.floatRandomMat(b, b, (float)(-1f), (float)1f, seedBase + 5u));
            builder.AddBlock(3, 3, arena.floatRandomMat(b, b, (float)(-1f), (float)1f, seedBase + 6u));
            return builder.ToBSR(ref arena);
        }

        static floatMxN SymDiagBlock(ref Arena arena, int b, uint seed)
        {
            var M = arena.floatRandomMat(b, b, (float)(-1f), (float)1f, seed);
            return Blas.dot(M, M, true);   // M^T M -- symmetric by construction
        }

        static floatBSR BuildRandomSymmetric(ref Arena arena, int b, uint seedBase)
        {
            var builder = arena.floatBSRBuilder(4, 4, b, b);
            builder.AddBlock(0, 0, SymDiagBlock(ref arena, b, seedBase + 1u));
            builder.AddBlock(1, 1, SymDiagBlock(ref arena, b, seedBase + 2u));
            builder.AddBlock(2, 2, SymDiagBlock(ref arena, b, seedBase + 3u));
            builder.AddBlock(3, 3, SymDiagBlock(ref arena, b, seedBase + 4u));
            builder.AddBlock(0, 1, arena.floatRandomMat(b, b, (float)(-1f), (float)1f, seedBase + 5u));
            builder.AddBlock(1, 3, arena.floatRandomMat(b, b, (float)(-1f), (float)1f, seedBase + 6u));
            builder.AddBlock(0, 3, arena.floatRandomMat(b, b, (float)(-1f), (float)1f, seedBase + 7u));
            return builder.ToBSRSymmetric(ref arena);
        }

        // AV[r,:] must equal BSR.spMV(A, V[r,:]) EXACTLY for every r in [0, rows).
        static void CheckSpMMAgainstSpMV(ref Arena arena, in floatBSR A, int rows, uint seed)
        {
            var V = arena.floatRandomMat(rows, A.N_Cols, (float)(-1f), (float)1f, seed);
            var AV = arena.floatMat(rows, A.M_Rows);
            BSR.spMM(in A, in V, ref AV, rows);

            var rowIn = arena.floatVec(A.N_Cols);
            var rowOut = arena.floatVec(A.M_Rows);
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++) rowIn[c] = V[r, c];
                BSR.spMV(in A, in rowIn, ref rowOut);
                for (int c = 0; c < A.M_Rows; c++)
                    Assert.AreEqual((double)rowOut[c], (double)AV[r, c]);   // bit-exact
            }
        }

        void CheckSpMM(int b, uint seedBase, bool symmetric)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = symmetric ? BuildRandomSymmetric(ref arena, b, seedBase) : BuildRandomSquare(ref arena, b, seedBase);
            for (int t = 0; t < Ks.Length; t++)
                CheckSpMMAgainstSpMV(ref arena, in A, Ks[t], seedBase + 800u + (uint)(t * 10));
            arena.Dispose();
        }

        // Rectangular blocks (BR != BC) always route through the general bsrMatMat fallback,
        // regardless of BR/BC individually matching a specialized size -- mirrors
        // floatSparseUnrollTests.CheckRectangularSpMV's boundary case.
        void CheckRectangular()
        {
            var arena = new Arena(Allocator.Persistent);
            const int BR = 2, BC = 3;
            var builder = arena.floatBSRBuilder(3, 3, BR, BC);
            builder.AddBlock(0, 0, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 121001));
            builder.AddBlock(0, 2, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 121002));
            builder.AddBlock(1, 1, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 121003));
            builder.AddBlock(2, 0, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 121004));
            var A = builder.ToBSR(ref arena);
            for (int t = 0; t < Ks.Length; t++)
                CheckSpMMAgainstSpMV(ref arena, in A, Ks[t], (uint)(121100 + t * 10));
            arena.Dispose();
        }

        // floatBSROperator.ApplyBlock is a one-line forward to BSR.spMM -- proves the wiring, not
        // just the kernel in isolation.
        void CheckApplyBlockForwards()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = BuildRandomSquare(ref arena, 3, 131000u);
            var op = new floatBSROperator(in A);

            int rows = 4;
            var V = arena.floatRandomMat(rows, A.N_Cols, (float)(-1f), (float)1f, 131900u);
            var AVdirect = arena.floatMat(rows, A.M_Rows);
            BSR.spMM(in A, in V, ref AVdirect, rows);

            var AVop = arena.floatMat(rows, A.M_Rows);
            op.ApplyBlock(in V, ref AVop, rows);

            for (int r = 0; r < rows; r++)
                for (int c = 0; c < A.M_Rows; c++)
                    Assert.AreEqual((double)AVdirect[r, c], (double)AVop[r, c]);

            arena.Dispose();
        }
    }

    // ---- (a) SpMM == k separate spMV calls, per block size / storage mode -----------------

    [Test] public void SpMM_B1_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.SpMM_B1 }.Run();
    [Test] public void SpMM_B2_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.SpMM_B2 }.Run();
    [Test] public void SpMM_B3_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.SpMM_B3 }.Run();
    [Test] public void SpMM_B4_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.SpMM_B4 }.Run();
    [Test] public void SpMM_B6_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.SpMM_B6 }.Run();
    [Test] public void SpMM_B5Fallback_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.SpMM_B5Fallback }.Run();

    [Test] public void SpMM_Sym_B1_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.SpMM_Sym_B1 }.Run();
    [Test] public void SpMM_Sym_B2_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.SpMM_Sym_B2 }.Run();
    [Test] public void SpMM_Sym_B3_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.SpMM_Sym_B3 }.Run();
    [Test] public void SpMM_Sym_B4_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.SpMM_Sym_B4 }.Run();
    [Test] public void SpMM_Sym_B6_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.SpMM_Sym_B6 }.Run();
    [Test] public void SpMM_Sym_B5Fallback_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.SpMM_Sym_B5Fallback }.Run();

    [Test] public void SpMM_Rectangular_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.SpMM_Rectangular }.Run();
    [Test] public void ApplyBlockForwardsToSpMM_Test() => new SparseSpMMTestJob { Type = SparseSpMMTestJob.TestType.ApplyBlockForwardsToSpMM }.Run();

    // ==============================================================================
    // (b) LOBPCG results unchanged by the SpMM kernel swap.
    // ==============================================================================

    // The "before" oracle: the exact scalar per-row loop floatBSROperator.ApplyBlock used before
    // BSR.spMM replaced it -- kept ONLY as an independent "pre-change" reference for the A/B test
    // below, not for production use.
    readonly struct OldStyleBSROperatorFloat : IfloatLinearOperator
    {
        public readonly floatBSR A;
        public OldStyleBSROperatorFloat(in floatBSR a) { A = a; }
        public int Rows => A.M_Rows;
        public int Cols => A.N_Cols;
        public void Apply(in floatN x, ref floatN y) => BSR.spMV(in A, in x, ref y);
        public void ApplyT(in floatN x, ref floatN y) => BSR.spMVT(in A, in x, ref y);
        public float ApplyDot(in floatN x, ref floatN y) => BSR.spMVDot(in A, in x, ref y);

        public void ApplyBlock(in floatMxN Vrows, ref floatMxN AVrows, int rows)
        {
            int cols = Vrows.N_Cols;
            var rin = new floatN(cols, Allocator.Temp, false);
            var rout = new floatN(cols, Allocator.Temp, false);
            for (int i = 0; i < rows; i++)
            {
                for (int c = 0; c < cols; c++) rin[c] = Vrows[i, c];
                Apply(in rin, ref rout);
                for (int c = 0; c < cols; c++) AVrows[i, c] = rout[c];
            }
            rout.Dispose();
            rin.Dispose();
        }
    }

    // Same 6x6 grid instance as floatLOBPCGSmokeTests.Laplacian2DGuardedMatchesAnalyticSmallest
    // (n=36, BR=6 -- a specialized SpMM block size), guard=0 (no guard vectors, so the cache is
    // exactly k rows -- the plain no-guard path per the class doc's own "bit-identical to the
    // pre-guard implementation" note). Managed [Test] (no Burst job), matching
    // floatLOBPCGSmokeTests' style -- this is a one-shot comparison, not a hot kernel.
    [Test]
    public void LOBPCGResultsUnchangedBySpMMKernelChange()
    {
        var arena = new Arena(Allocator.Persistent);

        int g = 6;
        int n = g * g;
        var A = arena.floatLaplacian2D(g, g);
        int k = 3;

        var wsOld = arena.floatLOBPCGCache(n, k);
        var infoOld = Eigen.lobpcg(new OldStyleBSROperatorFloat(in A), ref wsOld, k, Consts.floatSqrtEps, 1000);

        var wsNew = arena.floatLOBPCGCache(n, k);
        var infoNew = Eigen.lobpcg(in A, ref wsNew, k, Consts.floatSqrtEps, 1000);

        Assert.IsTrue(infoOld.Solved, infoOld.ToString());
        Assert.IsTrue(infoNew.Solved, infoNew.ToString());
        Assert.AreEqual(infoOld.iterations, infoNew.iterations);
        Assert.AreEqual(infoOld.converged, infoNew.converged);

        for (int i = 0; i < k; i++)
        {
            Assert.AreEqual((double)wsOld.lambda[i], (double)wsNew.lambda[i]);
            for (int c = 0; c < n; c++)
                Assert.AreEqual((double)wsOld.X[i, c], (double)wsNew.X[i, c]);
        }

        arena.Dispose();
    }
}
