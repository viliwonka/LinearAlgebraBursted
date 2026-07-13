using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Blas.dotSym (upper-triangle + mirror) and the symmetric matAtA route must match the full
// TransA kernel on symmetric-by-construction products, and their output must be EXACTLY
// symmetric. Sizes chosen to cross the 8x16 register-tile bulk, the tile-skip diagonal, and
// both remainder paths, plus a small whole-matrix-fallback case.
public class fProxyDotSymTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            DotSymMatchesFullKernel_Tiled,
            DotSymMatchesFullKernel_Small,
            AtAMatchesDistinctBufferKernel,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.DotSymMatchesFullKernel_Tiled: DotSymMatchesFullKernel(37, 40, 71001); break;
                case TestType.DotSymMatchesFullKernel_Small: DotSymMatchesFullKernel(5, 7, 71002); break;
                case TestType.AtAMatchesDistinctBufferKernel: AtAMatchesDistinctBufferKernel(); break;
            }
        }

        static fProxy Tol() => /*+choose[1e-4f|1e-10]*/1e-4f/*-choose*/;

        // Symmetric product by construction: W = S·A with S = MᵀM symmetric, then AᵀW = AᵀSA.
        void DotSymMatchesFullKernel(int m, int rows, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyRandomMat(rows, m, -1f, 1f, seed);
            var M = arena.fProxyRandomMat(rows, rows, -1f, 1f, seed + 1);
            var S = arena.fProxyMat(rows, rows);
            Blas.dot(in M, in M, ref S, transposeA: true);   // S = MᵀM, symmetric

            var W = arena.fProxyMat(rows, m);
            Blas.dot(in S, in A, ref W);                     // W = S·A

            var Cref = arena.fProxyMat(m, m);
            Blas.dot(in A, in W, ref Cref, transposeA: true);

            var Csym = arena.fProxyMat(m, m);
            Blas.dotSym(in A, in W, ref Csym);

            Assert.IsTrue(Analysis.isZero(Cref - Csym, Tol()));

            // Exact symmetry (mirrored, not just numerically close).
            for (int r = 0; r < m; r++)
                for (int c = 0; c < r; c++)
                    Assert.IsTrue(Csym[r, c] == Csym[c, r]);

            arena.Dispose();
        }

        // dot(A, A, transposeA) routes through the symmetric matAtA; it must match the full
        // TransA kernel fed two DISTINCT buffers holding the same values.
        void AtAMatchesDistinctBufferKernel()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 41, m = 23;
            var A = arena.fProxyRandomMat(rows, m, -1f, 1f, 72001);
            var A2 = A.Copy();

            var Cref = arena.fProxyMat(m, m);
            Blas.dot(in A, in A2, ref Cref, transposeA: true);   // distinct buffers: full kernel

            var C = arena.fProxyMat(m, m);
            Blas.dot(in A, in A, ref C, transposeA: true);        // same buffer: matAtA route

            Assert.IsTrue(Analysis.isZero(Cref - C, Tol()));
            for (int r = 0; r < m; r++)
                for (int c = 0; c < r; c++)
                    Assert.IsTrue(C[r, c] == C[c, r]);

            arena.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void Test(TestJob.TestType type)
    {
        new TestJob() { Type = type }.Run();
    }
}
