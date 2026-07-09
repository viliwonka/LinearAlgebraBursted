using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of TriangularSolveBenchmark (timed IJobs + build+measure method). The
    // dtype-agnostic harness (Sizes, Run, Section) and the shared row formatter (TriSolveFmt) are
    // hand-written in Assets/LinearAlgebra/Benchmarks/TriangularSolveBenchmark.cs.

    // Factors A ONCE into the compact LU form (with pivot) -- run via a plain .Run() call OUTSIDE
    // Bench.Time in the build+measure method below, so the O(n^3) factorization cost is never part of
    // any timed sample. Every solve-only job further down reuses this SAME A/P read-only (none of
    // decompSolve/decompSolveTransA/triUpperLU/triUpperLUTransA modify the factor or the pivot).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TriSolveFactorJobFProxy : IJob
    {
        public fProxyMxN A;     // receives Src, factored in place into the compact LU form
        public fProxyMxN Src;
        public Pivot P;

        public void Execute()
        {
            A.Data.CopyFrom(Src.Data);
            LU.decompInPlace(ref A, ref P);
        }
    }

    // Forward vector solve only: LU.decompSolve (compact form; triLowerLU then triUpperLU) against the
    // pre-factored A/P. Re-copies bSrc into b every Execute (timed) since the solve overwrites b_to_x
    // in place -- otherwise rep 2+ would solve the previous rep's already-solved output.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TriSolveFwdVecJobFProxy : IJob
    {
        public fProxyMxN A;     // pre-factored compact LU (read-only here)
        public Pivot P;         // pre-computed pivot (read-only here)
        public fProxyN b;
        public fProxyN bSrc;

        public void Execute()
        {
            for (int i = 0; i < bSrc.N; i++) b[i] = bSrc[i];
            LU.decompSolve(ref A, in P, ref b);
        }
    }

    // Transposed vector solve only: LU.decompSolveTransA (compact form; triUpperLUTransA then
    // triLowerLUTransA, then the pivot scatter) against the SAME pre-factored A/P.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TriSolveTransAVecJobFProxy : IJob
    {
        public fProxyMxN A;
        public Pivot P;
        public fProxyN b;
        public fProxyN bSrc;

        public void Execute()
        {
            for (int i = 0; i < bSrc.N; i++) b[i] = bSrc[i];
            LU.decompSolveTransA(ref A, in P, ref b);
        }
    }

    // Forward matrix-RHS solve only (K columns): LU.decompSolve multi-RHS (TRSM-shaped) overload.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TriSolveFwdMatJobFProxy : IJob
    {
        public fProxyMxN A;
        public Pivot P;
        public fProxyMxN BX;
        public fProxyMxN BXsrc;

        public void Execute()
        {
            BX.Data.CopyFrom(BXsrc.Data);
            LU.decompSolve(ref A, in P, ref BX);
        }
    }

    // Transposed matrix-RHS solve only (K columns): LU.decompSolveTransA multi-RHS overload.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TriSolveTransAMatJobFProxy : IJob
    {
        public fProxyMxN A;
        public Pivot P;
        public fProxyMxN BX;
        public fProxyMxN BXsrc;

        public void Execute()
        {
            BX.Data.CopyFrom(BXsrc.Data);
            LU.decompSolveTransA(ref A, in P, ref BX);
        }
    }

    // ---- underlying-kernel isolation: ONE triangular pass per direction, no pivot gather/scatter and
    // no second (lower / Lᵀ) pass -- separates the pivot-indirected row-dot back-substitution
    // (triUpperLU) from the pivot-indirected right-looking/axpy forward step (triUpperLUTransA), and
    // from the pivot-apply cost the full decompSolve/decompSolveTransA rows above include.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TriSolveKernelFwdVecJobFProxy : IJob
    {
        public fProxyMxN A;
        public Pivot P;
        public fProxyN b;
        public fProxyN bSrc;

        public void Execute()
        {
            for (int i = 0; i < bSrc.N; i++) b[i] = bSrc[i];
            Blas.triUpperLU(ref A, in P, ref b);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TriSolveKernelTransAVecJobFProxy : IJob
    {
        public fProxyMxN A;
        public Pivot P;
        public fProxyN b;
        public fProxyN bSrc;

        public void Execute()
        {
            for (int i = 0; i < bSrc.N; i++) b[i] = bSrc[i];
            Blas.triUpperLUTransA(ref A, in P, ref b);
        }
    }

    public static partial class TriangularSolveBenchmark
    {
        static string TriSolveFProxy(int n)
        {
            // Fixed RHS width for the multi-RHS (TRSM-shaped) rows; a method-local const (not a
            // class-level one) since this partial class is shared by the float/double generated files
            // (a class-level const of the same name would collide across them, CS0102 -- see LU_BLOCK
            // in LU.fProxy.cs for the same convention).
            const int K = 8;

            var arena = new Arena(Allocator.Persistent);
            var Src = arena.fProxyMat(n, n);
            var A = arena.fProxyMat(n, n);
            var P = new Pivot(n, Allocator.Persistent);

            var b = arena.fProxyVec(n);
            var bSrc = arena.fProxyVec(n);
            var BX = arena.fProxyMat(n, K);
            var BXsrc = arena.fProxyMat(n, K);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;                     // diagonal dominance => well-conditioned, full rank
            for (int i = 0; i < n; i++)
                bSrc[i] = rng.NextFProxy(-1f, 1f);
            for (int i = 0; i < n; i++)
                for (int c = 0; c < K; c++)
                    BXsrc[i, c] = rng.NextFProxy(-1f, 1f);

            // Factor ONCE, outside every timed region below -- shared read-only by every job that follows.
            new TriSolveFactorJobFProxy { A = A, Src = Src, P = P }.Run();

            var fwdVec = Bench.Time(() => new TriSolveFwdVecJobFProxy { A = A, P = P, b = b, bSrc = bSrc }.Run());
            var transAVec = Bench.Time(() => new TriSolveTransAVecJobFProxy { A = A, P = P, b = b, bSrc = bSrc }.Run());
            var fwdMat = Bench.Time(() => new TriSolveFwdMatJobFProxy { A = A, P = P, BX = BX, BXsrc = BXsrc }.Run());
            var transAMat = Bench.Time(() => new TriSolveTransAMatJobFProxy { A = A, P = P, BX = BX, BXsrc = BXsrc }.Run());
            var kernelFwdVec = Bench.Time(() => new TriSolveKernelFwdVecJobFProxy { A = A, P = P, b = b, bSrc = bSrc }.Run());
            var kernelTransAVec = Bench.Time(() => new TriSolveKernelTransAVecJobFProxy { A = A, P = P, b = b, bSrc = bSrc }.Run());

            P.Dispose();
            arena.Dispose();

            return TriSolveFmt.RowKernel("fProxy", "LU solve fwd (vec)", n, fwdVec)
                 + "\n" + TriSolveFmt.RowKernel("fProxy", "LU solve TransA (vec)", n, transAVec)
                 + "\n" + TriSolveFmt.RowKernel("fProxy", "LU solve fwd (mat k=8)", n, fwdMat)
                 + "\n" + TriSolveFmt.RowKernel("fProxy", "LU solve TransA (mat k=8)", n, transAMat)
                 + "\n" + TriSolveFmt.RowKernel("fProxy", "Blas triUpperLU (vec)", n, kernelFwdVec)
                 + "\n" + TriSolveFmt.RowKernel("fProxy", "Blas triUpperLUTransA (vec)", n, kernelTransAVec);
        }
    }
}
