using System.Collections.Generic;
using System.Globalization;
using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Sparse;

namespace LinearAlgebra.Benchmarks
{
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
    // ================================================================================================

    // ---- CG scratch: r, p, Ap (all A.Rows length) --------------------------------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGDenseJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x, r, p, Ap;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Solvers.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGSparseJobFloat : IJob
    {
        public floatBSR A;
        public floatN b, x, r, p, Ap;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Solvers.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGDenseJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x, r, p, Ap;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0.0;
            Solvers.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0.0);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN b, x, r, p, Ap;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0.0;
            Solvers.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0.0);
        }
    }

    // ---- MINRES scratch: y, r1, r2, v, w, w1, w2 (all A.Rows length) -------------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MinresDenseJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x, y, r1, r2, v, w, w1, w2;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Solvers.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MinresSparseJobFloat : IJob
    {
        public floatBSR A;
        public floatN b, x, y, r1, r2, v, w, w1, w2;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Solvers.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MinresDenseJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x, y, r1, r2, v, w, w1, w2;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0.0;
            Solvers.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, K, 0.0);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MinresSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN b, x, y, r1, r2, v, w, w1, w2;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0.0;
            Solvers.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, K, 0.0);
        }
    }

    // ---- BiCGSTAB scratch: r, rHat0, p, v, t (all A.Rows length) -----------------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BiCGStabDenseJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x, r, rHat0, p, v, t;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Solvers.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BiCGStabSparseJobFloat : IJob
    {
        public floatBSR A;
        public floatN b, x, r, rHat0, p, v, t;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Solvers.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BiCGStabDenseJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x, r, rHat0, p, v, t;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0.0;
            Solvers.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, K, 0.0);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BiCGStabSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN b, x, r, rHat0, p, v, t;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0.0;
            Solvers.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, K, 0.0);
        }
    }

    // ---- CGLS scratch: r, q (A.Rows length), s, p (A.Cols length) ----------------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CglsDenseJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x, r, s, p, q;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Solvers.cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CglsSparseJobFloat : IJob
    {
        public floatBSR A;
        public floatN b, x, r, s, p, q;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Solvers.cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CglsDenseJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x, r, s, p, q;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0.0;
            Solvers.cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, K, 0.0);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CglsSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN b, x, r, s, p, q;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0.0;
            Solvers.cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, K, 0.0);
        }
    }

    // ---- LSQR scratch: u, tmpM (A.Rows length), v, w, tmpN (A.Cols length) ------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrDenseJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x, u, v, w, tmpM, tmpN;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Solvers.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrSparseJobFloat : IJob
    {
        public floatBSR A;
        public floatN b, x, u, v, w, tmpM, tmpN;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Solvers.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrDenseJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x, u, v, w, tmpM, tmpN;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0.0;
            Solvers.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0.0);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN b, x, u, v, w, tmpM, tmpN;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0.0;
            Solvers.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0.0);
        }
    }

    // ---- transpose-optimized sparse CGLS/LSQR jobs (Milestone B): use a materialized Aᵀ so ApplyT
    //      runs as a forward spMV over Aᵀ instead of the cache-unfriendly on-the-fly spMVT. Aᵀ is built
    //      ONCE outside the timed region (a real solve builds it once and reuses it every iteration),
    //      so the timing isolates the per-iteration ApplyT improvement from the one-time build cost. ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CglsSparseTJobFloat : IJob
    {
        public floatBSR A, AT;
        public floatN b, x, r, s, p, q;
        public int K;
        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Solvers.cgls(in A, in AT, in b, ref x, ref r, ref s, ref p, ref q, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CglsSparseTJobDouble : IJob
    {
        public doubleBSR A, AT;
        public doubleN b, x, r, s, p, q;
        public int K;
        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0.0;
            Solvers.cgls(in A, in AT, in b, ref x, ref r, ref s, ref p, ref q, K, 0.0);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrSparseTJobFloat : IJob
    {
        public floatBSR A, AT;
        public floatN b, x, u, v, w, tmpM, tmpN;
        public int K;
        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Solvers.lsqr(in A, in AT, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrSparseTJobDouble : IJob
    {
        public doubleBSR A, AT;
        public doubleN b, x, u, v, w, tmpM, tmpN;
        public int K;
        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0.0;
            Solvers.lsqr(in A, in AT, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0.0);
        }
    }

    // ---- Section 4: hand-inlined dense CG (no IfloatLinearOperator/IdoubleLinearOperator, no cg<TOp>
    //      generic dispatch -- a raw GEMV loop + axpy/dot written directly in Execute()). Same algorithm
    //      as Solvers.cg<TOp> (see Solvers.fProxy.cs), just with every step spelled out inline against
    //      raw pointers instead of going through fProxyDenseOperator.Apply / the generic solver loop.
    //      x is reset to zero and tol is effectively 0 (K fixed iterations), matching the other jobs. ---

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGHandInlinedJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x, r, p, Ap;
        public int K;

        public unsafe void Execute()
        {
            int n = x.N;
            float* Ad = A.Data.Ptr;
            float* bd = b.Data.Ptr;
            float* xd = x.Data.Ptr;
            float* rd = r.Data.Ptr;
            float* pd = p.Data.Ptr;
            float* Apd = Ap.Data.Ptr;

            for (int i = 0; i < n; i++) xd[i] = 0f;
            for (int i = 0; i < n; i++) rd[i] = bd[i];       // r = b - A*0 = b
            for (int i = 0; i < n; i++) pd[i] = rd[i];       // p = r

            float rsold = 0f;
            for (int i = 0; i < n; i++) rsold += rd[i] * rd[i];

            for (int k = 0; k < K; k++)
            {
                for (int row = 0; row < n; row++)
                {
                    float sum = 0f;
                    int baseIdx = row * n;
                    for (int col = 0; col < n; col++)
                        sum += Ad[baseIdx + col] * pd[col];
                    Apd[row] = sum;
                }

                float pAp = 0f;
                for (int i = 0; i < n; i++) pAp += pd[i] * Apd[i];
                if (!(pAp > 0f)) break;

                float alpha = rsold / pAp;
                for (int i = 0; i < n; i++) xd[i] += alpha * pd[i];
                for (int i = 0; i < n; i++) rd[i] -= alpha * Apd[i];

                float rsnew = 0f;
                for (int i = 0; i < n; i++) rsnew += rd[i] * rd[i];

                float beta = rsnew / rsold;
                for (int i = 0; i < n; i++) pd[i] = beta * pd[i] + rd[i];

                rsold = rsnew;
            }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGHandInlinedJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x, r, p, Ap;
        public int K;

        public unsafe void Execute()
        {
            int n = x.N;
            double* Ad = A.Data.Ptr;
            double* bd = b.Data.Ptr;
            double* xd = x.Data.Ptr;
            double* rd = r.Data.Ptr;
            double* pd = p.Data.Ptr;
            double* Apd = Ap.Data.Ptr;

            for (int i = 0; i < n; i++) xd[i] = 0.0;
            for (int i = 0; i < n; i++) rd[i] = bd[i];
            for (int i = 0; i < n; i++) pd[i] = rd[i];

            double rsold = 0.0;
            for (int i = 0; i < n; i++) rsold += rd[i] * rd[i];

            for (int k = 0; k < K; k++)
            {
                for (int row = 0; row < n; row++)
                {
                    double sum = 0.0;
                    int baseIdx = row * n;
                    for (int col = 0; col < n; col++)
                        sum += Ad[baseIdx + col] * pd[col];
                    Apd[row] = sum;
                }

                double pAp = 0.0;
                for (int i = 0; i < n; i++) pAp += pd[i] * Apd[i];
                if (!(pAp > 0.0)) break;

                double alpha = rsold / pAp;
                for (int i = 0; i < n; i++) xd[i] += alpha * pd[i];
                for (int i = 0; i < n; i++) rd[i] -= alpha * Apd[i];

                double rsnew = 0.0;
                for (int i = 0; i < n; i++) rsnew += rd[i] * rd[i];

                double beta = rsnew / rsold;
                for (int i = 0; i < n; i++) pd[i] = beta * pd[i] + rd[i];

                rsold = rsnew;
            }
        }
    }

    // ---- operator matvec microbench jobs (REPS back-to-back matvecs, zero-alloc) -------------------
    //
    // Isolate the per-iteration operator cost -- dense GEMV (Blas.dot) vs sparse spMV -- that
    // dominates every Krylov iteration, with NO convergence/breakdown variability. The reps loop
    // PING-PONGS x<->y (each matvec feeds the next) specifically to defeat Burst dead-store
    // elimination: if every rep just overwrote the same y that nothing reads between reps, the
    // optimizer could collapse REPS matvecs down to one. Values may diverge to Inf across the chain
    // (the SPD system is diagonally dominant, radius >> 1) -- irrelevant to TIMING (Inf/NaN float ops
    // cost the same), and the numerical cross-check (maxAbsDiff) is computed separately from a clean
    // single untimed matvec, not from these buffers.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MatvecDenseJobFloat : IJob
    {
        public floatMxN A;
        public floatN x, y;
        public int reps;
        public void Execute()
        {
            for (int k = 0; k < reps; k++)
            {
                if ((k & 1) == 0) Blas.dot(in A, in x, ref y);
                else              Blas.dot(in A, in y, ref x);
            }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MatvecSparseJobFloat : IJob
    {
        public floatBSR A;
        public floatN x, y;
        public int reps;
        public void Execute()
        {
            for (int k = 0; k < reps; k++)
            {
                if ((k & 1) == 0) BSR.spMV(in A, in x, ref y);
                else              BSR.spMV(in A, in y, ref x);
            }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MatvecDenseJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN x, y;
        public int reps;
        public void Execute()
        {
            for (int k = 0; k < reps; k++)
            {
                if ((k & 1) == 0) Blas.dot(in A, in x, ref y);
                else              Blas.dot(in A, in y, ref x);
            }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MatvecSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN x, y;
        public int reps;
        public void Execute()
        {
            for (int k = 0; k < reps; k++)
            {
                if ((k & 1) == 0) BSR.spMV(in A, in x, ref y);
                else              BSR.spMV(in A, in y, ref x);
            }
        }
    }

    public static class SparseSolverBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-sparse-solvers.txt", Section);

        // Block-aligned sizes (N = nb * BR). All three sizes fit comfortably within a few minutes for
        // this section (see the report's own timings); if a future change makes this section too slow,
        // drop 768 or lower Bench.Runs for just this section and note it here.
        static readonly int[] BlockSizesN = { 192, 384, 768 };
        const int BR = 3; // block size b=3 (the FEM/cloth/PD workhorse)
        static readonly float[] Densities = { 0.07f, 0.33f }; // ~7% / ~33% of BLOCKS nonzero

        const int K_CG = 40;        // CG / MINRES iteration budget (fixed, tol=0)
        const int K_BICGSTAB = 40;  // BiCGSTAB iteration budget
        const int K_LS = 24;        // CGLS / LSQR iteration budget
        const int REPS_MATVEC = 64; // operator microbench: back-to-back matvecs per timed sample

        // ---- small position record used by the block-pattern choosers below (avoids depending on
        //      ValueTuple support one way or the other) ----
        readonly struct BlockPos
        {
            public readonly int Bi, Bj;
            public BlockPos(int bi, int bj) { Bi = bi; Bj = bj; }
        }

        static uint Seed(int n, float density, int tag)
        {
            unchecked
            {
                int d = (int)math.round(density * 10000f);
                uint s = (uint)(n * 100003 + d * 131 + tag * 7919 + 12345);
                return s == 0 ? 1u : s;
            }
        }

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

        static string RowHeader() => string.Format("{0,-7} {1,-6} {2,7} {3,-20} {4,11} {5,11} {6,14}",
            "dtype", "N", "dens%", "path", "med(ms)", "min(ms)", "residual");

        static string Row(string dtype, int n, float density, string path, Bench.Stat st, double residual) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,7:F1} {3,-20} {4,11:F4} {5,11:F4} {6,14:E3}",
                dtype, n, density * 100f, path, st.Median, st.Min, residual);

        // ==== residual helpers (always evaluated against the DENSE reference matrix) ===================

        static double ResidualLinSys(in floatMxN A, in floatN x, in floatN b)
        {
            var Ax = Blas.dot(A, x);
            double num = 0, den = 0;
            for (int i = 0; i < b.N; i++)
            {
                double diff = (double)Ax[i] - (double)b[i];
                num += diff * diff;
                den += (double)b[i] * (double)b[i];
            }
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }

        static double ResidualLinSys(in doubleMxN A, in doubleN x, in doubleN b)
        {
            var Ax = Blas.dot(A, x);
            double num = 0, den = 0;
            for (int i = 0; i < b.N; i++)
            {
                double diff = Ax[i] - b[i];
                num += diff * diff;
                den += b[i] * b[i];
            }
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }

        // Least-squares optimality: ||A^T(Ax-b)|| / ||A^T b|| -- the correct acceptance criterion for
        // a (possibly inconsistent) rectangular system, NOT ||Ax-b|| (nonzero even at the LS optimum).
        static double ResidualLS(in floatMxN A, in floatN x, in floatN b)
        {
            var Ax = Blas.dot(A, x);
            var res = Ax - b;
            var atr = Blas.dot(res, A);
            var atb = Blas.dot(b, A);
            double num = 0, den = 0;
            for (int i = 0; i < atr.N; i++) num += (double)atr[i] * (double)atr[i];
            for (int i = 0; i < atb.N; i++) den += (double)atb[i] * (double)atb[i];
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }

        static double ResidualLS(in doubleMxN A, in doubleN x, in doubleN b)
        {
            var Ax = Blas.dot(A, x);
            var res = Ax - b;
            var atr = Blas.dot(res, A);
            var atb = Blas.dot(b, A);
            double num = 0, den = 0;
            for (int i = 0; i < atr.N; i++) num += atr[i] * atr[i];
            for (int i = 0; i < atb.N; i++) den += atb[i] * atb[i];
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }

        // ==== block-pattern choosers (dtype-independent: index/count logic only) =======================

        // Symmetric off-diagonal pairs (bi<bj); caller mirrors each into (bj,bi) via the transposed
        // block, so nnzb = nb (diagonal) + 2*pairs.Count.
        static List<BlockPos> ChooseOffDiagPairsSymmetric(int nb, float density, uint seed, out int nnzb)
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
        static List<BlockPos> ChooseOffDiagPairsAsymmetric(int nb, float density, uint seed, out int nnzb)
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
        static List<BlockPos> ChooseOffDiagPairsRect(int mb, int nb, float density, uint seed, out int nnzb)
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

        // ==== block-matrix builders (float) =============================================================

        static void BuildBlockSPDFloat(ref Arena arena, int nb, float density, uint seed, out floatMxN dense, out floatBSR sparse)
        {
            int dim = nb * BR;
            dense = arena.floatMat(dim, dim);
            var pairs = ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzb);
            var builder = arena.floatBSRBuilder(nb, nb, BR, BR, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            float strong = dim;
            const float offScale = 0.3f;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextFloat(-1f, 1f);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < BR; d++) Di[d, d] += strong;

                builder.AddBlock(i, i, in Di);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = Di[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextFloat(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];

                var blockT = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        blockT[r, c] = block[c, r];

                builder.AddBlock(bj, bi, in blockT);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bj * BR + r, bi * BR + c] = blockT[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        // Same recipe as BuildBlockSPDFloat (identical rng sequence: diagonal blocks Di = Mi^T Mi +
        // strong*I, then off-diagonal pairs at offScale), but assembles TWO block-CSR encodings of
        // the SAME dense SPD matrix side by side: `full` (every stored block, incl. the explicit
        // mirrored lower block bj,bi) and `sym` (upper-triangle + diagonal ONLY, via
        // ToBSRSymmetric -- the lower triangle is implicit). Used by Section 0b to isolate the
        // symmetric-storage spMV win (bsrMatVecSym does half the stored-block work of bsrMatVec)
        // on a matrix that is byte-for-byte identical between the two storage forms.
        static void BuildBlockSPDPairFloat(ref Arena arena, int nb, float density, uint seed,
                                           out floatMxN dense, out floatBSR full, out floatBSR sym)
        {
            int dim = nb * BR;
            dense = arena.floatMat(dim, dim);
            var pairs = ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzbFull);
            int nnzbSym = nb + pairs.Count;
            var fullBuilder = arena.floatBSRBuilder(nb, nb, BR, BR, nnzbFull);
            var symBuilder = arena.floatBSRBuilder(nb, nb, BR, BR, nnzbSym);
            var rng = new Random(seed ^ 0x9E3779B9u);
            float strong = dim;
            const float offScale = 0.3f;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextFloat(-1f, 1f);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < BR; d++) Di[d, d] += strong;

                fullBuilder.AddBlock(i, i, in Di);
                symBuilder.AddBlock(i, i, in Di);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = Di[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextFloat(-offScale, offScale);

                fullBuilder.AddBlock(bi, bj, in block);
                symBuilder.AddBlock(bi, bj, in block);   // upper only -- sym never sees the mirror
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];

                var blockT = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        blockT[r, c] = block[c, r];

                fullBuilder.AddBlock(bj, bi, in blockT); // mirrored lower block -- FULL storage only
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bj * BR + r, bi * BR + c] = blockT[r, c];
            }

            full = fullBuilder.ToBSR(ref arena);
            sym = symBuilder.ToBSRSymmetric(ref arena);
        }

        static void BuildBlockNonSymFloat(ref Arena arena, int nb, float density, uint seed, out floatMxN dense, out floatBSR sparse)
        {
            int dim = nb * BR;
            dense = arena.floatMat(dim, dim);
            var pairs = ChooseOffDiagPairsAsymmetric(nb, density, seed, out int nnzb);
            var builder = arena.floatBSRBuilder(nb, nb, BR, BR, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            float strong = dim;
            const float offScale = 0.3f;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextFloat(-1f, 1f);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < BR; d++) Di[d, d] += strong;

                builder.AddBlock(i, i, in Di);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = Di[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextFloat(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        static void BuildBlockRectFloat(ref Arena arena, int mb, int nb, float density, uint seed, out floatMxN dense, out floatBSR sparse)
        {
            int rows = mb * BR, cols = nb * BR;
            dense = arena.floatMat(rows, cols);
            int diagCount = math.min(mb, nb);
            var pairs = ChooseOffDiagPairsRect(mb, nb, density, seed, out int nnzb);
            var builder = arena.floatBSRBuilder(mb, nb, BR, BR, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);

            for (int i = 0; i < diagCount; i++)
            {
                var block = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = (r == c ? 2f : 0f) + rng.NextFloat(-0.2f, 0.2f);

                builder.AddBlock(i, i, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = block[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextFloat(-0.3f, 0.3f);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        // ==== block-matrix builders (double) =============================================================

        static void BuildBlockSPDDouble(ref Arena arena, int nb, float density, uint seed, out doubleMxN dense, out doubleBSR sparse)
        {
            int dim = nb * BR;
            dense = arena.doubleMat(dim, dim);
            var pairs = ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzb);
            var builder = arena.doubleBSRBuilder(nb, nb, BR, BR, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            double strong = dim;
            const double offScale = 0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextDouble(-1.0, 1.0);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < BR; d++) Di[d, d] += strong;

                builder.AddBlock(i, i, in Di);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = Di[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextDouble(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];

                var blockT = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        blockT[r, c] = block[c, r];

                builder.AddBlock(bj, bi, in blockT);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bj * BR + r, bi * BR + c] = blockT[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        // Double counterpart of BuildBlockSPDPairFloat -- see that method's doc comment.
        static void BuildBlockSPDPairDouble(ref Arena arena, int nb, float density, uint seed,
                                            out doubleMxN dense, out doubleBSR full, out doubleBSR sym)
        {
            int dim = nb * BR;
            dense = arena.doubleMat(dim, dim);
            var pairs = ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzbFull);
            int nnzbSym = nb + pairs.Count;
            var fullBuilder = arena.doubleBSRBuilder(nb, nb, BR, BR, nnzbFull);
            var symBuilder = arena.doubleBSRBuilder(nb, nb, BR, BR, nnzbSym);
            var rng = new Random(seed ^ 0x9E3779B9u);
            double strong = dim;
            const double offScale = 0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextDouble(-1.0, 1.0);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < BR; d++) Di[d, d] += strong;

                fullBuilder.AddBlock(i, i, in Di);
                symBuilder.AddBlock(i, i, in Di);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = Di[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextDouble(-offScale, offScale);

                fullBuilder.AddBlock(bi, bj, in block);
                symBuilder.AddBlock(bi, bj, in block);   // upper only -- sym never sees the mirror
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];

                var blockT = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        blockT[r, c] = block[c, r];

                fullBuilder.AddBlock(bj, bi, in blockT); // mirrored lower block -- FULL storage only
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bj * BR + r, bi * BR + c] = blockT[r, c];
            }

            full = fullBuilder.ToBSR(ref arena);
            sym = symBuilder.ToBSRSymmetric(ref arena);
        }

        static void BuildBlockNonSymDouble(ref Arena arena, int nb, float density, uint seed, out doubleMxN dense, out doubleBSR sparse)
        {
            int dim = nb * BR;
            dense = arena.doubleMat(dim, dim);
            var pairs = ChooseOffDiagPairsAsymmetric(nb, density, seed, out int nnzb);
            var builder = arena.doubleBSRBuilder(nb, nb, BR, BR, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            double strong = dim;
            const double offScale = 0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextDouble(-1.0, 1.0);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < BR; d++) Di[d, d] += strong;

                builder.AddBlock(i, i, in Di);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = Di[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextDouble(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        static void BuildBlockRectDouble(ref Arena arena, int mb, int nb, float density, uint seed, out doubleMxN dense, out doubleBSR sparse)
        {
            int rows = mb * BR, cols = nb * BR;
            dense = arena.doubleMat(rows, cols);
            int diagCount = math.min(mb, nb);
            var pairs = ChooseOffDiagPairsRect(mb, nb, density, seed, out int nnzb);
            var builder = arena.doubleBSRBuilder(mb, nb, BR, BR, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);

            for (int i = 0; i < diagCount; i++)
            {
                var block = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = (r == c ? 2.0 : 0.0) + rng.NextDouble(-0.2, 0.2);

                builder.AddBlock(i, i, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = block[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextDouble(-0.3, 0.3);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        // ==== block-matrix builders, parameterized block size (used ONLY by the dedicated b=4/N=1024 =====
        //      subsection below -- Section 1 above keeps its own BR=3-hardcoded builders untouched since
        //      its numbers are cited in docs). Same recipe as BuildBlockSPDFloat/Double, generalized so
        //      the block size isn't tied to the file-wide BR=3 constant.

        static void BuildBlockSPDFloatSized(ref Arena arena, int nb, int br, float density, uint seed, out floatMxN dense, out floatBSR sparse)
        {
            int dim = nb * br;
            dense = arena.floatMat(dim, dim);
            var pairs = ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzb);
            var builder = arena.floatBSRBuilder(nb, nb, br, br, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            float strong = dim;
            const float offScale = 0.3f;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.floatMat(br, br);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        Mi[r, c] = rng.NextFloat(-1f, 1f);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < br; d++) Di[d, d] += strong;

                builder.AddBlock(i, i, in Di);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        dense[i * br + r, i * br + c] = Di[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.floatMat(br, br);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        block[r, c] = rng.NextFloat(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        dense[bi * br + r, bj * br + c] = block[r, c];

                var blockT = arena.floatMat(br, br);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        blockT[r, c] = block[c, r];

                builder.AddBlock(bj, bi, in blockT);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        dense[bj * br + r, bi * br + c] = blockT[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        static void BuildBlockSPDDoubleSized(ref Arena arena, int nb, int br, float density, uint seed, out doubleMxN dense, out doubleBSR sparse)
        {
            int dim = nb * br;
            dense = arena.doubleMat(dim, dim);
            var pairs = ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzb);
            var builder = arena.doubleBSRBuilder(nb, nb, br, br, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            double strong = dim;
            const double offScale = 0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.doubleMat(br, br);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        Mi[r, c] = rng.NextDouble(-1.0, 1.0);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < br; d++) Di[d, d] += strong;

                builder.AddBlock(i, i, in Di);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        dense[i * br + r, i * br + c] = Di[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.doubleMat(br, br);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        block[r, c] = rng.NextDouble(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        dense[bi * br + r, bj * br + c] = block[r, c];

                var blockT = arena.doubleMat(br, br);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        blockT[r, c] = block[c, r];

                builder.AddBlock(bj, bi, in blockT);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        dense[bj * br + r, bi * br + c] = blockT[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        // ==== Section 0: operator matvec throughput (dense GEMV vs sparse spMV) =========================
        //
        // The purest dense-vs-sparse signal: REPS back-to-back matvecs y = A x in each storage form on
        // the SAME SPD system Section 1 uses. This is the per-iteration operator cost that dominates
        // every Krylov solver, isolated from convergence. speedup = dense_med / this_row_med (dense row
        // = 1.00x baseline; >1 means sparse is faster). maxAbsDiff = max_i |y_dense - y_sparse| from a
        // clean single untimed matvec on identical input -- must be ~0 (both forms encode one matrix).

        static string MatvecHeader() => string.Format("{0,-7} {1,-6} {2,7} {3,-12} {4,11} {5,11} {6,9} {7,12}",
            "dtype", "N", "dens%", "path", "med(ms)", "min(ms)", "speedup", "maxAbsDiff");

        static string MatvecRow(string dtype, int n, float density, string path, Bench.Stat st, double speedup, double? maxAbsDiff)
        {
            string sp = string.Format(CultureInfo.InvariantCulture, "{0:F2}x", speedup);
            string md = maxAbsDiff.HasValue ? maxAbsDiff.Value.ToString("E2", CultureInfo.InvariantCulture) : "-";
            return string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,7:F1} {3,-12} {4,11:F4} {5,11:F4} {6,9} {7,12}",
                dtype, n, density * 100f, path, st.Median, st.Min, sp, md);
        }

        static void Section0Float(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 0. Operator matvec throughput (b={0}): dense GEMV vs sparse spMV, REPS={1} ---", BR, REPS_MATVEC));
            sb.AppendLine(MatvecHeader());

            foreach (var n in BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockSPDFloat(ref arena, nb, density, Seed(n, density, 91), out var dense, out var sparse);
                    uint sx = Seed(n, density, 92);

                    var xd = arena.floatRandomVec(n, -1f, 1f, sx);   // ping-pong clobbers input -> fresh copy per timing
                    var yd = arena.floatVec(n);
                    var denseJob = new MatvecDenseJobFloat { A = dense, x = xd, y = yd, reps = REPS_MATVEC };
                    var denseStat = Bench.Time(() => denseJob.Run());
                    sb.AppendLine(MatvecRow("float", n, density, "GEMV-dense", denseStat, 1.0, null));

                    var xs = arena.floatRandomVec(n, -1f, 1f, sx);   // identical contents to xd
                    var ys = arena.floatVec(n);
                    var sparseJob = new MatvecSparseJobFloat { A = sparse, x = xs, y = ys, reps = REPS_MATVEC };
                    var sparseStat = Bench.Time(() => sparseJob.Run());

                    // clean single-matvec numerical cross-check (untimed; identical input)
                    var xc = arena.floatRandomVec(n, -1f, 1f, sx);
                    var yDc = Blas.dot(dense, xc);
                    var ySc = BSR.spMV(sparse, xc);
                    double md = 0;
                    for (int i = 0; i < n; i++) md = math.max(md, math.abs((double)yDc[i] - (double)ySc[i]));
                    double speedup = denseStat.Median / math.max(sparseStat.Median, 1e-30);
                    sb.AppendLine(MatvecRow("float", n, density, "spMV-sparse", sparseStat, speedup, md));

                    arena.Dispose();
                }
            }
        }

        static void Section0Double(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 0. Operator matvec throughput (b={0}): dense GEMV vs sparse spMV, REPS={1} [double] ---", BR, REPS_MATVEC));
            sb.AppendLine(MatvecHeader());

            foreach (var n in BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockSPDDouble(ref arena, nb, density, Seed(n, density, 93), out var dense, out var sparse);
                    uint sx = Seed(n, density, 94);

                    var xd = arena.doubleRandomVec(n, -1.0, 1.0, sx);
                    var yd = arena.doubleVec(n);
                    var denseJob = new MatvecDenseJobDouble { A = dense, x = xd, y = yd, reps = REPS_MATVEC };
                    var denseStat = Bench.Time(() => denseJob.Run());
                    sb.AppendLine(MatvecRow("double", n, density, "GEMV-dense", denseStat, 1.0, null));

                    var xs = arena.doubleRandomVec(n, -1.0, 1.0, sx);
                    var ys = arena.doubleVec(n);
                    var sparseJob = new MatvecSparseJobDouble { A = sparse, x = xs, y = ys, reps = REPS_MATVEC };
                    var sparseStat = Bench.Time(() => sparseJob.Run());

                    var xc = arena.doubleRandomVec(n, -1.0, 1.0, sx);
                    var yDc = Blas.dot(dense, xc);
                    var ySc = BSR.spMV(sparse, xc);
                    double md = 0;
                    for (int i = 0; i < n; i++) md = math.max(md, math.abs(yDc[i] - ySc[i]));
                    double speedup = denseStat.Median / math.max(sparseStat.Median, 1e-30);
                    sb.AppendLine(MatvecRow("double", n, density, "spMV-sparse", sparseStat, speedup, md));

                    arena.Dispose();
                }
            }
        }

        // ==== Section 0b: symmetric-storage spMV vs full-storage spMV on the SAME SPD matrix ============
        //
        // Section 0 already compares dense GEMV vs sparse spMV; this isolates the OTHER half of the
        // Milestone-A story -- storing a genuinely symmetric matrix as upper-triangle-only
        // (Symmetric=true, built via ToBSRSymmetric) vs full block-CSR (every block, incl. the
        // explicit mirrored lower block), on the identical matrix (BuildBlockSPDPairFloat/Double pins
        // the rng sequence so `full` and `sym` encode byte-for-byte the same SPD system). bsrMatVecSym
        // does one accumulate per stored block for the diagonal and TWO (K*x_j and K^T*x_i) for each
        // off-diagonal, touching half as many STORED blocks as bsrMatVec's full traversal for the same
        // logical matrix -- expected speedup ~2x with denser off-diagonal fill (dense%=33) and less
        // pronounced at sparse fill (dense%=7, where per-block/per-row overhead dominates more).
        // maxAbsDiff cross-checks spMV(full) against spMV(sym) on a clean untimed matvec -- must be ~0.

        static void Section0bFloat(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 0b. Symmetric-storage spMV vs full-storage spMV, SAME SPD matrix (b={0}), REPS={1} [float] ---", BR, REPS_MATVEC));
            sb.AppendLine(MatvecHeader());

            foreach (var n in BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockSPDPairFloat(ref arena, nb, density, Seed(n, density, 95), out _, out var full, out var sym);
                    uint sx = Seed(n, density, 96);

                    var xf = arena.floatRandomVec(n, -1f, 1f, sx);
                    var yf = arena.floatVec(n);
                    var fullJob = new MatvecSparseJobFloat { A = full, x = xf, y = yf, reps = REPS_MATVEC };
                    var fullStat = Bench.Time(() => fullJob.Run());
                    sb.AppendLine(MatvecRow("float", n, density, "spMV-full", fullStat, 1.0, null));

                    var xs = arena.floatRandomVec(n, -1f, 1f, sx);   // identical contents to xf
                    var ys = arena.floatVec(n);
                    var symJob = new MatvecSparseJobFloat { A = sym, x = xs, y = ys, reps = REPS_MATVEC };
                    var symStat = Bench.Time(() => symJob.Run());

                    // clean single-matvec numerical cross-check (untimed; identical input)
                    var xc = arena.floatRandomVec(n, -1f, 1f, sx);
                    var yFc = BSR.spMV(full, xc);
                    var ySc = BSR.spMV(sym, xc);
                    double md = 0;
                    for (int i = 0; i < n; i++) md = math.max(md, math.abs((double)yFc[i] - (double)ySc[i]));
                    double speedup = fullStat.Median / math.max(symStat.Median, 1e-30);
                    sb.AppendLine(MatvecRow("float", n, density, "spMV-sym", symStat, speedup, md));

                    arena.Dispose();
                }
            }
        }

        static void Section0bDouble(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 0b. Symmetric-storage spMV vs full-storage spMV, SAME SPD matrix (b={0}), REPS={1} [double] ---", BR, REPS_MATVEC));
            sb.AppendLine(MatvecHeader());

            foreach (var n in BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockSPDPairDouble(ref arena, nb, density, Seed(n, density, 97), out _, out var full, out var sym);
                    uint sx = Seed(n, density, 98);

                    var xf = arena.doubleRandomVec(n, -1.0, 1.0, sx);
                    var yf = arena.doubleVec(n);
                    var fullJob = new MatvecSparseJobDouble { A = full, x = xf, y = yf, reps = REPS_MATVEC };
                    var fullStat = Bench.Time(() => fullJob.Run());
                    sb.AppendLine(MatvecRow("double", n, density, "spMV-full", fullStat, 1.0, null));

                    var xs = arena.doubleRandomVec(n, -1.0, 1.0, sx);
                    var ys = arena.doubleVec(n);
                    var symJob = new MatvecSparseJobDouble { A = sym, x = xs, y = ys, reps = REPS_MATVEC };
                    var symStat = Bench.Time(() => symJob.Run());

                    var xc = arena.doubleRandomVec(n, -1.0, 1.0, sx);
                    var yFc = BSR.spMV(full, xc);
                    var ySc = BSR.spMV(sym, xc);
                    double md = 0;
                    for (int i = 0; i < n; i++) md = math.max(md, math.abs(yFc[i] - ySc[i]));
                    double speedup = fullStat.Median / math.max(symStat.Median, 1e-30);
                    sb.AppendLine(MatvecRow("double", n, density, "spMV-sym", symStat, speedup, md));

                    arena.Dispose();
                }
            }
        }

        // ==== Section 1: SPD -> cg & minres ==============================================

        static void Section1Float(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 1. SPD block-sparse (b={0}): cg & minres, K={1}, tol=0 [float] ---", BR, K_CG));
            sb.AppendLine(RowHeader());

            foreach (var n in BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockSPDFloat(ref arena, nb, density, Seed(n, density, 11), out var dense, out var sparse);
                    var b = arena.floatRandomVec(n, -1f, 1f, Seed(n, density, 12));

                    var xCgD = arena.floatVec(n); var rCgD = arena.floatVec(n); var pCgD = arena.floatVec(n); var ApCgD = arena.floatVec(n);
                    var cgDenseJob = new CGDenseJobFloat { A = dense, b = b, x = xCgD, r = rCgD, p = pCgD, Ap = ApCgD, K = K_CG };
                    var cgDenseStat = Bench.Time(() => cgDenseJob.Run());
                    sb.AppendLine(Row("float", n, density, "CG-dense", cgDenseStat, ResidualLinSys(in dense, in xCgD, in b)));

                    var xCgS = arena.floatVec(n); var rCgS = arena.floatVec(n); var pCgS = arena.floatVec(n); var ApCgS = arena.floatVec(n);
                    var cgSparseJob = new CGSparseJobFloat { A = sparse, b = b, x = xCgS, r = rCgS, p = pCgS, Ap = ApCgS, K = K_CG };
                    var cgSparseStat = Bench.Time(() => cgSparseJob.Run());
                    sb.AppendLine(Row("float", n, density, "CG-sparse", cgSparseStat, ResidualLinSys(in dense, in xCgS, in b)));

                    var xMrD = arena.floatVec(n);
                    var yD = arena.floatVec(n); var r1D = arena.floatVec(n); var r2D = arena.floatVec(n); var vD = arena.floatVec(n);
                    var wD = arena.floatVec(n); var w1D = arena.floatVec(n); var w2D = arena.floatVec(n);
                    var mrDenseJob = new MinresDenseJobFloat { A = dense, b = b, x = xMrD, y = yD, r1 = r1D, r2 = r2D, v = vD, w = wD, w1 = w1D, w2 = w2D, K = K_CG };
                    var mrDenseStat = Bench.Time(() => mrDenseJob.Run());
                    sb.AppendLine(Row("float", n, density, "MINRES-dense", mrDenseStat, ResidualLinSys(in dense, in xMrD, in b)));

                    var xMrS = arena.floatVec(n);
                    var yS = arena.floatVec(n); var r1S = arena.floatVec(n); var r2S = arena.floatVec(n); var vS = arena.floatVec(n);
                    var wS = arena.floatVec(n); var w1S = arena.floatVec(n); var w2S = arena.floatVec(n);
                    var mrSparseJob = new MinresSparseJobFloat { A = sparse, b = b, x = xMrS, y = yS, r1 = r1S, r2 = r2S, v = vS, w = wS, w1 = w1S, w2 = w2S, K = K_CG };
                    var mrSparseStat = Bench.Time(() => mrSparseJob.Run());
                    sb.AppendLine(Row("float", n, density, "MINRES-sparse", mrSparseStat, ResidualLinSys(in dense, in xMrS, in b)));

                    arena.Dispose();
                }
            }
        }

        static void Section1Double(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 1. SPD block-sparse (b={0}): cg & minres, K={1}, tol=0 [double] ---", BR, K_CG));
            sb.AppendLine(RowHeader());

            foreach (var n in BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockSPDDouble(ref arena, nb, density, Seed(n, density, 13), out var dense, out var sparse);
                    var b = arena.doubleRandomVec(n, -1.0, 1.0, Seed(n, density, 14));

                    var xCgD = arena.doubleVec(n); var rCgD = arena.doubleVec(n); var pCgD = arena.doubleVec(n); var ApCgD = arena.doubleVec(n);
                    var cgDenseJob = new CGDenseJobDouble { A = dense, b = b, x = xCgD, r = rCgD, p = pCgD, Ap = ApCgD, K = K_CG };
                    var cgDenseStat = Bench.Time(() => cgDenseJob.Run());
                    sb.AppendLine(Row("double", n, density, "CG-dense", cgDenseStat, ResidualLinSys(in dense, in xCgD, in b)));

                    var xCgS = arena.doubleVec(n); var rCgS = arena.doubleVec(n); var pCgS = arena.doubleVec(n); var ApCgS = arena.doubleVec(n);
                    var cgSparseJob = new CGSparseJobDouble { A = sparse, b = b, x = xCgS, r = rCgS, p = pCgS, Ap = ApCgS, K = K_CG };
                    var cgSparseStat = Bench.Time(() => cgSparseJob.Run());
                    sb.AppendLine(Row("double", n, density, "CG-sparse", cgSparseStat, ResidualLinSys(in dense, in xCgS, in b)));

                    var xMrD = arena.doubleVec(n);
                    var yD = arena.doubleVec(n); var r1D = arena.doubleVec(n); var r2D = arena.doubleVec(n); var vD = arena.doubleVec(n);
                    var wD = arena.doubleVec(n); var w1D = arena.doubleVec(n); var w2D = arena.doubleVec(n);
                    var mrDenseJob = new MinresDenseJobDouble { A = dense, b = b, x = xMrD, y = yD, r1 = r1D, r2 = r2D, v = vD, w = wD, w1 = w1D, w2 = w2D, K = K_CG };
                    var mrDenseStat = Bench.Time(() => mrDenseJob.Run());
                    sb.AppendLine(Row("double", n, density, "MINRES-dense", mrDenseStat, ResidualLinSys(in dense, in xMrD, in b)));

                    var xMrS = arena.doubleVec(n);
                    var yS = arena.doubleVec(n); var r1S = arena.doubleVec(n); var r2S = arena.doubleVec(n); var vS = arena.doubleVec(n);
                    var wS = arena.doubleVec(n); var w1S = arena.doubleVec(n); var w2S = arena.doubleVec(n);
                    var mrSparseJob = new MinresSparseJobDouble { A = sparse, b = b, x = xMrS, y = yS, r1 = r1S, r2 = r2S, v = vS, w = wS, w1 = w1S, w2 = w2S, K = K_CG };
                    var mrSparseStat = Bench.Time(() => mrSparseJob.Run());
                    sb.AppendLine(Row("double", n, density, "MINRES-sparse", mrSparseStat, ResidualLinSys(in dense, in xMrS, in b)));

                    arena.Dispose();
                }
            }
        }

        // ==== Section 1x: SPD block-sparse, b=4, N=1024 (CG only, 7% fill) ==============================
        //
        // The b=3 sweep above (Section 1) tops out at N=768 because 1024 isn't divisible by 3. b=4 IS
        // one of the compile-time-unrolled bsrMatVecB4 kernel sizes (see SparseOP.fProxy.cs's spMV
        // dispatch switch), so nb=256 blocks of 4x4 gives a genuine 1024x1024 CG dense-vs-sparse
        // comparison at the same convention as Section 1 (7% block density, K=40 iterations, tol=0).
        // Only CG is timed here (not MINRES) -- this subsection exists specifically to backfill the
        // README's 1024x1024 CG row, not to duplicate the full Section-1 solver sweep.

        const int N_B4 = 1024, BR4 = 4, NB_B4 = N_B4 / BR4; // 256 blocks of 4x4

        static void Section1xFloat(StringBuilder sb)
        {
            float density = Densities[0]; // 7%
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 1x. SPD block-sparse (b={0}, N={1}): cg, K={2}, tol=0 [float] ---", BR4, N_B4, K_CG));
            sb.AppendLine(RowHeader());

            var arena = new Arena(Allocator.Persistent);
            BuildBlockSPDFloatSized(ref arena, NB_B4, BR4, density, Seed(N_B4, density, 111), out var dense, out var sparse);
            var b = arena.floatRandomVec(N_B4, -1f, 1f, Seed(N_B4, density, 112));

            var xCgD = arena.floatVec(N_B4); var rCgD = arena.floatVec(N_B4); var pCgD = arena.floatVec(N_B4); var ApCgD = arena.floatVec(N_B4);
            var cgDenseJob = new CGDenseJobFloat { A = dense, b = b, x = xCgD, r = rCgD, p = pCgD, Ap = ApCgD, K = K_CG };
            var cgDenseStat = Bench.Time(() => cgDenseJob.Run());
            sb.AppendLine(Row("float", N_B4, density, "CG-dense", cgDenseStat, ResidualLinSys(in dense, in xCgD, in b)));

            var xCgS = arena.floatVec(N_B4); var rCgS = arena.floatVec(N_B4); var pCgS = arena.floatVec(N_B4); var ApCgS = arena.floatVec(N_B4);
            var cgSparseJob = new CGSparseJobFloat { A = sparse, b = b, x = xCgS, r = rCgS, p = pCgS, Ap = ApCgS, K = K_CG };
            var cgSparseStat = Bench.Time(() => cgSparseJob.Run());
            sb.AppendLine(Row("float", N_B4, density, "CG-sparse", cgSparseStat, ResidualLinSys(in dense, in xCgS, in b)));

            arena.Dispose();
        }

        static void Section1xDouble(StringBuilder sb)
        {
            float density = Densities[0]; // 7%
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 1x. SPD block-sparse (b={0}, N={1}): cg, K={2}, tol=0 [double] ---", BR4, N_B4, K_CG));
            sb.AppendLine(RowHeader());

            var arena = new Arena(Allocator.Persistent);
            BuildBlockSPDDoubleSized(ref arena, NB_B4, BR4, density, Seed(N_B4, density, 113), out var dense, out var sparse);
            var b = arena.doubleRandomVec(N_B4, -1.0, 1.0, Seed(N_B4, density, 114));

            var xCgD = arena.doubleVec(N_B4); var rCgD = arena.doubleVec(N_B4); var pCgD = arena.doubleVec(N_B4); var ApCgD = arena.doubleVec(N_B4);
            var cgDenseJob = new CGDenseJobDouble { A = dense, b = b, x = xCgD, r = rCgD, p = pCgD, Ap = ApCgD, K = K_CG };
            var cgDenseStat = Bench.Time(() => cgDenseJob.Run());
            sb.AppendLine(Row("double", N_B4, density, "CG-dense", cgDenseStat, ResidualLinSys(in dense, in xCgD, in b)));

            var xCgS = arena.doubleVec(N_B4); var rCgS = arena.doubleVec(N_B4); var pCgS = arena.doubleVec(N_B4); var ApCgS = arena.doubleVec(N_B4);
            var cgSparseJob = new CGSparseJobDouble { A = sparse, b = b, x = xCgS, r = rCgS, p = pCgS, Ap = ApCgS, K = K_CG };
            var cgSparseStat = Bench.Time(() => cgSparseJob.Run());
            sb.AppendLine(Row("double", N_B4, density, "CG-sparse", cgSparseStat, ResidualLinSys(in dense, in xCgS, in b)));

            arena.Dispose();
        }

        // ==== Section 2: non-symmetric -> biCGStab =======================================================

        static void Section2Float(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 2. Non-symmetric block-sparse (b={0}): biCGStab, K={1}, tol=0 [float] ---", BR, K_BICGSTAB));
            sb.AppendLine(RowHeader());

            foreach (var n in BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockNonSymFloat(ref arena, nb, density, Seed(n, density, 21), out var dense, out var sparse);
                    var b = arena.floatRandomVec(n, -1f, 1f, Seed(n, density, 22));

                    var xD = arena.floatVec(n); var rD = arena.floatVec(n); var rh0D = arena.floatVec(n);
                    var pD = arena.floatVec(n); var vD = arena.floatVec(n); var tD = arena.floatVec(n);
                    var jobD = new BiCGStabDenseJobFloat { A = dense, b = b, x = xD, r = rD, rHat0 = rh0D, p = pD, v = vD, t = tD, K = K_BICGSTAB };
                    var statD = Bench.Time(() => jobD.Run());
                    sb.AppendLine(Row("float", n, density, "BiCGStab-dense", statD, ResidualLinSys(in dense, in xD, in b)));

                    var xS = arena.floatVec(n); var rS = arena.floatVec(n); var rh0S = arena.floatVec(n);
                    var pS = arena.floatVec(n); var vS = arena.floatVec(n); var tS = arena.floatVec(n);
                    var jobS = new BiCGStabSparseJobFloat { A = sparse, b = b, x = xS, r = rS, rHat0 = rh0S, p = pS, v = vS, t = tS, K = K_BICGSTAB };
                    var statS = Bench.Time(() => jobS.Run());
                    sb.AppendLine(Row("float", n, density, "BiCGStab-sparse", statS, ResidualLinSys(in dense, in xS, in b)));

                    arena.Dispose();
                }
            }
        }

        static void Section2Double(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 2. Non-symmetric block-sparse (b={0}): biCGStab, K={1}, tol=0 [double] ---", BR, K_BICGSTAB));
            sb.AppendLine(RowHeader());

            foreach (var n in BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockNonSymDouble(ref arena, nb, density, Seed(n, density, 23), out var dense, out var sparse);
                    var b = arena.doubleRandomVec(n, -1.0, 1.0, Seed(n, density, 24));

                    var xD = arena.doubleVec(n); var rD = arena.doubleVec(n); var rh0D = arena.doubleVec(n);
                    var pD = arena.doubleVec(n); var vD = arena.doubleVec(n); var tD = arena.doubleVec(n);
                    var jobD = new BiCGStabDenseJobDouble { A = dense, b = b, x = xD, r = rD, rHat0 = rh0D, p = pD, v = vD, t = tD, K = K_BICGSTAB };
                    var statD = Bench.Time(() => jobD.Run());
                    sb.AppendLine(Row("double", n, density, "BiCGStab-dense", statD, ResidualLinSys(in dense, in xD, in b)));

                    var xS = arena.doubleVec(n); var rS = arena.doubleVec(n); var rh0S = arena.doubleVec(n);
                    var pS = arena.doubleVec(n); var vS = arena.doubleVec(n); var tS = arena.doubleVec(n);
                    var jobS = new BiCGStabSparseJobDouble { A = sparse, b = b, x = xS, r = rS, rHat0 = rh0S, p = pS, v = vS, t = tS, K = K_BICGSTAB };
                    var statS = Bench.Time(() => jobS.Run());
                    sb.AppendLine(Row("double", n, density, "BiCGStab-sparse", statS, ResidualLinSys(in dense, in xS, in b)));

                    arena.Dispose();
                }
            }
        }

        // ==== Section 3: rectangular -> cgls & lsqr (over- and under-determined) ========================

        static void RunRectCaseFloat(int nRef, int mb, int nb, float density, string tag, int tagSeed, StringBuilder sb)
        {
            var arena = new Arena(Allocator.Persistent);
            BuildBlockRectFloat(ref arena, mb, nb, density, Seed(nRef, density, tagSeed), out var dense, out var sparse);
            int rows = mb * BR, cols = nb * BR;
            var b = arena.floatRandomVec(rows, -1f, 1f, Seed(nRef, density, tagSeed + 1));

            var xCD = arena.floatVec(cols); var rCD = arena.floatVec(rows); var sCD = arena.floatVec(cols); var pCD = arena.floatVec(cols); var qCD = arena.floatVec(rows);
            var cglsDenseJob = new CglsDenseJobFloat { A = dense, b = b, x = xCD, r = rCD, s = sCD, p = pCD, q = qCD, K = K_LS };
            var cglsDenseStat = Bench.Time(() => cglsDenseJob.Run());
            sb.AppendLine(Row("float", nRef, density, "CGLS-dense-" + tag, cglsDenseStat, ResidualLS(in dense, in xCD, in b)));

            var xCS = arena.floatVec(cols); var rCS = arena.floatVec(rows); var sCS = arena.floatVec(cols); var pCS = arena.floatVec(cols); var qCS = arena.floatVec(rows);
            var cglsSparseJob = new CglsSparseJobFloat { A = sparse, b = b, x = xCS, r = rCS, s = sCS, p = pCS, q = qCS, K = K_LS };
            var cglsSparseStat = Bench.Time(() => cglsSparseJob.Run());
            sb.AppendLine(Row("float", nRef, density, "CGLS-sparse-" + tag, cglsSparseStat, ResidualLS(in dense, in xCS, in b)));

            var xLD = arena.floatVec(cols); var uLD = arena.floatVec(rows); var vLD = arena.floatVec(cols); var wLD = arena.floatVec(cols);
            var tmMLD = arena.floatVec(rows); var tmNLD = arena.floatVec(cols);
            var lsqrDenseJob = new LsqrDenseJobFloat { A = dense, b = b, x = xLD, u = uLD, v = vLD, w = wLD, tmpM = tmMLD, tmpN = tmNLD, K = K_LS };
            var lsqrDenseStat = Bench.Time(() => lsqrDenseJob.Run());
            sb.AppendLine(Row("float", nRef, density, "LSQR-dense-" + tag, lsqrDenseStat, ResidualLS(in dense, in xLD, in b)));

            var xLS = arena.floatVec(cols); var uLS = arena.floatVec(rows); var vLS = arena.floatVec(cols); var wLS = arena.floatVec(cols);
            var tmMLS = arena.floatVec(rows); var tmNLS = arena.floatVec(cols);
            var lsqrSparseJob = new LsqrSparseJobFloat { A = sparse, b = b, x = xLS, u = uLS, v = vLS, w = wLS, tmpM = tmMLS, tmpN = tmNLS, K = K_LS };
            var lsqrSparseStat = Bench.Time(() => lsqrSparseJob.Run());
            sb.AppendLine(Row("float", nRef, density, "LSQR-sparse-" + tag, lsqrSparseStat, ResidualLS(in dense, in xLS, in b)));

            // Milestone B: transpose-optimized variants -- Aᵀ materialized ONCE (outside timing), ApplyT
            // becomes a forward spMV over Aᵀ. Compare "sparseT" rows against the "sparse" rows above.
            var AT = arena.floatBSRTranspose(in sparse);

            var xCST = arena.floatVec(cols); var rCST = arena.floatVec(rows); var sCST = arena.floatVec(cols); var pCST = arena.floatVec(cols); var qCST = arena.floatVec(rows);
            var cglsSparseTJob = new CglsSparseTJobFloat { A = sparse, AT = AT, b = b, x = xCST, r = rCST, s = sCST, p = pCST, q = qCST, K = K_LS };
            var cglsSparseTStat = Bench.Time(() => cglsSparseTJob.Run());
            sb.AppendLine(Row("float", nRef, density, "CGLS-sparseT-" + tag, cglsSparseTStat, ResidualLS(in dense, in xCST, in b)));

            var xLST = arena.floatVec(cols); var uLST = arena.floatVec(rows); var vLST = arena.floatVec(cols); var wLST = arena.floatVec(cols);
            var tmMLST = arena.floatVec(rows); var tmNLST = arena.floatVec(cols);
            var lsqrSparseTJob = new LsqrSparseTJobFloat { A = sparse, AT = AT, b = b, x = xLST, u = uLST, v = vLST, w = wLST, tmpM = tmMLST, tmpN = tmNLST, K = K_LS };
            var lsqrSparseTStat = Bench.Time(() => lsqrSparseTJob.Run());
            sb.AppendLine(Row("float", nRef, density, "LSQR-sparseT-" + tag, lsqrSparseTStat, ResidualLS(in dense, in xLST, in b)));

            arena.Dispose();
        }

        static void RunRectCaseDouble(int nRef, int mb, int nb, float density, string tag, int tagSeed, StringBuilder sb)
        {
            var arena = new Arena(Allocator.Persistent);
            BuildBlockRectDouble(ref arena, mb, nb, density, Seed(nRef, density, tagSeed), out var dense, out var sparse);
            int rows = mb * BR, cols = nb * BR;
            var b = arena.doubleRandomVec(rows, -1.0, 1.0, Seed(nRef, density, tagSeed + 1));

            var xCD = arena.doubleVec(cols); var rCD = arena.doubleVec(rows); var sCD = arena.doubleVec(cols); var pCD = arena.doubleVec(cols); var qCD = arena.doubleVec(rows);
            var cglsDenseJob = new CglsDenseJobDouble { A = dense, b = b, x = xCD, r = rCD, s = sCD, p = pCD, q = qCD, K = K_LS };
            var cglsDenseStat = Bench.Time(() => cglsDenseJob.Run());
            sb.AppendLine(Row("double", nRef, density, "CGLS-dense-" + tag, cglsDenseStat, ResidualLS(in dense, in xCD, in b)));

            var xCS = arena.doubleVec(cols); var rCS = arena.doubleVec(rows); var sCS = arena.doubleVec(cols); var pCS = arena.doubleVec(cols); var qCS = arena.doubleVec(rows);
            var cglsSparseJob = new CglsSparseJobDouble { A = sparse, b = b, x = xCS, r = rCS, s = sCS, p = pCS, q = qCS, K = K_LS };
            var cglsSparseStat = Bench.Time(() => cglsSparseJob.Run());
            sb.AppendLine(Row("double", nRef, density, "CGLS-sparse-" + tag, cglsSparseStat, ResidualLS(in dense, in xCS, in b)));

            var xLD = arena.doubleVec(cols); var uLD = arena.doubleVec(rows); var vLD = arena.doubleVec(cols); var wLD = arena.doubleVec(cols);
            var tmMLD = arena.doubleVec(rows); var tmNLD = arena.doubleVec(cols);
            var lsqrDenseJob = new LsqrDenseJobDouble { A = dense, b = b, x = xLD, u = uLD, v = vLD, w = wLD, tmpM = tmMLD, tmpN = tmNLD, K = K_LS };
            var lsqrDenseStat = Bench.Time(() => lsqrDenseJob.Run());
            sb.AppendLine(Row("double", nRef, density, "LSQR-dense-" + tag, lsqrDenseStat, ResidualLS(in dense, in xLD, in b)));

            var xLS = arena.doubleVec(cols); var uLS = arena.doubleVec(rows); var vLS = arena.doubleVec(cols); var wLS = arena.doubleVec(cols);
            var tmMLS = arena.doubleVec(rows); var tmNLS = arena.doubleVec(cols);
            var lsqrSparseJob = new LsqrSparseJobDouble { A = sparse, b = b, x = xLS, u = uLS, v = vLS, w = wLS, tmpM = tmMLS, tmpN = tmNLS, K = K_LS };
            var lsqrSparseStat = Bench.Time(() => lsqrSparseJob.Run());
            sb.AppendLine(Row("double", nRef, density, "LSQR-sparse-" + tag, lsqrSparseStat, ResidualLS(in dense, in xLS, in b)));

            // Milestone B: transpose-optimized variants -- Aᵀ materialized ONCE (outside timing), ApplyT
            // becomes a forward spMV over Aᵀ. Compare "sparseT" rows against the "sparse" rows above.
            var AT = arena.doubleBSRTranspose(in sparse);

            var xCST = arena.doubleVec(cols); var rCST = arena.doubleVec(rows); var sCST = arena.doubleVec(cols); var pCST = arena.doubleVec(cols); var qCST = arena.doubleVec(rows);
            var cglsSparseTJob = new CglsSparseTJobDouble { A = sparse, AT = AT, b = b, x = xCST, r = rCST, s = sCST, p = pCST, q = qCST, K = K_LS };
            var cglsSparseTStat = Bench.Time(() => cglsSparseTJob.Run());
            sb.AppendLine(Row("double", nRef, density, "CGLS-sparseT-" + tag, cglsSparseTStat, ResidualLS(in dense, in xCST, in b)));

            var xLST = arena.doubleVec(cols); var uLST = arena.doubleVec(rows); var vLST = arena.doubleVec(cols); var wLST = arena.doubleVec(cols);
            var tmMLST = arena.doubleVec(rows); var tmNLST = arena.doubleVec(cols);
            var lsqrSparseTJob = new LsqrSparseTJobDouble { A = sparse, AT = AT, b = b, x = xLST, u = uLST, v = vLST, w = wLST, tmpM = tmMLST, tmpN = tmNLST, K = K_LS };
            var lsqrSparseTStat = Bench.Time(() => lsqrSparseTJob.Run());
            sb.AppendLine(Row("double", nRef, density, "LSQR-sparseT-" + tag, lsqrSparseTStat, ResidualLS(in dense, in xLST, in b)));

            arena.Dispose();
        }

        static void Section3Float(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 3. Rectangular block-sparse (b={0}): cgls & lsqr, K={1}, tol=0 [float] ---", BR, K_LS));
            sb.AppendLine("    over: rows=2xcols block grid (m=2n, overdetermined); under: cols=2xrows block grid (m=n/2, underdetermined).");
            sb.AppendLine("    residual = ||A^T(Ax-b)|| / ||A^T b|| (least-squares optimality, not ||Ax-b||).");
            sb.AppendLine(RowHeader());

            foreach (var n in BlockSizesN)
            {
                int nb0 = n / BR;
                foreach (var density in Densities)
                {
                    RunRectCaseFloat(n, 2 * nb0, nb0, density, "over", 31, sb);
                    RunRectCaseFloat(n, nb0, 2 * nb0, density, "under", 33, sb);
                }
            }
        }

        static void Section3Double(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 3. Rectangular block-sparse (b={0}): cgls & lsqr, K={1}, tol=0 [double] ---", BR, K_LS));
            sb.AppendLine("    over: rows=2xcols block grid (m=2n, overdetermined); under: cols=2xrows block grid (m=n/2, underdetermined).");
            sb.AppendLine("    residual = ||A^T(Ax-b)|| / ||A^T b|| (least-squares optimality, not ||Ax-b||).");
            sb.AppendLine(RowHeader());

            foreach (var n in BlockSizesN)
            {
                int nb0 = n / BR;
                foreach (var density in Densities)
                {
                    RunRectCaseDouble(n, 2 * nb0, nb0, density, "over", 35, sb);
                    RunRectCaseDouble(n, nb0, 2 * nb0, density, "under", 37, sb);
                }
            }
        }

        // ==== Section 4: zero-cost-abstraction probe (THE fork datapoint) ===============================
        //
        // Generic Solvers.cg(in floatMxN/doubleMxN,...) -- which internally wraps A in
        // floatDenseOperator/doubleDenseOperator and calls the generic cg<TOp> loop -- vs a hand-inlined
        // dense CG written directly against raw pointers in CGHandInlinedJobFloat/Double (see above), no
        // IfloatLinearOperator/IdoubleLinearOperator, no generic dispatch. Same matrix, same K, same
        // algorithm. If Burst fully monomorphizes/inlines the generic operator call, the two times should
        // be ~equal (ratio ~1) -- the operator abstraction is then free, and there is no perf case for
        // forking dense-specific vs sparse-specific solver bodies. A material gap would argue otherwise.

        static void Section4Float(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 4. Zero-cost-abstraction probe (dense SPD, K={0}, tol=0) [float] ---", K_CG));
            sb.AppendLine("    generic = Solvers.cg(in floatMxN,...) via cg<floatDenseOperator>;");
            sb.AppendLine("    hand-inlined = raw-pointer GEMV/axpy/dot written directly in the job, no operator interface.");
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11} {4,11} {5,10}",
                "dtype", "N", "path", "med(ms)", "min(ms)", "ratio"));

            foreach (var n in BlockSizesN)
            {
                var arena = new Arena(Allocator.Persistent);
                var M = arena.floatRandomMat(n, n, -1f, 1f, Seed(n, 0f, 41));
                var A = Blas.dot(M, M, true);
                for (int d = 0; d < n; d++) A[d, d] += n;
                var b = arena.floatRandomVec(n, -1f, 1f, Seed(n, 0f, 42));

                var xG = arena.floatVec(n); var rG = arena.floatVec(n); var pG = arena.floatVec(n); var ApG = arena.floatVec(n);
                var genericJob = new CGDenseJobFloat { A = A, b = b, x = xG, r = rG, p = pG, Ap = ApG, K = K_CG };
                var genericStat = Bench.Time(() => genericJob.Run());

                var xH = arena.floatVec(n); var rH = arena.floatVec(n); var pH = arena.floatVec(n); var ApH = arena.floatVec(n);
                var handJob = new CGHandInlinedJobFloat { A = A, b = b, x = xH, r = rH, p = pH, Ap = ApH, K = K_CG };
                var handStat = Bench.Time(() => handJob.Run());

                double ratio = genericStat.Median / math.max(handStat.Median, 1e-9);
                sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11:F4} {4,11:F4} {5,10:F3}",
                    "float", n, "generic", genericStat.Median, genericStat.Min, ratio));
                sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11:F4} {4,11:F4} {5,10}",
                    "float", n, "hand-inlined", handStat.Median, handStat.Min, "--"));

                arena.Dispose();
            }
        }

        static void Section4Double(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 4. Zero-cost-abstraction probe (dense SPD, K={0}, tol=0) [double] ---", K_CG));
            sb.AppendLine("    generic = Solvers.cg(in doubleMxN,...) via cg<doubleDenseOperator>;");
            sb.AppendLine("    hand-inlined = raw-pointer GEMV/axpy/dot written directly in the job, no operator interface.");
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11} {4,11} {5,10}",
                "dtype", "N", "path", "med(ms)", "min(ms)", "ratio"));

            foreach (var n in BlockSizesN)
            {
                var arena = new Arena(Allocator.Persistent);
                var M = arena.doubleRandomMat(n, n, -1.0, 1.0, Seed(n, 0f, 43));
                var A = Blas.dot(M, M, true);
                for (int d = 0; d < n; d++) A[d, d] += n;
                var b = arena.doubleRandomVec(n, -1.0, 1.0, Seed(n, 0f, 44));

                var xG = arena.doubleVec(n); var rG = arena.doubleVec(n); var pG = arena.doubleVec(n); var ApG = arena.doubleVec(n);
                var genericJob = new CGDenseJobDouble { A = A, b = b, x = xG, r = rG, p = pG, Ap = ApG, K = K_CG };
                var genericStat = Bench.Time(() => genericJob.Run());

                var xH = arena.doubleVec(n); var rH = arena.doubleVec(n); var pH = arena.doubleVec(n); var ApH = arena.doubleVec(n);
                var handJob = new CGHandInlinedJobDouble { A = A, b = b, x = xH, r = rH, p = pH, Ap = ApH, K = K_CG };
                var handStat = Bench.Time(() => handJob.Run());

                double ratio = genericStat.Median / math.max(handStat.Median, 1e-9);
                sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11:F4} {4,11:F4} {5,10:F3}",
                    "double", n, "generic", genericStat.Median, genericStat.Min, ratio));
                sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11:F4} {4,11:F4} {5,10}",
                    "double", n, "hand-inlined", handStat.Median, handStat.Min, "--"));

                arena.Dispose();
            }
        }
    }
}
