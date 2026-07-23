using System;

using LinearAlgebra;
using LinearAlgebra.Sparse;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Krylov compatibility grid -- (BSR gallery matrix) x (preconditioner) x (single-RHS square Krylov
// solver). Every compatible cell is classified Skipped / Converged / MaxIterations / Errored /
// FalseConverged through the same IfProxySquareSolverInvoker struct-functor the square battery uses;
// incompatible cells are Skipped (no matrix built, no solve run). Two pinned invariants:
//   PRIMARY  -- across the whole grid, no cell is Errored and no cell is FalseConverged (the honest
//               anti-silent-divergence net: a Converged status must survive a fresh residual check,
//               and no solve may return NaN/Inf or a Breakdown/Degenerate status).
//   SECONDARY -- on the two SPD galleries, cg + {IC0, FSAI, AMG, BlockJacobi} all reach Converged
//               (these pass in the preconditioner battery, so a regression here is real).
// MaxIterations is an ALLOWED outcome (a solver may legitimately run out of budget on a hard cell).
// One NUnit case per gallery, so a failure names the gallery; the first offending cell is surfaced
// via Fail. New solver/preconditioner columns extend the GridSolver/GridPrecond enums and their
// dispatch arms, nothing else in this file changes.
public class fProxyKrylovGridTests
{
    // The 10 single-RHS square Krylov solvers (one IfProxySquareSolverInvoker each), same order as
    // KrylovBattery.Invokers.fProxy.cs.
    public enum GridSolver { Cg, Fcg, Minres, MinresQLP, BiCGStab, Gmres, Fgmres, Idr, Tfqmr, Gcrodr }

    // The 11 preconditioner columns incl. Identity. Symmetric-M columns (SPD galleries only):
    // BlockJacobi, SSOR, IC0, Chebyshev, FSAI, AdditiveSchwarz, AMG. Nonsym-M columns (any square
    // gallery): ILU0, SPAI, RestrictedSchwarz. Identity: any gallery (plain unpreconditioned Solve).
    public enum GridPrecond { Identity, BlockJacobi, SSOR, IC0, Chebyshev, FSAI, AdditiveSchwarz, AMG, ILU0, SPAI, RestrictedSchwarz }

    // Per-cell classification. Skipped cells are never built or solved.
    public enum CellOutcome { Skipped, Converged, MaxIterations, Errored, FalseConverged }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public GalleryBSRMatrix Gallery;

        // [0] flag (1 = failure recorded) [1] which (0 = honesty-net PRIMARY, 1 = known-good
        // SECONDARY) [2] solver-as-int [3] precond-as-int [4] outcome-as-int [5] fresh residual.
        public NativeArray<fProxy> Fail;

        static fProxy Tol() => Consts.fProxySqrtEps;

        public void Execute()
        {
            MatrixProfile tags = GalleryProfiles.Of(Gallery);

            RunSolver(new fProxyCgInvoker { TolValue = Tol(), MaxIterMul = 8 }, GridSolver.Cg, tags);
            RunSolver(new fProxyFcgInvoker { TolValue = Tol(), MaxIterMul = 8 }, GridSolver.Fcg, tags);
            RunSolver(new fProxyMinresInvoker { TolValue = Tol(), MaxIterMul = 8 }, GridSolver.Minres, tags);
            RunSolver(new fProxyMinresQLPInvoker { TolValue = Tol(), MaxIterMul = 8 }, GridSolver.MinresQLP, tags);
            RunSolver(new fProxyBiCGStabInvoker { TolValue = Tol(), MaxIterMul = 8 }, GridSolver.BiCGStab, tags);
            RunSolver(new fProxyGmresInvoker { TolValue = Tol(), MaxIterMul = 8, Restart = 30 }, GridSolver.Gmres, tags);
            RunSolver(new fProxyFgmresInvoker { TolValue = Tol(), MaxIterMul = 8, Restart = 30 }, GridSolver.Fgmres, tags);
            RunSolver(new fProxyIdrInvoker { TolValue = Tol(), MaxIterMul = 8, S = 4, Seed = 0x9E3779B1u }, GridSolver.Idr, tags);
            RunSolver(new fProxyTfqmrInvoker { TolValue = Tol(), MaxIterMul = 16 }, GridSolver.Tfqmr, tags);
            RunSolver(new fProxyGcrodrInvoker { TolValue = Tol(), MaxIterMul = 8, Restart = 30, Recycle = 10 }, GridSolver.Gcrodr, tags);
        }

        // Runs one solver across every applicable preconditioner column on this gallery. The solver
        // is Skipped wholesale on a gallery it does not accept (rule 1); each surviving precond column
        // builds a fresh Allocator.Temp matrix/preconditioner per cell.
        void RunSolver<TInv>(TInv inv, GridSolver solver, MatrixProfile tags) where TInv : struct, IfProxySquareSolverInvoker
        {
            if (!MatrixProfileMatch.Applicable(inv.Requires, inv.Forbids, tags)) return;   // whole solver Skipped

            for (int pi = 0; pi <= (int)GridPrecond.RestrictedSchwarz; pi++)
            {
                var precond = (GridPrecond)pi;
                if (!PrecondApplicable(inv.PrecondKind, precond, tags)) continue;           // cell Skipped

                var A = fProxyKrylovBatteryGallery.Build(Gallery);
                int n = A.M_Rows;
                var op = new fProxyBSROperator(in A);
                uint seed = 0xC000u + (uint)Gallery * 1000u + (uint)solver * 100u + (uint)precond;
                var b = GenerateOP.fProxyRandomVec(n, (fProxy)(-1), (fProxy)1, seed);
                inv.Init(n);
                var x = new fProxyN(n, Allocator.Temp);
                for (int i = 0; i < n; i++) x[i] = (fProxy)0;

                CellOutcome outcome = RunCell(inv, in A, in op, in b, ref x, precond, out fProxy resid);

                bool primaryOk = outcome != CellOutcome.Errored && outcome != CellOutcome.FalseConverged;
                Record(primaryOk, (fProxy)0, (int)solver, (int)precond, outcome, resid);

                if (IsKnownGoodCombo(solver, precond, tags))
                    Record(outcome == CellOutcome.Converged, (fProxy)1, (int)solver, (int)precond, outcome, resid);
            }
        }

        // Rule 2 (solver vs precond symmetry) + rule 3 (precond vs gallery). Rule 1 (solver vs
        // gallery) is checked once per solver in RunSolver, before this loop.
        static bool PrecondApplicable(PreconditionerKind solverKind, GridPrecond p, MatrixProfile tags)
        {
            if (p == GridPrecond.Identity) return true;
            bool sym = IsSymmetricPrecond(p);
            if (solverKind == PreconditionerKind.SymmetricBSR && !sym) return false;         // sym solver rejects nonsym-M
            if (sym && (tags & MatrixProfile.SPD) == 0) return false;                        // sym-M needs SPD gallery
            return true;
        }

        static bool IsSymmetricPrecond(GridPrecond p) =>
            p == GridPrecond.BlockJacobi || p == GridPrecond.SSOR || p == GridPrecond.IC0 ||
            p == GridPrecond.Chebyshev || p == GridPrecond.FSAI || p == GridPrecond.AdditiveSchwarz || p == GridPrecond.AMG;

        // The SECONDARY known-good set: cg on an SPD gallery with a strong SPD preconditioner. These
        // converge in the preconditioner battery, so a MaxIterations/other outcome here is a regression.
        static bool IsKnownGoodCombo(GridSolver solver, GridPrecond precond, MatrixProfile tags) =>
            solver == GridSolver.Cg && (tags & MatrixProfile.SPD) != 0 &&
            (precond == GridPrecond.IC0 || precond == GridPrecond.FSAI || precond == GridPrecond.AMG || precond == GridPrecond.BlockJacobi);

        // Builds the requested preconditioner (Identity => plain Solve), runs the solve, classifies.
        // TPre is inferred per arm so no boxing. All allocations are Allocator.Temp (job/frame scope
        // frees them; no Dispose needed, AMG included).
        static CellOutcome RunCell<TInv>(TInv inv, in fProxyBSR A, in fProxyBSROperator op, in fProxyN b,
                                         ref fProxyN x, GridPrecond pk, out fProxy resid)
            where TInv : struct, IfProxySquareSolverInvoker
        {
            SolveInfo info;
            switch (pk)
            {
                case GridPrecond.Identity:        info = inv.Solve(in op, in b, ref x); break;
                case GridPrecond.BlockJacobi:     { var M = new fProxyBlockJacobi(in A, Allocator.Temp);     info = inv.SolveWithPrecond(in op, in M, in b, ref x); } break;
                case GridPrecond.SSOR:            { var M = new fProxySSOR(in A, Allocator.Temp);            info = inv.SolveWithPrecond(in op, in M, in b, ref x); } break;
                case GridPrecond.IC0:             { var M = new fProxyIC0(in A, Allocator.Temp);             info = inv.SolveWithPrecond(in op, in M, in b, ref x); } break;
                case GridPrecond.Chebyshev:       { var M = new fProxyChebyshev(in A, Allocator.Temp);       info = inv.SolveWithPrecond(in op, in M, in b, ref x); } break;
                case GridPrecond.FSAI:            { var M = new fProxyFSAI(in A, Allocator.Temp);            info = inv.SolveWithPrecond(in op, in M, in b, ref x); } break;
                case GridPrecond.AdditiveSchwarz: { var M = new fProxyAdditiveSchwarz(in A, Allocator.Temp); info = inv.SolveWithPrecond(in op, in M, in b, ref x); } break;
                case GridPrecond.AMG:             { var amg = new fProxyAMG(in A, out _, Allocator.Temp); var M = new fProxyAMGPreconditioner(in amg); info = inv.SolveWithPrecond(in op, in M, in b, ref x); } break;
                case GridPrecond.ILU0:            { var M = new fProxyILU0(in A, Allocator.Temp);            info = inv.SolveWithPrecond(in op, in M, in b, ref x); } break;
                case GridPrecond.SPAI:            { var M = new fProxySPAI(in A, Allocator.Temp);            info = inv.SolveWithPrecond(in op, in M, in b, ref x); } break;
                case GridPrecond.RestrictedSchwarz: { var M = new fProxyRestrictedSchwarz(in A, Allocator.Temp); info = inv.SolveWithPrecond(in op, in M, in b, ref x); } break;
                default: info = default; break;
            }
            return Classify(info, in A, in x, in b, inv.Tol, out resid);
        }

        // Errored = NaN/Inf iterate OR Breakdown/Degenerate status. FalseConverged = Converged status
        // whose fresh relative residual overshoots the generous band (must never happen -- the
        // honesty invariant). Converged = Converged status inside the band. Anything else is the sole
        // remaining status, MaxIterations (allowed).
        static CellOutcome Classify(SolveInfo info, in fProxyBSR A, in fProxyN x, in fProxyN b, fProxy tol, out fProxy resid)
        {
            bool anyBad = false;
            for (int i = 0; i < x.N; i++)
                if (math.isnan(x[i]) || math.isinf(x[i])) { anyBad = true; break; }

            if (anyBad || info.status == IterativeSolveStatus.Breakdown || info.status == IterativeSolveStatus.Degenerate)
            {
                resid = anyBad ? (fProxy)(-1) : (fProxy)(int)info.status;
                return CellOutcome.Errored;
            }

            resid = fProxyKrylovBatteryOracles.RelResidualBSR(in A, in x, in b);
            fProxy bound = (fProxy)100 * tol;

            if (info.status == IterativeSolveStatus.Converged)
                return resid <= bound ? CellOutcome.Converged : CellOutcome.FalseConverged;

            return CellOutcome.MaxIterations;
        }

        void Record(bool ok, fProxy which, int solver, int precond, CellOutcome outcome, fProxy resid)
        {
            if (!ok && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = which;
                Fail[2] = (fProxy)solver;
                Fail[3] = (fProxy)precond;
                Fail[4] = (fProxy)(int)outcome;
                Fail[5] = resid;
            }
            Assert.IsTrue(ok);
        }
    }

    static readonly GalleryBSRMatrix[] Galleries =
    {
        GalleryBSRMatrix.Laplacian2D_16x16,
        GalleryBSRMatrix.RandomSparseSPD_120_2,
        GalleryBSRMatrix.RandomSparseNonsym_80,
    };

    public static GalleryBSRMatrix[] GetGalleries() => Galleries;

    [TestCaseSource(nameof(GetGalleries))]
    public void KrylovGrid(GalleryBSRMatrix gallery)
    {
        var fail = new NativeArray<fProxy>(6, Allocator.TempJob);
        try
        {
            new TestJob { Gallery = gallery, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
            {
                string which = fail[1] == (fProxy)0 ? "honesty-net" : "known-good";
                Assert.Fail($"{gallery}: {which} solver={(GridSolver)(int)fail[2]} precond={(GridPrecond)(int)fail[3]} outcome={(CellOutcome)(int)fail[4]} resid={fail[5]}");
            }
        }
        finally { fail.Dispose(); }
    }
}
