using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Unit battery for QP's PERSISTENT working-set factorization (fProxyQPFactorState + QP.TryAddToFactor /
// DropFromFactor / ApplyFactorQtForward / FormNullSpaceBasisFromFactor / RefactorWorkingSet) -- the
// up/downdated QR of A_Wᵀ the active-set loop carries across iterations. INTERNAL API, reached via the
// same InternalsVisibleTo route as QPActiveSetTests.fProxy.cs; same Burst job + Fail[] pattern.
//
// The factorization invariants verified after every add/drop below (VerifyState):
//   (1) Q̂ᵀ·a_kk == R's column kk on its leading entries, ~0 below them and on the whole tail -- i.e.
//       the log-represented Q̂ and the maintained R really factor the CURRENT A_Wᵀ, column by column.
//   (2) Z from the log satisfies A_W·Z ~ 0 (null-space property) and ZᵀZ ~ I (orthonormality).
// These pin the update algebra itself (reflector replay order, Givens convention, R shifts), not just
// end-to-end solver answers.
//
// CornerToInterior is the integration case: box QP started at a fully-tight corner (k = n = 12) whose
// optimum is interior, forcing 12 consecutive drops -- more than fProxyQPFactorState.DeadCap (8), so at
// least one RefactorWorkingSet fallback runs inside the loop; it also passes through k = 0 (empty
// working set) before converging.
public class fProxyQPFactorStateTests
{
    // ================================================================================================
    // Add / drop / re-add sequence over a mixed working set (general rows + a bound row), verifying
    // the factorization invariants after every mutation. Drops hit the middle, first, and last column
    // (the last-column drop produces no rotations -- the shift/rotation loop's empty edge).
    // ================================================================================================
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct UpDownDateJob : IJob
    {
        public NativeArray<double> Fail;

        static void BuildCol(in fProxyMxN A, int m, int n, int t, WorkingSetStatus st, ref fProxyN v)
        {
            fProxy sign = st == WorkingSetStatus.ActiveUpper ? (fProxy)(-1) : (fProxy)1;
            if (t < m)
                for (int i = 0; i < n; i++) v[i] = sign * A[t, i];
            else
            {
                for (int i = 0; i < n; i++) v[i] = (fProxy)0;
                v[t - m] = sign;
            }
        }

        void VerifyState(ref fProxyQPFactorState s, in fProxyMxN A, int m, int n,
                         NativeArray<byte> statusByT, int baseId, double tol)
        {
            int k = s.k, nz = n - k;
            var v = new fProxyN(n, Allocator.Temp, false);

            // (1) Q̂ᵀ A_Wᵀ == [R; 0], column by column
            double dev = 0;
            for (int kk = 0; kk < k; kk++)
            {
                int t = s.rowOfCol[kk];
                BuildCol(in A, m, n, t, (WorkingSetStatus)statusByT[t], ref v);
                QP.ApplyFactorQtForward(ref s, ref v);
                for (int i = 0; i <= kk; i++) dev = math.max(dev, math.abs((double)(v[i] - s.R[i, kk])));
                for (int i = kk + 1; i < n; i++) dev = math.max(dev, math.abs((double)v[i]));
            }
            H.AssertLE(Fail, baseId + 0, dev, tol);

            // (2) Z: A_W Z == 0 and ZᵀZ == I
            if (nz > 0)
            {
                var Z = new fProxyMxN(n, nz, Allocator.Temp, true);
                QP.FormNullSpaceBasisFromFactor(ref s, ref Z);

                double devNull = 0;
                for (int kk = 0; kk < k; kk++)
                {
                    int t = s.rowOfCol[kk];
                    BuildCol(in A, m, n, t, (WorkingSetStatus)statusByT[t], ref v);
                    for (int j = 0; j < nz; j++)
                    {
                        double acc = 0;
                        for (int i = 0; i < n; i++) acc += (double)v[i] * (double)Z[i, j];
                        devNull = math.max(devNull, math.abs(acc));
                    }
                }
                H.AssertLE(Fail, baseId + 1, devNull, tol);

                double devOrtho = 0;
                for (int a = 0; a < nz; a++)
                    for (int b = 0; b < nz; b++)
                    {
                        double acc = 0;
                        for (int i = 0; i < n; i++) acc += (double)Z[i, a] * (double)Z[i, b];
                        devOrtho = math.max(devOrtho, math.abs(acc - (a == b ? 1.0 : 0.0)));
                    }
                H.AssertLE(Fail, baseId + 2, devOrtho, tol);
                Z.Dispose();
            }
            v.Dispose();
        }

        void Add(ref fProxyQPFactorState s, in fProxyMxN A, int m, int n, int t, WorkingSetStatus st,
                 NativeArray<byte> statusByT, fProxy thr, int id)
        {
            H.AssertTrue(Fail, id, QP.TryAddToFactor(in A, m, n, t, st, ref s, thr));
            statusByT[t] = (byte)st;
        }

        void Drop(ref fProxyQPFactorState s, int col, NativeArray<byte> statusByT)
        {
            statusByT[s.rowOfCol[col]] = (byte)WorkingSetStatus.Inactive;
            QP.DropFromFactor(col, ref s);
        }

        public void Execute()
        {
            const int n = 6, m = 3;
            const int T = m + n;
            double tol = /*+choose[5e-5|1e-12]*/5e-5/*-choose*/;

            var A = new fProxyMxN(m, n, Allocator.Temp);
            A[0, 0] = 1f; A[0, 1] = 2f; A[0, 3] = -1f; A[0, 4] = 0.5f;
            A[1, 1] = 1f; A[1, 2] = -1f; A[1, 3] = 2f; A[1, 5] = 1f;
            A[2, 0] = 3f; A[2, 2] = 1f; A[2, 4] = 2f; A[2, 5] = -1f;

            var statusByT = new NativeArray<byte>(T, Allocator.Temp);
            var s = fProxyQPFactorState.Create(n);
            fProxy thr = Consts.fProxyZeroThreshold * math.max(Norms.LInf(in A), (fProxy)1);

            // build a 3-column working set: two general rows (one upper-oriented) + one bound row
            Add(ref s, in A, m, n, 0, WorkingSetStatus.ActiveLower, statusByT, thr, 1);
            Add(ref s, in A, m, n, 1, WorkingSetStatus.ActiveUpper, statusByT, thr, 2);
            Add(ref s, in A, m, n, m + 2, WorkingSetStatus.ActiveLower, statusByT, thr, 3);
            VerifyState(ref s, in A, m, n, statusByT, 10, tol);

            // drop the MIDDLE column (one Givens rotation), then re-add a new row over the rotated log
            Drop(ref s, 1, statusByT);
            VerifyState(ref s, in A, m, n, statusByT, 20, tol);

            Add(ref s, in A, m, n, 2, WorkingSetStatus.ActiveLower, statusByT, thr, 4);
            VerifyState(ref s, in A, m, n, statusByT, 30, tol);

            // drop the FIRST column (rotation chain over every remaining column)
            Drop(ref s, 0, statusByT);
            VerifyState(ref s, in A, m, n, statusByT, 40, tol);

            // drop the LAST column (no rotations at all)
            Drop(ref s, s.k - 1, statusByT);
            H.AssertTrue(Fail, 5, s.k == 1);
            VerifyState(ref s, in A, m, n, statusByT, 50, tol);

            s.Dispose();
            statusByT.Dispose();
            A.Dispose();
        }
    }

    // ================================================================================================
    // Rank guard through the update path: a duplicated (scaled) row must be REJECTED by
    // TryAddToFactor with the state left untouched, exactly like the from-scratch trial factor it
    // replaced; an independent row must still be accepted afterwards.
    // ================================================================================================
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct DependentAddJob : IJob
    {
        public NativeArray<double> Fail;

        public void Execute()
        {
            const int n = 4, m = 3;

            var A = new fProxyMxN(m, n, Allocator.Temp);
            A[0, 0] = 1f; A[0, 1] = 1f;
            A[1, 0] = 2f; A[1, 1] = 2f;   // = 2 * row 0: dependent
            A[2, 2] = 1f; A[2, 3] = 1f;

            var s = fProxyQPFactorState.Create(n);
            fProxy thr = (fProxy)/*+choose[1e-4|1e-10]*/1e-4/*-choose*/;

            H.AssertTrue(Fail, 1, QP.TryAddToFactor(in A, m, n, 0, WorkingSetStatus.ActiveLower, ref s, thr));
            int k0 = s.k, ops0 = s.opCount, refl0 = s.reflCount;

            H.AssertTrue(Fail, 2, !QP.TryAddToFactor(in A, m, n, 1, WorkingSetStatus.ActiveLower, ref s, thr));
            H.AssertTrue(Fail, 3, s.k == k0 && s.opCount == ops0 && s.reflCount == refl0);

            H.AssertTrue(Fail, 4, QP.TryAddToFactor(in A, m, n, 2, WorkingSetStatus.ActiveLower, ref s, thr));
            H.AssertTrue(Fail, 5, s.k == 2);

            s.Dispose();
            A.Dispose();
        }
    }

    // ================================================================================================
    // Integration: box QP from a fully-tight corner (k = n = 12) with an interior optimum -- 12
    // consecutive drops force the DeadCap refactor fallback inside qpActiveSetCore's loop, and
    // the working set passes through k = 0 before the final step. min 1/2 xᵀx - 2·1ᵀx, 0 <= x <= 10,
    // x0 = 0: optimum x* = 2·ones (strictly interior), objective = -2n.
    // ================================================================================================
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CornerToInteriorJob : IJob
    {
        public NativeArray<double> Fail;

        public void Execute()
        {
            const int n = 12;
            var arena = new Arena(Allocator.Persistent);

            var Q = arena.fProxyMat(n, n);
            for (int i = 0; i < n; i++) Q[i, i] = 1f;
            var c = arena.fProxyVec(n, (fProxy)(-2));
            var A = arena.fProxyMat(0, n);
            var b = arena.fProxyVec(0);
            var senses = new NativeArray<ConstraintSense>(0, Allocator.Temp);
            var xl = arena.fProxyVec(n);                  // 0
            var xu = arena.fProxyVec(n, (fProxy)10);
            var x = arena.fProxyVec(n);                   // x0 = 0: every lower bound tight

            var info = QP.qpActiveSetCore(in Q, in c, in A, in b, senses, in xl, in xu, ref x, out double obj, 0);

            H.AssertTrue(Fail, 1, info.status == QPStatus.Optimal);
            H.AssertTrue(Fail, 2, info.iterations >= n);   // at least one drop per variable
            double tol = /*+choose[1e-4|1e-10]*/1e-4/*-choose*/;
            for (int i = 0; i < n; i++)
                H.AssertLE(Fail, 3, math.abs((double)x[i] - 2.0), tol);
            H.AssertLE(Fail, 4, math.abs(obj - (-2.0 * n)), tol * n);

            senses.Dispose();
            arena.Dispose();
        }
    }

    [Test] public void UpDownDate() => H.Run(fail => new UpDownDateJob { Fail = fail }.Run());
    [Test] public void DependentAdd() => H.Run(fail => new DependentAddJob { Fail = fail }.Run());
    [Test] public void CornerToInterior() => H.Run(fail => new CornerToInteriorJob { Fail = fail }.Run());

    // ---- shared test-side helpers (Fail[]-array Burst diagnostic pattern, see QPEqpTests.fProxy.cs) ----
    static class H
    {
        public static void AssertTrue(NativeArray<double> fail, int id, bool cond)
        {
            if (!cond && fail[0] == 0) { fail[0] = 1; fail[1] = id; fail[2] = 0; fail[3] = 1; fail[4] = 0; }
            Assert.IsTrue(cond);
        }
        public static void AssertLE(NativeArray<double> fail, int id, double val, double limit)
        {
            bool ok = val <= limit;
            if (!ok && fail[0] == 0) { fail[0] = 1; fail[1] = id; fail[2] = val; fail[3] = limit; fail[4] = val - limit; }
            Assert.IsTrue(ok);
        }

        public static void Run(Action<NativeArray<double>> runJob)
        {
            var fail = new NativeArray<double>(5, Allocator.TempJob);
            try
            {
                runJob(fail);
                if (fail[0] != 0)
                    Assert.Fail($"check {fail[1]}: got {fail[2]:G6}, limit/expected {fail[3]:G6}, diff {fail[4]:G6}");
            }
            finally { fail.Dispose(); }
        }
    }
}
