using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using BULA;
using BULA.Control;

namespace BULA.Benchmarks
{
    // GENERATED per-dtype half of KalmanBenchmark (the timed IJobs + the instance builder + build+measure
    // methods). The dtype-agnostic harness (sizes/seeds, row formatter, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/KalmanBenchmark.cs.
    //
    // Every job loops predict+update (or predictFixed+updateFixed, or ekf/ukf predict+update) `steps`
    // times per Execute() call; Bench.Time's own 1 warmup + 4 timed calls run consecutively on the SAME
    // persistent state (fProxyKFState/fProxyUKFCache fields are NativeArray-backed, so mutations survive
    // across repeated job.Run() calls) -- harmless here since every size below uses contractive/bounded
    // dynamics (see BuildKFInstanceFProxy and EkfCycleFProxy's own comments). The final x/status are
    // written to xOut/statusOut from inside Execute() so the whole predict/update chain is a real,
    // non-eliminable dependency of the reported numbers.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct KfCycleJobFProxy : IJob
    {
        public fProxyKFState s;
        public fProxyMxN A, H, Q, R;
        public fProxyN z;
        public int steps;
        public NativeArray<double> xOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            KFInfo info = default;
            for (int k = 0; k < steps; k++)
            {
                Kalman.predict(ref s, in A, in Q);
                info = Kalman.update(ref s, in H, in R, in z);
            }
            xOut[0] = (double)s.x[0];
            statusOut[0] = (int)info.status;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct KfFixedCycleJobFProxy : IJob
    {
        public fProxyKFState s;
        public fProxyMxN A, H, Kss;
        public fProxyN z;
        public int steps;
        public NativeArray<double> xOut;

        public void Execute()
        {
            for (int k = 0; k < steps; k++)
            {
                Kalman.predictFixed(ref s, in A);
                Kalman.updateFixed(ref s, in Kss, in H, in z);
            }
            xOut[0] = (double)s.x[0];
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct KfSteadyStateGainJobFProxy : IJob
    {
        public fProxyMxN A, H, Q, R, Kss;
        public int reps;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            for (int r = 0; r < reps; r++)
            {
                var info = Kalman.steadyStateGain(in A, in H, in Q, in R, ref Kss);
                itersOut[0] = info.iterations;
                statusOut[0] = (int)info.status;
            }
        }
    }

    // Pendulum model/measurement for Section 4 (EKF vs UKF): state [theta, omega], NONLINEAR dynamics
    // theta'=theta+omega*dt, omega'=omega-(g/L)sin(theta)*dt; NONLINEAR measurement h=sin(theta). Same
    // dynamics as KalmanTests.fProxy.cs's own pendulum fixture, redeclared here (a separate assembly).
    public struct PendulumModelFProxy : IfProxyKFModel
    {
        public fProxy dt;
        public fProxy gOverL;

        public void F(in fProxyN x, in fProxyN u, ref fProxyN xNext)
        {
            fProxy th = x[0], om = x[1];
            xNext[0] = th + om * dt;
            xNext[1] = om - gOverL * math.sin(th) * dt;
        }

        public void JacobianF(in fProxyN x, in fProxyN u, ref fProxyMxN J)
        {
            fProxy th = x[0];
            J[0, 0] = (fProxy)1; J[0, 1] = dt;
            J[1, 0] = -gOverL * math.cos(th) * dt; J[1, 1] = (fProxy)1;
        }
    }

    public struct PendulumMeasFProxy : IfProxyKFMeasurement
    {
        public void H(in fProxyN x, ref fProxyN z) => z[0] = math.sin(x[0]);

        public void JacobianH(in fProxyN x, ref fProxyMxN J)
        {
            J[0, 0] = math.cos(x[0]); J[0, 1] = (fProxy)0;
        }
    }

    // Synthetic drone-scale nonlinear model for Section 4's n=12 row: a ring of n coupled damped
    // oscillators, xNext[i] = a*x[i] + b*sin(x[(i+1)%n]) -- smooth, and contractive REGARDLESS of state
    // magnitude (sin is bounded in [-1,1], so |a|<1 alone bounds every orbit), unlike the pendulum's
    // forward-Euler drift. Exists only to exercise ekfPredict/ukfPredict's Jacobian/sigma-point cost at
    // n=12, not to model any real system.
    public struct RingModelFProxy : IfProxyKFModel
    {
        public int n;
        public fProxy a;
        public fProxy b;

        public void F(in fProxyN x, in fProxyN u, ref fProxyN xNext)
        {
            for (int i = 0; i < n; i++)
            {
                int j = (i + 1) % n;
                xNext[i] = a * x[i] + b * math.sin(x[j]);
            }
        }

        public void JacobianF(in fProxyN x, in fProxyN u, ref fProxyMxN J)
        {
            for (int i = 0; i < n; i++)
                for (int k = 0; k < n; k++)
                    J[i, k] = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                int j = (i + 1) % n;
                J[i, i] += a;
                J[i, j] += b * math.cos(x[j]);
            }
        }
    }

    // Measurement for RingModelFProxy: picks m components evenly spread through the ring
    // (indices 0, stride, 2*stride, ...) through sin(), so the Jacobian stays a simple diagonal-in-
    // the-picked-columns pattern regardless of m/n.
    public struct RingMeasFProxy : IfProxyKFMeasurement
    {
        public int m;
        public int stride;

        public void H(in fProxyN x, ref fProxyN z)
        {
            for (int j = 0; j < m; j++) z[j] = math.sin(x[j * stride]);
        }

        public void JacobianH(in fProxyN x, ref fProxyMxN J)
        {
            int n = J.N_Cols;
            for (int i = 0; i < m; i++)
                for (int k = 0; k < n; k++)
                    J[i, k] = (fProxy)0;
            for (int j = 0; j < m; j++)
                J[j, j * stride] = math.cos(x[j * stride]);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EkfCycleJobFProxy : IJob
    {
        public fProxyKFState s;
        public PendulumModelFProxy model;
        public PendulumMeasFProxy meas;
        public fProxyMxN Q, R;
        public fProxyN u, z;
        public int steps;
        public NativeArray<double> xOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            KFInfo info = default;
            for (int k = 0; k < steps; k++)
            {
                Kalman.ekfPredict(ref s, in model, in u, in Q);
                info = Kalman.ekfUpdate(ref s, in meas, in R, in z);
            }
            xOut[0] = (double)s.x[0];
            statusOut[0] = (int)info.status;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct UkfCycleJobFProxy : IJob
    {
        public fProxyKFState s;
        public fProxyUKFCache cache;
        public PendulumModelFProxy model;
        public PendulumMeasFProxy meas;
        public fProxyMxN Q, R;
        public fProxyN u, z;
        public int steps;
        public NativeArray<double> xOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            KFInfo info = default;
            for (int k = 0; k < steps; k++)
            {
                Kalman.ukfPredict(ref s, ref cache, in model, in u, in Q);
                info = Kalman.ukfUpdate(ref s, ref cache, in meas, in R, in z);
            }
            xOut[0] = (double)s.x[0];
            statusOut[0] = (int)info.status;
        }
    }

    // Same shape as EkfCycleJobFProxy but over RingModelFProxy/RingMeasFProxy (Section 4's n=12 row).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EkfRingCycleJobFProxy : IJob
    {
        public fProxyKFState s;
        public RingModelFProxy model;
        public RingMeasFProxy meas;
        public fProxyMxN Q, R;
        public fProxyN u, z;
        public int steps;
        public NativeArray<double> xOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            KFInfo info = default;
            for (int k = 0; k < steps; k++)
            {
                Kalman.ekfPredict(ref s, in model, in u, in Q);
                info = Kalman.ekfUpdate(ref s, in meas, in R, in z);
            }
            xOut[0] = (double)s.x[0];
            statusOut[0] = (int)info.status;
        }
    }

    // Same shape as UkfCycleJobFProxy but over RingModelFProxy/RingMeasFProxy (Section 4's n=12 row).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct UkfRingCycleJobFProxy : IJob
    {
        public fProxyKFState s;
        public fProxyUKFCache cache;
        public RingModelFProxy model;
        public RingMeasFProxy meas;
        public fProxyMxN Q, R;
        public fProxyN u, z;
        public int steps;
        public NativeArray<double> xOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            KFInfo info = default;
            for (int k = 0; k < steps; k++)
            {
                Kalman.ukfPredict(ref s, ref cache, in model, in u, in Q);
                info = Kalman.ukfUpdate(ref s, ref cache, in meas, in R, in z);
            }
            xOut[0] = (double)s.x[0];
            statusOut[0] = (int)info.status;
        }
    }

    public static partial class KalmanBenchmark
    {
        // Diagonal-dominant, contractive dynamics (diag in [0.8,1.0), off-diagonal 0.1/n) -- bounded
        // state/covariance over any step count regardless of size. H = [I_m | 0] (selects the first m
        // state components), always full row rank so every update is well-posed regardless of seed.
        static void BuildKFInstanceFProxy(int n, int m, uint seed, Allocator allocator,
                                          out fProxyMxN A, out fProxyMxN H, out fProxyMxN Q, out fProxyMxN R)
        {
            var rng = new Unity.Mathematics.Random(seed);
            fProxy off = (fProxy)(0.1 / n);

            A = new fProxyMxN(n, n, allocator);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (i == j) ? rng.NextFProxy(0.8f, 1.0f) : rng.NextFProxy(-1f, 1f) * off;

            H = new fProxyMxN(m, n, allocator);
            for (int i = 0; i < m; i++) H[i, i] = (fProxy)1;

            Q = new fProxyMxN(n, n, allocator);
            for (int i = 0; i < n; i++) Q[i, i] = (fProxy)1e-3;

            R = new fProxyMxN(m, m, allocator);
            for (int i = 0; i < m; i++) R[i, i] = (fProxy)1e-2;
        }

        // ==== Section 1: linear predict+update, full covariance path ====
        static string KfCycleFProxy(int n, int m, int steps, uint seed)
        {
            BuildKFInstanceFProxy(n, m, seed, Allocator.Persistent, out var A, out var H, out var Q, out var R);
            var z = new fProxyN(m, Allocator.Persistent);   // zero measurement -- timing only, not a tracking-accuracy check

            var s = new fProxyKFState(n, m, Allocator.Persistent);
            var xOut = new NativeArray<double>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new KfCycleJobFProxy { s = s, A = A, H = H, Q = Q, R = R, z = z, steps = steps, xOut = xOut, statusOut = statusOut };
            var stat = Bench.Time(() => job.Run());
            string row = KalmanBenchmarkFmt.Row("fProxy", "full", n, m, steps, stat, statusOut[0]);

            s.Dispose(); xOut.Dispose(); statusOut.Dispose();
            A.Dispose(); H.Dispose(); Q.Dispose(); R.Dispose(); z.Dispose();
            return row;
        }

        // ==== Section 2: predictFixed+updateFixed (steady-state gain, no covariance math) ====
        static string KfFixedFProxy(int n, int m, int steps, uint seed)
        {
            BuildKFInstanceFProxy(n, m, seed, Allocator.Persistent, out var A, out var H, out var Q, out var R);
            var z = new fProxyN(m, Allocator.Persistent);

            // Untimed: steady-state gain from the SAME (A,H,Q,R) KfCycleFProxy builds at this (n,m), so
            // the two rows are a direct full-covariance-vs-fixed-gain comparison.
            var Kss = new fProxyMxN(n, m, Allocator.Persistent);
            Kalman.steadyStateGain(in A, in H, in Q, in R, ref Kss);

            var s = new fProxyKFState(n, m, Allocator.Persistent);
            var xOut = new NativeArray<double>(1, Allocator.Persistent);
            var job = new KfFixedCycleJobFProxy { s = s, A = A, H = H, Kss = Kss, z = z, steps = steps, xOut = xOut };
            var stat = Bench.Time(() => job.Run());
            // status: n/a -- predictFixed/updateFixed return no KFInfo (no covariance solve to fail).
            string row = KalmanBenchmarkFmt.Row("fProxy", "fixed-gain", n, m, steps, stat, -1);

            s.Dispose(); xOut.Dispose();
            A.Dispose(); H.Dispose(); Q.Dispose(); R.Dispose(); z.Dispose(); Kss.Dispose();
            return row;
        }

        // ==== Section 3: steadyStateGain one-shot cost ====
        static string SteadyStateGainFProxy(int n, int m, int reps, uint seed)
        {
            BuildKFInstanceFProxy(n, m, seed, Allocator.Persistent, out var A, out var H, out var Q, out var R);
            var Kss = new fProxyMxN(n, m, Allocator.Persistent);

            var itersOut = new NativeArray<int>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new KfSteadyStateGainJobFProxy { A = A, H = H, Q = Q, R = R, Kss = Kss, reps = reps, itersOut = itersOut, statusOut = statusOut };
            var stat = Bench.Time(() => job.Run());
            // RiccatiInfo is the same info type LQR.lqr reports -- reuse LQRBenchmarkFmt's own formatter.
            string row = LQRBenchmarkFmt.Row("fProxy", "steadyStateGain", n, m, reps, stat, itersOut[0], statusOut[0]);

            itersOut.Dispose(); statusOut.Dispose();
            A.Dispose(); H.Dispose(); Q.Dispose(); R.Dispose(); Kss.Dispose();
            return row;
        }

        // ==== Section 4: EKF vs UKF on the same pendulum model (n=2, m=1) ====
        // NonlinearSteps is far shorter than Sections 1-2's per-row steps -- the pendulum's own
        // forward-Euler self-simulation (ekfPredict/ukfPredict driving x via model.F every step, with no
        // independent ground truth to correct against) has a slow secular amplitude drift typical of
        // explicit Euler on an oscillatory system. NonlinearSteps=100 per Execute(), and Bench.Time's own
        // 1 warmup + 4 timed calls run on this SAME persistent state, so the cumulative step count is
        // ~500 per benchmark run -- past KalmanTests.fProxy.cs's own 80-step EKF acceptance test (which
        // exercises a different regime anyway: real tracking + noisy measurements + seeded P, vs this
        // zero-measurement self-simulation). Smeas = H P Hᵀ + R stays positive as long as R > 0
        // regardless of state/covariance magnitude, so the drift does not by itself trip the
        // Cholesky-based innovation solve into InnovationSolveFailed here. Applies identically to
        // UkfCycleFProxy below.
        static string EkfCycleFProxy(int steps)
        {
            var model = new PendulumModelFProxy { dt = (fProxy)0.05, gOverL = (fProxy)4 };
            var meas = new PendulumMeasFProxy();
            var Q = new fProxyMxN(2, 2, Allocator.Persistent); Q[0, 0] = (fProxy)1e-6; Q[1, 1] = (fProxy)1e-6;
            var R = new fProxyMxN(1, 1, Allocator.Persistent); R[0, 0] = (fProxy)1e-3;
            var u = new fProxyN(1, Allocator.Persistent);
            var z = new fProxyN(1, Allocator.Persistent);   // zero measurement -- timing only

            var s = new fProxyKFState(2, 1, Allocator.Persistent);
            s.x[0] = (fProxy)0.3;

            var xOut = new NativeArray<double>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new EkfCycleJobFProxy { s = s, model = model, meas = meas, Q = Q, R = R, u = u, z = z, steps = steps, xOut = xOut, statusOut = statusOut };
            var stat = Bench.Time(() => job.Run());
            string row = KalmanBenchmarkFmt.Row("fProxy", "EKF", 2, 1, steps, stat, statusOut[0]);

            s.Dispose(); Q.Dispose(); R.Dispose(); u.Dispose(); z.Dispose(); xOut.Dispose(); statusOut.Dispose();
            return row;
        }

        static string UkfCycleFProxy(int steps)
        {
            var model = new PendulumModelFProxy { dt = (fProxy)0.05, gOverL = (fProxy)4 };
            var meas = new PendulumMeasFProxy();
            var Q = new fProxyMxN(2, 2, Allocator.Persistent); Q[0, 0] = (fProxy)1e-6; Q[1, 1] = (fProxy)1e-6;
            var R = new fProxyMxN(1, 1, Allocator.Persistent); R[0, 0] = (fProxy)1e-3;
            var u = new fProxyN(1, Allocator.Persistent);
            var z = new fProxyN(1, Allocator.Persistent);

            var s = new fProxyKFState(2, 1, Allocator.Persistent);
            s.x[0] = (fProxy)0.3;
            var cache = new fProxyUKFCache(2, Allocator.Persistent);

            var xOut = new NativeArray<double>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new UkfCycleJobFProxy { s = s, cache = cache, model = model, meas = meas, Q = Q, R = R, u = u, z = z, steps = steps, xOut = xOut, statusOut = statusOut };
            var stat = Bench.Time(() => job.Run());
            string row = KalmanBenchmarkFmt.Row("fProxy", "UKF", 2, 1, steps, stat, statusOut[0]);

            s.Dispose(); cache.Dispose(); Q.Dispose(); R.Dispose(); u.Dispose(); z.Dispose(); xOut.Dispose(); statusOut.Dispose();
            return row;
        }

        // ==== Section 4 (drone-scale row): EKF vs UKF on the n-ring model, n=12/m=6 in practice ====
        // a=0.9 (self-decay), b=0.1 (coupling magnitude) keep every orbit bounded in roughly [-1,1]
        // regardless of step count (see RingModelFProxy's own comment) -- no drift-safety concern the
        // way the pendulum's forward-Euler self-simulation has, so NonlinearSteps is a wall-time choice
        // only here, not a numerical-safety one.
        static string EkfRingFProxy(int n, int m, int steps, uint seed)
        {
            var model = new RingModelFProxy { n = n, a = (fProxy)0.9, b = (fProxy)0.1 };
            var meas = new RingMeasFProxy { m = m, stride = n / m };
            var Q = new fProxyMxN(n, n, Allocator.Persistent);
            for (int i = 0; i < n; i++) Q[i, i] = (fProxy)1e-6;
            var R = new fProxyMxN(m, m, Allocator.Persistent);
            for (int i = 0; i < m; i++) R[i, i] = (fProxy)1e-3;
            var u = new fProxyN(1, Allocator.Persistent);   // ignored by model.F
            var z = new fProxyN(m, Allocator.Persistent);   // zero measurement -- timing only

            var s = new fProxyKFState(n, m, Allocator.Persistent);
            var rng = new Unity.Mathematics.Random(seed);
            for (int i = 0; i < n; i++) s.x[i] = rng.NextFProxy(-0.5f, 0.5f);

            var xOut = new NativeArray<double>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new EkfRingCycleJobFProxy { s = s, model = model, meas = meas, Q = Q, R = R, u = u, z = z, steps = steps, xOut = xOut, statusOut = statusOut };
            var stat = Bench.Time(() => job.Run());
            string row = KalmanBenchmarkFmt.Row("fProxy", "EKF", n, m, steps, stat, statusOut[0]);

            s.Dispose(); Q.Dispose(); R.Dispose(); u.Dispose(); z.Dispose(); xOut.Dispose(); statusOut.Dispose();
            return row;
        }

        static string UkfRingFProxy(int n, int m, int steps, uint seed)
        {
            var model = new RingModelFProxy { n = n, a = (fProxy)0.9, b = (fProxy)0.1 };
            var meas = new RingMeasFProxy { m = m, stride = n / m };
            var Q = new fProxyMxN(n, n, Allocator.Persistent);
            for (int i = 0; i < n; i++) Q[i, i] = (fProxy)1e-6;
            var R = new fProxyMxN(m, m, Allocator.Persistent);
            for (int i = 0; i < m; i++) R[i, i] = (fProxy)1e-3;
            var u = new fProxyN(1, Allocator.Persistent);
            var z = new fProxyN(m, Allocator.Persistent);

            var s = new fProxyKFState(n, m, Allocator.Persistent);
            var rng = new Unity.Mathematics.Random(seed);
            for (int i = 0; i < n; i++) s.x[i] = rng.NextFProxy(-0.5f, 0.5f);
            var cache = new fProxyUKFCache(n, Allocator.Persistent);

            var xOut = new NativeArray<double>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new UkfRingCycleJobFProxy { s = s, cache = cache, model = model, meas = meas, Q = Q, R = R, u = u, z = z, steps = steps, xOut = xOut, statusOut = statusOut };
            var stat = Bench.Time(() => job.Run());
            string row = KalmanBenchmarkFmt.Row("fProxy", "UKF", n, m, steps, stat, statusOut[0]);

            s.Dispose(); cache.Dispose(); Q.Dispose(); R.Dispose(); u.Dispose(); z.Dispose(); xOut.Dispose(); statusOut.Dispose();
            return row;
        }
    }
}
