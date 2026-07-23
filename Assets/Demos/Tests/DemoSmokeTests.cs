using LinearAlgebra;
using LinearAlgebra.Control;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LinearAlgebraDemos.Tests
{
    /// <summary>
    /// Headless smoke tests: every demo's Burst job runs once (or for a short
    /// simulated horizon) with real inputs and must report success and finite
    /// output. Catches runtime contract violations (dimension guards, solver
    /// failures) that compilation cannot.
    /// </summary>
    public class DemoSmokeTests
    {
        [Test]
        public void LeastSquaresJob_QuadricFit_Succeeds()
        {
            var points = new NativeArray<float3>(128, Allocator.TempJob);
            var coeffs = new NativeArray<float>(6, Allocator.TempJob);
            var stats = new NativeArray<float>(2, Allocator.TempJob);

            new GenerateAndFitJob
            {
                Points = points, Coeffs = coeffs, Stats = stats,
                Model = 1, NoiseSigma = 0.02f, OutlierFraction = 0f, OutlierScale = 0f,
                Time = 1.7f, Seed = 7u,
            }.Run();

            Assert.IsTrue(stats[1] == 1f, "QR solve failed");
            Assert.IsTrue(stats[0] < 0.1f, $"rms residual too high: {stats[0]}");

            points.Dispose(); coeffs.Dispose(); stats.Dispose();
        }

        [Test]
        public void LadJob_OutlierRobust_L1BeatsL2()
        {
            var points = new NativeArray<float3>(256, Allocator.TempJob);
            var l2 = new NativeArray<float>(3, Allocator.TempJob);
            var l1 = new NativeArray<float>(3, Allocator.TempJob);
            var stats = new NativeArray<float>(5, Allocator.TempJob);

            new LadFitJob
            {
                Points = points, L2Coeffs = l2, L1Coeffs = l1, Stats = stats,
                NoiseSigma = 0.02f, OutlierFraction = 0.3f, OutlierScale = 4f,
                Tau = 0.5f, Time = 0.9f, Seed = 11u,
            }.Run();

            Assert.IsTrue(stats[3] == 1f, "L1 solve not optimal");
            Assert.IsTrue(stats[4] == 1f, "L2 solve failed");
            // upward-biased outliers drag the L2 intercept up; L1 must sit well below it
            Assert.IsTrue(l1[2] < l2[2], $"L1 intercept {l1[2]} not below L2 {l2[2]}");

            points.Dispose(); l2.Dispose(); l1.Dispose(); stats.Dispose();
        }

        [Test]
        public void EconomyLPJob_WarmResolve_StaysOptimal()
        {
            const int n = 4, m = 7;
            var A = new floatMxN(m, n, Allocator.TempJob);
            var b = new floatN(m, Allocator.TempJob);
            var c = new floatN(n, Allocator.TempJob);
            var x = new floatN(n, Allocator.TempJob);
            var senses = new NativeArray<ConstraintSense>(m, Allocator.TempJob);
            var outStats = new NativeArray<float>(3, Allocator.TempJob);
            var basis = new LPBasis(n, m, Allocator.TempJob);
            var cache = new floatLPCache(n, m, Allocator.TempJob);

            float[,] use = { { 2f, 1f, 0.5f, 1.5f }, { 0.5f, 2f, 1f, 0.5f }, { 1f, 0.5f, 2f, 1.5f } };
            for (int r = 0; r < 3; r++) for (int j = 0; j < n; j++) A[r, j] = use[r, j];
            for (int j = 0; j < n; j++) A[3 + j, j] = 1f;
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;
            b[0] = 200f; b[1] = 150f; b[2] = 180f; for (int j = 0; j < n; j++) b[3 + j] = 80f;
            c[0] = -3f; c[1] = -4f; c[2] = -6f; c[3] = -5f;

            var job = new EconomyLPJob { A = A, B = b, C = c, X = x, Senses = senses, Basis = basis, Cache = cache, Out = outStats };
            IJobExtensions.RunByRef(ref job);
            Assert.IsTrue(outStats[2] == 1f, "cold LP solve not optimal");
            float coldObjective = outStats[0];
            float coldPivots = outStats[1];   // cold-from-scratch pivot count

            b[0] = 190f;   // small RHS perturbation -> warm re-solve
            IJobExtensions.RunByRef(ref job);
            Assert.IsTrue(outStats[2] == 1f, "warm LP re-solve not optimal");
            // Warm resume must stay cheap. On this 3-pivot toy there is almost nothing to warm up, so
            // warm ~ cold and the exact count carries +-1 of Burst codegen/struct-layout jitter; assert
            // "no worse than cold + 1 pivot" rather than an exact number. (A real resume FAILURE would
            // cost many extra pivots -- near a full cold solve -- not one.)
            Assert.IsTrue(outStats[1] <= coldPivots + 1, $"warm re-solve took {outStats[1]} pivots vs cold {coldPivots}");
            Assert.IsTrue(outStats[0] >= coldObjective - 5f, "objective moved implausibly far");

            A.Dispose(); b.Dispose(); c.Dispose(); x.Dispose();
            senses.Dispose(); outStats.Dispose(); basis.Dispose(); cache.Dispose();
        }

        [BurstCompile(CompileSynchronously = true)]
        struct LqrWarmRunJob : IJob
        {
            public floatMxN A, B, Q, R, K;
            public floatLQRState State;
            public NativeArray<int> Out;   // [0] = converged flag
            public void Execute()
            {
                var info = LQR.lqr(in A, in B, in Q, in R, ref K, ref State);
                Out[0] = info.status == RiccatiStatus.Converged ? 1 : 0;
            }
        }

        // Decisive regression for the warm-state native-mirror: a plain .Run() executes Execute on a
        // BY-VALUE copy of the job, so a plain-bool warm flag set inside would be dropped on return.
        // floatLQRState.populated is native-backed (NativeReference), so the write survives the copy and
        // is visible on the caller's `state` here. This FAILS on the pre-fix (plain-bool) code.
        [Test]
        public void LqrWarmState_SurvivesRunByValueCopy()
        {
            // Published discrete double integrator (same instance as the LQR literature test): converges.
            var A = new floatMxN(2, 2, Allocator.TempJob); A[0, 0] = 1; A[0, 1] = 1; A[1, 0] = 0; A[1, 1] = 1;
            var B = new floatMxN(2, 1, Allocator.TempJob); B[0, 0] = 0; B[1, 0] = 1;
            var Q = new floatMxN(2, 2, Allocator.TempJob); Q[0, 0] = 1; Q[0, 1] = 0; Q[1, 0] = 0; Q[1, 1] = 1;
            var R = new floatMxN(1, 1, Allocator.TempJob); R[0, 0] = 1;
            var K = new floatMxN(1, 2, Allocator.TempJob);
            var state = new floatLQRState(2, Allocator.TempJob);
            var outFlag = new NativeArray<int>(1, Allocator.TempJob);

            var job = new LqrWarmRunJob { A = A, B = B, Q = Q, R = R, K = K, State = state, Out = outFlag };
            Assert.IsFalse(state.populated, "fresh LQR state should not be populated");
            job.Run();   // plain .Run() -> Execute runs on a by-value copy of `job`
            Assert.IsTrue(outFlag[0] == 1, "cold LQR solve did not converge");
            Assert.IsTrue(state.populated,
                "warm state 'populated' did not survive the .Run() by-value copy -- a native-backed flag is what makes this pass; a plain bool would be lost");

            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); K.Dispose(); state.Dispose(); outFlag.Dispose();
        }

        // LP counterpart: solve the SAME LP twice via plain .Run() (by-value copy) through one cache +
        // basis. The second solve is a cache HIT (near-zero pivots) ONLY if BOTH the fProxyLPCache scalar
        // mirror and LPBasis.populated survived the copy. Without the native mirrors the second .Run()
        // re-seeds + re-solves cold (== first). Fails on the pre-fix code.
        [Test]
        public void EconomyLPJob_WarmState_SurvivesRunByValueCopy()
        {
            const int n = 4, m = 7;
            var A = new floatMxN(m, n, Allocator.TempJob);
            var b = new floatN(m, Allocator.TempJob);
            var c = new floatN(n, Allocator.TempJob);
            var x = new floatN(n, Allocator.TempJob);
            var senses = new NativeArray<ConstraintSense>(m, Allocator.TempJob);
            var outStats = new NativeArray<float>(3, Allocator.TempJob);
            var basis = new LPBasis(n, m, Allocator.TempJob);
            var cache = new floatLPCache(n, m, Allocator.TempJob);

            float[,] use = { { 2f, 1f, 0.5f, 1.5f }, { 0.5f, 2f, 1f, 0.5f }, { 1f, 0.5f, 2f, 1.5f } };
            for (int r = 0; r < 3; r++) for (int j = 0; j < n; j++) A[r, j] = use[r, j];
            for (int j = 0; j < n; j++) A[3 + j, j] = 1f;
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;
            b[0] = 200f; b[1] = 150f; b[2] = 180f; for (int j = 0; j < n; j++) b[3 + j] = 80f;
            c[0] = -3f; c[1] = -4f; c[2] = -6f; c[3] = -5f;

            var job = new EconomyLPJob { A = A, B = b, C = c, X = x, Senses = senses, Basis = basis, Cache = cache, Out = outStats };
            job.Run();   // cold solve on a by-value copy
            Assert.IsTrue(outStats[2] == 1f, "cold LP solve not optimal");
            float coldPivots = outStats[1];

            job.Run();   // SAME LP again -- a cache HIT (near-zero pivots) iff the warm state survived
            Assert.IsTrue(outStats[2] == 1f, "warm LP re-solve not optimal");
            float warmPivots = outStats[1];

            Assert.IsTrue(coldPivots >= 1f, $"cold solve did no pivots ({coldPivots}); cannot distinguish warm");
            Assert.IsTrue(warmPivots < coldPivots,
                $"warm re-solve of the UNCHANGED LP took {warmPivots} pivots vs cold {coldPivots} -- cache/basis warm state did not survive the .Run() by-value copy");

            A.Dispose(); b.Dispose(); c.Dispose(); x.Dispose();
            senses.Dispose(); outStats.Dispose(); basis.Dispose(); cache.Dispose();
        }

        [Test]
        public void TrussEigenJob_BracedSquare_PositiveSpectrum()
        {
            // 4-node square truss, both diagonals braced, two nodes pinned: stiff.
            float2[] nodes = { new float2(0, 0), new float2(1, 0), new float2(1, 1), new float2(0, 1) };
            int2[] bars = {
                new int2(0, 1), new int2(1, 2), new int2(2, 3), new int2(3, 0),
                new int2(0, 2), new int2(1, 3),
            };
            const int K = 2;
            int n = nodes.Length * 2;

            // assemble twice: symmetric lower-block storage AND full storage, same matrix
            var symBuilder = new floatBSRBuilder(nodes.Length, nodes.Length, 2, 2, Allocator.Temp, 32);
            var fullBuilder = new floatBSRBuilder(nodes.Length, nodes.Length, 2, 2, Allocator.Temp, 32);
            foreach (var bar in bars)
            {
                float2 d = nodes[bar.y] - nodes[bar.x];
                float2 u = math.normalize(d);
                float k = 5f / math.length(d);
                int lo = math.min(bar.x, bar.y), hi = math.max(bar.x, bar.y);
                for (int r = 0; r < 2; r++)
                    for (int c = 0; c < 2; c++)
                    {
                        float v = k * u[r] * u[c];
                        symBuilder.AddValue(2 * bar.x + r, 2 * bar.x + c, v);
                        symBuilder.AddValue(2 * bar.y + r, 2 * bar.y + c, v);
                        symBuilder.AddValue(2 * hi + r, 2 * lo + c, -v);

                        fullBuilder.AddValue(2 * bar.x + r, 2 * bar.x + c, v);
                        fullBuilder.AddValue(2 * bar.y + r, 2 * bar.y + c, v);
                        fullBuilder.AddValue(2 * hi + r, 2 * lo + c, -v);
                        fullBuilder.AddValue(2 * lo + r, 2 * hi + c, -v);
                    }
            }
            // penalty within ~3 decades of bar stiffness — see TrussStabilityDemo.Build
            for (int d = 0; d < 2; d++)
            {
                symBuilder.AddValue(0 + d, 0 + d, 1e3f);
                symBuilder.AddValue(2 + d, 2 + d, 1e3f);
                fullBuilder.AddValue(0 + d, 0 + d, 1e3f);
                fullBuilder.AddValue(2 + d, 2 + d, 1e3f);
            }

            var A = symBuilder.ToBSRSymmetric(Allocator.Temp);
            symBuilder.Dispose();
            var Afull = fullBuilder.ToBSR(Allocator.Temp);
            fullBuilder.Dispose();

            // diagnostic: symmetric-storage spMV must match full-storage spMV
            var probe = new floatN(n, Allocator.Temp);
            var ySym = new floatN(n, Allocator.Temp);
            var yFull = new floatN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) probe[i] = 0.1f * (i + 1);
            BSR.spMV(in A, in probe, ref ySym);
            BSR.spMV(in Afull, in probe, ref yFull);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(ySym[i] - yFull[i]) < 1e-2f * math.max(1f, math.abs(yFull[i])),
                    $"sym vs full spMV mismatch at {i}: {ySym[i]} vs {yFull[i]}");

            var precond = new floatBlockJacobi(in A, Allocator.Temp);
            var cache = new floatLOBPCGCache(n, K, Allocator.Temp);
            var outStats = new NativeArray<float>(2, Allocator.TempJob);

            var job = new TrussEigenJob
            {
                Op = new floatBSROperator(in A), Precond = precond, Cache = cache,
                Out = outStats, K = K,
            };
            IJobExtensions.RunByRef(ref job);

            Assert.IsTrue(outStats[1] == 1f, "LOBPCG did not converge");
            var lambda = job.Cache.lambda;

            // reference: same solve over full storage
            var precondFull = new floatBlockJacobi(in Afull, Allocator.Temp);
            var cacheFull = new floatLOBPCGCache(n, K, Allocator.Temp);
            var outFull = new NativeArray<float>(2, Allocator.TempJob);
            var jobFull = new TrussEigenJob
            {
                Op = new floatBSROperator(in Afull), Precond = precondFull, Cache = cacheFull,
                Out = outFull, K = K,
            };
            IJobExtensions.RunByRef(ref jobFull);
            var lambdaFull = jobFull.Cache.lambda;

            Assert.IsTrue(lambda[0] > 0.01f,
                $"lambda1 = {lambda[0]} (full-storage reference: {lambdaFull[0]}, converged={outFull[1]}) — braced square must be stiff");
            Assert.IsTrue(lambda[0] <= lambda[1], "eigenvalues not ascending");

            outStats.Dispose(); outFull.Dispose();
        }

        [Test]
        public void CartPoleJob_Stabilizes_From_Tilt()
        {
            var state = new NativeArray<float>(4, Allocator.TempJob);
            var outStats = new NativeArray<float>(4, Allocator.TempJob);
            var K = new floatMxN(1, 4, Allocator.TempJob);
            var lqr = new floatLQRState(4, Allocator.TempJob);
            state[2] = 0.25f;

            var job = new CartPoleStepJob
            {
                K = K, LqrState = lqr, State = state, Out = outStats,
                CartMass = 1f, PoleMass = 0.3f, PoleLength = 1f,
                QPos = 10f, QAngle = 50f, RCost = 1f, MaxForce = 30f,
                Dt = 1f / 240f, Steps = 4,
            };

            for (int frame = 0; frame < 120; frame++)   // 2 simulated seconds
                IJobExtensions.RunByRef(ref job);

            Assert.IsTrue(outStats[2] == 1f, "LQR did not converge");
            Assert.IsTrue(math.abs(state[2]) < 0.05f, $"pole not stabilized: theta = {state[2]}");
            Assert.IsTrue(math.abs(state[0]) < 2f, $"cart ran away: p = {state[0]}");

            state.Dispose(); outStats.Dispose(); K.Dispose(); lqr.Dispose();
        }

        [Test]
        public void DoubleCartPoleJob_Stays_Upright()
        {
            var state = new NativeArray<float>(6, Allocator.TempJob);
            var outStats = new NativeArray<float>(4, Allocator.TempJob);
            var K = new floatMxN(1, 6, Allocator.TempJob);
            var lqr = new floatLQRState(6, Allocator.TempJob);
            state[1] = 0.06f; state[2] = -0.08f;

            var job = new DoubleCartPoleStepJob
            {
                K = K, LqrState = lqr, State = state, Out = outStats,
                Mc = 1.5f, M1 = 0.3f, M2 = 0.3f, L1 = 0.6f, L2 = 0.6f,
                QPos = 8f, QAngle = 80f, RCost = 1f, MaxForce = 80f,
                Dt = 1f / 480f, Steps = 8,
            };

            for (int frame = 0; frame < 180; frame++)   // 3 simulated seconds
                IJobExtensions.RunByRef(ref job);

            Assert.IsTrue(outStats[3] == 1f, "upright mass matrix not SPD");
            Assert.IsTrue(outStats[2] == 1f, "LQR did not converge");
            Assert.IsTrue(math.abs(state[1]) < 0.05f && math.abs(state[2]) < 0.05f,
                $"double pole fell: th1={state[1]} th2={state[2]}");

            state.Dispose(); outStats.Dispose(); K.Dispose(); lqr.Dispose();
        }

        [Test]
        public void DroneJob_Reaches_Fixed_Target()
        {
            var state = new NativeArray<float>(6, Allocator.TempJob);
            var target = new NativeArray<float>(2, Allocator.TempJob);
            var outStats = new NativeArray<float>(4, Allocator.TempJob);
            var wind = new NativeArray<float>(1, Allocator.TempJob);
            var K = new floatMxN(2, 6, Allocator.TempJob);
            var lqr = new floatLQRState(6, Allocator.TempJob);
            state[1] = 1f;
            target[0] = 1.5f; target[1] = 2f;

            var job = new DroneStepJob
            {
                K = K, LqrState = lqr, State = state, Target = target, Out = outStats, Wind = wind,
                Mass = 0.8f, Inertia = 0.15f, Arm = 0.25f,
                QPos = 20f, QAngle = 10f, RCost = 0.5f, MaxRotorForce = 12f,
                Dt = 1f / 240f, Steps = 4,
            };

            for (int frame = 0; frame < 300; frame++)   // 5 simulated seconds
                IJobExtensions.RunByRef(ref job);

            Assert.IsTrue(outStats[3] == 1f, "LQR did not converge");
            float ex = state[0] - target[0], ez = state[1] - target[1];
            Assert.IsTrue(math.sqrt(ex * ex + ez * ez) < 0.15f,
                $"drone missed target: pos=({state[0]}, {state[1]})");

            state.Dispose(); target.Dispose(); outStats.Dispose(); wind.Dispose();
            K.Dispose(); lqr.Dispose();
        }

        [Test]
        public void SpringJob_Cloth_Falls_And_Solver_Converges()
        {
            const int W = 6, Hn = 5;
            int n = W * Hn;
            var pos = new NativeArray<float3>(n, Allocator.TempJob);
            var vel = new NativeArray<float3>(n, Allocator.TempJob);
            var pinned = new NativeArray<byte>(n, Allocator.TempJob);
            var outStats = new NativeArray<float>(3, Allocator.TempJob);

            for (int j = 0; j < Hn; j++)
                for (int i = 0; i < W; i++)
                {
                    int id = j * W + i;
                    pos[id] = new float3(i * 0.25f, 2f - j * 0.25f, 0f);
                    pinned[id] = (byte)(j == 0 ? 1 : 0);
                }

            int edgeCount = (W - 1) * Hn + W * (Hn - 1) + 2 * (W - 1) * (Hn - 1);
            var edges = new NativeArray<int2>(edgeCount, Allocator.TempJob);
            var restLen = new NativeArray<float>(edgeCount, Allocator.TempJob);
            int e = 0;
            void AddEdge(int a, int b)
            {
                edges[e] = new int2(math.min(a, b), math.max(a, b));
                restLen[e] = math.distance(pos[a], pos[b]);
                e++;
            }
            for (int j = 0; j < Hn; j++)
                for (int i = 0; i < W; i++)
                {
                    int id = j * W + i;
                    if (i + 1 < W) AddEdge(id, id + 1);
                    if (j + 1 < Hn) AddEdge(id, id + W);
                    if (i + 1 < W && j + 1 < Hn) { AddEdge(id, id + W + 1); AddEdge(id + 1, id + W); }
                }

            // soft springs so 1 s of sag clears the assert threshold (stiff cloth
            // sags millimetres — the first version of this test failed on physics,
            // not on the solver)
            const float h = 1f / 60f, stiffness = 60f, nodeMass = 0.1f;
            float h2k = h * h * stiffness;
            var builder = new floatBSRBuilder(n, n, 3, 3, Allocator.Temp, edgeCount * 2 + n);
            var degree = new NativeArray<float>(n, Allocator.Temp);
            for (int k = 0; k < edgeCount; k++)
            {
                int a = edges[k].x, b = edges[k].y;
                degree[a] += h2k; degree[b] += h2k;
                for (int d = 0; d < 3; d++)
                    builder.AddValue(3 * b + d, 3 * a + d, -h2k);
            }
            for (int i = 0; i < n; i++)
            {
                float mi = pinned[i] == 1 ? 1e7f : nodeMass;
                for (int d = 0; d < 3; d++)
                    builder.AddValue(3 * i + d, 3 * i + d, mi + degree[i]);
            }
            degree.Dispose();
            var A = builder.ToBSRSymmetric(Allocator.Temp);
            builder.Dispose();
            var precond = new floatIC0(in A, Allocator.Temp);   // zero-copy from symmetric-lower storage

            var job = new SpringStepJob
            {
                A = A, Precond = precond,
                Pos = pos, Vel = vel, Edges = edges, RestLen = restLen, Pinned = pinned,
                Out = outStats,
                Stiffness = stiffness, NodeMass = nodeMass, Damping = 0.4f, WindZ = 0f, H = h,
            };

            float startY = pos[n - 1].y;
            for (int step = 0; step < 60; step++) job.Run();

            Assert.IsTrue(outStats[1] == 1f, "PCG did not converge");
            Assert.IsTrue(pos[n - 1].y < startY - 0.02f,
                $"free corner did not fall under gravity (dy = {pos[n - 1].y - startY})");
            Assert.IsTrue(math.all(math.isfinite(pos[n - 1])), "positions not finite");

            pos.Dispose(); vel.Dispose(); pinned.Dispose(); outStats.Dispose();
            edges.Dispose(); restLen.Dispose();
        }

        [Test]
        public void CircuitJob_DCSource_Enforced_And_Diffuses()
        {
            const int W = 6, Hn = 4;
            int n = W * Hn, nu = n + 2;
            const float h = 1f / 60f, resistance = 1f, capacitance = 0.05f;
            int srcNode = 0, gndNode = n - 1;

            float g = 1f / resistance, ch = capacitance / h;
            var builder = new floatBSRBuilder(nu, nu, 1, 1, Allocator.Temp, n * 6);
            for (int i = 0; i < nu; i++) builder.AddValue(i, i, 0f);
            void AddResistor(int p, int q)
            {
                builder.AddValue(p, p, g); builder.AddValue(q, q, g);
                builder.AddValue(p, q, -g); builder.AddValue(q, p, -g);
            }
            for (int j = 0; j < Hn; j++)
                for (int i = 0; i < W; i++)
                {
                    int id = j * W + i;
                    builder.AddValue(id, id, ch);
                    if (i + 1 < W) AddResistor(id, id + 1);
                    if (j + 1 < Hn) AddResistor(id, id + W);
                }
            builder.AddValue(srcNode, n, 1f); builder.AddValue(n, srcNode, 1f);
            builder.AddValue(gndNode, n + 1, 1f); builder.AddValue(n + 1, gndNode, 1f);

            var A = builder.ToBSR(Allocator.Temp);
            builder.Dispose();
            var precond = new floatILU0(in A, Allocator.Temp);

            var voltages = new NativeArray<float>(nu, Allocator.TempJob);
            var outStats = new NativeArray<float>(4, Allocator.TempJob);

            var job = new CircuitStepJob
            {
                A = A, Precond = precond, Voltages = voltages, Out = outStats,
                NodeCount = n, CapOverH = ch, VSource = 3f,
            };
            for (int step = 0; step < 60; step++) job.Run();

            Assert.IsTrue(outStats[1] == 1f, "BiCGStab did not converge");
            Assert.IsTrue(math.abs(voltages[srcNode] - 3f) < 1e-3f,
                $"source constraint violated: v_src = {voltages[srcNode]}");
            Assert.IsTrue(math.abs(voltages[gndNode]) < 1e-3f,
                $"ground constraint violated: v_gnd = {voltages[gndNode]}");
            Assert.IsTrue(voltages[1] > voltages[gndNode - 1],
                "voltage did not decay with distance from the source");

            voltages.Dispose(); outStats.Dispose();
        }
    }
}
