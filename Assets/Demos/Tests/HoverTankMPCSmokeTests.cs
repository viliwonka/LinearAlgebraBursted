using BULA;
using BULA.Control;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LinearAlgebraDemos.Tests
{
    /// <summary>
    /// Headless smoke tests for <see cref="HoverTankMPCDemo"/>. Four things are assertable without
    /// a scene: the shape contract of <see cref="TerrainField"/>, the stability of the hover model
    /// <see cref="HoverTankMPCStepJob.BuildHoverModel"/> builds, the demo's own step job run against
    /// synthetic corner ranges, and the 6x12 <see cref="GimbalAllocation"/> driven on its own.
    /// Nothing here touches Physics, Rigidbody or raycasts.
    ///
    /// Every job is run rather than merely referenced: running a [BurstCompile] job is what forces
    /// Burst to compile it, and a compile failure inside the demo's control path is exactly the kind
    /// of breakage these tests exist to catch.
    /// </summary>
    public class HoverTankMPCSmokeTests
    {
        [Test]
        public void TerrainField_ApronIsFlat_FeaturesPresent_HillsGentle()
        {
            var stats = new NativeArray<float>(6, Allocator.TempJob);
            var job = new TerrainSampleJob { Out = stats, Step = 0.5f, Delta = 0.25f };
            IJobExtensions.RunByRef(ref job);

            float apronMax = stats[0], absMax = stats[1];
            float gentleSlope = stats[2], escarpSlope = stats[3], wallSlope = stats[4];
            float wallProminence = stats[5];
            stats.Dispose();

            // The apron is what the tank spawns on: exactly flat, not nearly flat.
            Assert.IsTrue(apronMax < 1e-5f,
                $"terrain is not flat inside the {TerrainField.ApronRadius} m apron (max |h| = {apronMax})");
            Assert.IsTrue(TerrainField.Height(0f, 0f) == 0f,
                $"spawn point is not at height 0 (h = {TerrainField.Height(0f, 0f)})");

            // Relief stays inside what an 8 m sense ray and a 6 m ride-height command can work with.
            Assert.IsTrue(absMax < 20f, $"terrain relief is out of scale (max |h| = {absMax} m)");

            // Away from the two deliberate features the hover loop has to be able to track the ground,
            // and it levels the hull to the ground plane, so the gradient is also the lean angle.
            Assert.IsTrue(gentleSlope < 0.65f,
                $"rolling hills are too steep for the hover loop (max gradient = {gentleSlope}, {math.degrees(math.atan(gentleSlope))} deg)");

            // ...and the two features have to actually bite.
            Assert.IsTrue(escarpSlope > 0.5f,
                $"escarpment is not a steep face (max gradient = {escarpSlope}, {math.degrees(math.atan(escarpSlope))} deg)");
            Assert.IsTrue(wallSlope > 2f,
                $"wall is not wall-like (max gradient = {wallSlope}, {math.degrees(math.atan(wallSlope))} deg)");
            Assert.IsTrue(wallProminence > 5f,
                $"wall does not stand proud of its own flanks (prominence = {wallProminence} m)");
        }

        [Test]
        public void HoverModel_Stabilizes_From_Perturbed_State()
        {
            const int n = 6, m = 3;

            HoverTankMPCStepJob.BuildHoverModel(
                1f / 60f,
                40f, 6f, 90f, 8f, 0.02f, 0.4f,
                Allocator.TempJob, out var A, out var B, out var Q, out var R);

            var K = new floatMxN(m, n, Allocator.TempJob);
            RiccatiInfo info = LQR.lqr(in A, in B, in Q, in R, ref K);
            Assert.IsTrue(info, $"hover LQR did not converge: {info.status}");

            var BK = new floatMxN(n, n, Allocator.TempJob);
            Blas.dot(in B, in K, ref BK);
            var Acl = new floatMxN(n, n, Allocator.TempJob);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    Acl[i, j] = A[i, j] - BK[i, j];

            var x = new NativeArray<float>(n, Allocator.TempJob);
            var xNext = new NativeArray<float>(n, Allocator.TempJob);
            x[0] = 0.8f; x[2] = 0.3f; x[4] = -0.25f;   // perturbed height / roll / pitch

            float norm = StateNorm(x, n);
            int steps = 0;
            const int maxSteps = 2000;
            while (norm >= 1e-3f && steps < maxSteps)
            {
                for (int i = 0; i < n; i++)
                {
                    float s = 0f;
                    for (int j = 0; j < n; j++) s += Acl[i, j] * x[j];
                    xNext[i] = s;
                }
                for (int i = 0; i < n; i++) x[i] = xNext[i];
                norm = StateNorm(x, n);
                steps++;
            }

            Assert.IsTrue(norm < 1e-3f, $"hover closed loop did not decay below 1e-3 within {maxSteps} steps (||x|| = {norm})");

            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose();
            K.Dispose(); BK.Dispose(); Acl.Dispose();
            x.Dispose(); xNext.Dispose();
        }

        /// <summary>
        /// Drives <see cref="HoverTankMPCStepJob"/> for 300 fixed steps against corner ranges frozen
        /// at "level height, rolled right-side-down", so the demanded wrench is constant and the
        /// allocation has something definite to converge to. Checks the allocation's own contract —
        /// optimal solves, controls inside the rig's absolute ranges and inside its per-step rate
        /// limits — and then that the converged wrench actually carries the hull's weight and rolls
        /// in the commanded direction.
        /// </summary>
        [Test]
        public void StepJob_Allocation_StaysFeasible_And_TracksDemand()
        {
            const float dt = 1f / 60f;
            const float mass = 1500f, gravity = 9.81f;
            const float rideHeight = 2f, rollInertia = 2100f, pitchInertia = 4600f;
            const float halfWidth = 2f, halfLength = 3f;
            const int steps = 300;

            float cornerDX = halfWidth * 0.9f, cornerDZ = halfLength * 0.9f;
            GimbalSettings settings = GimbalSettings.Default;

            var cornerHeights = new NativeArray<float>(4, Allocator.TempJob);
            var prevCorner = new NativeArray<float>(4, Allocator.TempJob);
            // Mean 2 m (no height error) but the right pair reads 0.72 m closer than the left pair,
            // which the estimate reads as roll = -0.2 rad: right side down.
            cornerHeights[0] = 2.36f; cornerHeights[1] = 1.64f;
            cornerHeights[2] = 2.36f; cornerHeights[3] = 1.64f;
            for (int i = 0; i < 4; i++) prevCorner[i] = cornerHeights[i];

            var hoverState = new NativeArray<float>(6, Allocator.TempJob);
            var hoverOut = new NativeArray<float>(4, Allocator.TempJob);
            var controls = new NativeArray<float>(GimbalAllocation.ControlCount, Allocator.TempJob);
            var allocOut = new NativeArray<QPInfo>(1, Allocator.TempJob);
            var wrenchOut = new NativeArray<float>(2 * GimbalAllocation.WrenchRows, Allocator.TempJob);
            var hoverK = new floatMxN(3, 6, Allocator.TempJob);
            var hoverLqr = new floatLQRState(6, Allocator.TempJob);

            float trim = math.clamp(mass * gravity / (4f * settings.maxThrust),
                                    settings.minThrust / settings.maxThrust, 1f);
            for (int i = 0; i < 4; i++) { controls[i] = 0f; controls[4 + i] = 0f; controls[8 + i] = trim; }

            float angLo = math.radians(settings.servoMinDeg), angHi = math.radians(settings.servoMaxDeg);
            float thrLo = settings.minThrust / settings.maxThrust;
            float dAngle = math.radians(settings.servoRateDeg) * dt;
            float dThrottle = settings.thrustRate / settings.maxThrust * dt;

            var job = new HoverTankMPCStepJob
            {
                CornerHeights = cornerHeights, PrevCornerHeights = prevCorner,
                HoverState = hoverState,
                HoverK = hoverK, HoverLqrState = hoverLqr, HoverOut = hoverOut,
                Mass = mass, RollInertia = rollInertia, PitchInertia = pitchInertia, Gravity = gravity,
                QHeight = 40f, QHeightRate = 6f, QTilt = 90f, QTiltRate = 8f,
                RThrust = 0.02f, RTorque = 0.4f,
                TargetRideHeight = rideHeight, CornerDX = cornerDX, CornerDZ = cornerDZ, Dt = dt,

                Controls = controls, AllocOut = allocOut, WrenchOut = wrenchOut,
                Settings = settings, Health = new float4(1f),
                // Side flanks at hull mid-height, the geometry the demo builds.
                MountX = new float4(-halfWidth, halfWidth, -halfWidth, halfWidth),
                MountY = float4.zero,
                MountZ = new float4(halfLength, halfLength, -halfLength, -halfLength),
                MountArm = math.max(halfWidth, halfLength),
                DriveInput = 0f, DriveForce = 9000f,
                StrafeInput = 0f, StrafeForce = 7000f,
                SteerInput = 0f, SteerTorque = 9000f,
                BrakeInput = false, BrakeForce = 8000f, BrakeGain = 3000f, BrakeYawGain = 12000f,
                IdleLinearGain = 1500f, IdleAngularGain = 6500f,
                ForwardSpeed = 0f, LateralSpeed = 0f, YawRate = 0f, TiltCos = 1f,
            };

            var before = new float[GimbalAllocation.ControlCount];
            for (int s = 0; s < steps; s++)
            {
                for (int j = 0; j < GimbalAllocation.ControlCount; j++) before[j] = controls[j];

                // RunByRef mutates the job in place, so the warm floatLQRState carries across steps
                // without a copy-back.
                IJobExtensions.RunByRef(ref job);

                Assert.IsTrue(hoverOut[1] == 1f, $"step {s}: hover LQR did not converge");
                Assert.IsTrue(allocOut[0].status == QPStatus.Optimal,
                    $"step {s}: allocation QP returned {allocOut[0].status}");

                // 8 gimbal angles (4 pitch then 4 yaw), then 4 throttles.
                for (int i = 0; i < 2 * GimbalAllocation.Thrusters; i++)
                {
                    Assert.IsTrue(controls[i] >= angLo - 1e-4f && controls[i] <= angHi + 1e-4f,
                        $"step {s}: servo {i} left its range at {math.degrees(controls[i])} deg");
                    Assert.IsTrue(math.abs(controls[i] - before[i]) <= dAngle + 1e-4f,
                        $"step {s}: servo {i} slewed {math.degrees(math.abs(controls[i] - before[i]))} deg, limit {math.degrees(dAngle)}");
                }
                for (int i = 0; i < 4; i++)
                {
                    int u = 2 * GimbalAllocation.Thrusters + i;
                    Assert.IsTrue(controls[u] >= thrLo - 1e-4f && controls[u] <= 1f + 1e-4f,
                        $"step {s}: throttle {i} left its range at {controls[u]}");
                    Assert.IsTrue(math.abs(controls[u] - before[u]) <= dThrottle + 1e-4f,
                        $"step {s}: throttle {i} slewed {math.abs(controls[u] - before[u])}, limit {dThrottle}");
                }
            }

            // Converged wrench. Lift is the well-conditioned channel and must carry the weight; roll is
            // traded against the trim regularizer, so it is checked for direction and substance only.
            float weight = mass * gravity;
            float demandedLift = wrenchOut[1], demandedRoll = wrenchOut[5];
            float achievedLift = wrenchOut[7], achievedRoll = wrenchOut[11];

            Assert.IsTrue(math.abs(hoverState[0]) < 1e-5f,
                $"height error should be zero for these ranges, got {hoverState[0]}");
            Assert.IsTrue(demandedRoll > 0f,
                $"a right-side-down hull should be commanded to roll right side up, got {demandedRoll} Nm");

            Assert.IsTrue(math.abs(achievedLift - demandedLift) < 0.02f * weight,
                $"lift off by {achievedLift - demandedLift} N ({achievedLift} vs {demandedLift}), weight {weight} N");
            Assert.IsTrue(achievedRoll > 0.75f * demandedRoll && achievedRoll < 1.25f * demandedRoll,
                $"roll torque {achievedRoll} Nm does not track the demanded {demandedRoll} Nm");

            cornerHeights.Dispose(); prevCorner.Dispose();
            hoverState.Dispose(); hoverOut.Dispose();
            controls.Dispose(); allocOut.Dispose(); wrenchOut.Dispose();
            hoverK.Dispose(); job.HoverLqrState.Dispose();
        }

        /// <summary>
        /// The point of the second gimbal axis: a commanded SIDEWAYS force is reachable. Every thrust
        /// vector of a single-axis rig lies in a forward-up plane, so their sum does too and no lateral
        /// force exists at any control setting; here one is asked for and delivered.
        ///
        /// Lift is commanded at the hull's weight rather than at zero on purpose. A thruster only ever
        /// pushes and its cone is capped at <see cref="GimbalAllocation.MaxGimbalDeg"/>, so the total
        /// thrust always has a strictly positive vertical component and "2000 N sideways and nothing
        /// else" is not a wrench any such rig can produce. Hovering while strafing is.
        /// </summary>
        [Test]
        public void Allocation_Achieves_CommandedLateralWrench()
        {
            const float lateral = 2000f;
            var desired = new GimbalWrench { Lateral = lateral, Lift = AllocWeight };

            AllocResult r = RunAllocation(in desired, new float4(1f), 0f);

            Assert.IsTrue(r.Info.status == QPStatus.Optimal, $"allocation QP returned {r.Info.status}");

            Assert.IsTrue(math.abs(r.Achieved[0] - lateral) < 0.05f * lateral,
                $"lateral force {r.Achieved[0]} N does not track the commanded {lateral} N");
            Assert.IsTrue(math.abs(r.Achieved[1] - AllocWeight) < 0.02f * AllocWeight,
                $"lift {r.Achieved[1]} N does not carry the hull's {AllocWeight} N");
            Assert.IsTrue(math.abs(r.Achieved[2]) < 0.02f * AllocWeight,
                $"drive should be zero while strafing, got {r.Achieved[2]} N");

            for (int k = 3; k < GimbalAllocation.WrenchRows; k++)
                Assert.IsTrue(math.abs(r.Achieved[k]) < 0.02f * AllocTorqueScale,
                    $"torque row {k} should be zero, got {r.Achieved[k]} Nm (scale {AllocTorqueScale} Nm)");
        }

        /// <summary>
        /// JᵀJ is rank 6 of 12 at best, and a dead thruster zeroes three more of its columns, so the
        /// servo and trim weights are the only thing making the allocation Hessian positive definite —
        /// which is also what keeps the solve well posed at a degenerate gimbal pose. Measured at the
        /// converged controls of a live solve, so the Jacobian is the one the QP was actually handed.
        /// </summary>
        [Test]
        public void AllocationHessian_IsPositiveDefinite_WithDeadThruster()
        {
            GimbalSettings settings = GimbalSettings.Default;
            float floor = math.min(settings.servoWeight, settings.trimWeight);

            var desired = new GimbalWrench { Lateral = 1500f, Lift = AllocWeight, Drive = 3000f };
            AllocResult r = RunAllocation(in desired, new float4(1f, 1f, 1f, 0f), 0f);

            Assert.IsTrue(r.Info.status == QPStatus.Optimal,
                $"allocation QP returned {r.Info.status} with a dead thruster");
            Assert.IsTrue(r.EigConverged, "symmetric eigensolve on the allocation Hessian did not converge");

            Assert.IsTrue(r.MinEig > 0f,
                $"min eig(Q) = {r.MinEig} is not positive (max eig {r.MaxEig})");
            Assert.IsTrue(r.MinEig > 0.95f * floor,
                $"min eig(Q) = {r.MinEig} fell below the regularizer floor {floor} (max eig {r.MaxEig})");
        }

        /// <summary>
        /// With the mounts at hull mid-height the rig's pitch moment is -Σ z_i·Fy_i, so only the
        /// front-versus-rear LIFT SPLIT can pitch the hull and forward thrust cannot. Off mid-height
        /// the same demand forces a split of y·drive / halfLength to cancel the coupling, which is what
        /// the front and rear servos used to spend travel on. Both geometries are run, so the claim is
        /// a measured difference rather than an identity restated.
        /// </summary>
        [Test]
        public void Allocation_MidHeightMounts_RemovePitchCouplingFromDrive()
        {
            const float drive = 6000f;
            var desired = new GimbalWrench { Lift = AllocWeight, Drive = drive };

            AllocResult mid = RunAllocation(in desired, new float4(1f), 0f);
            Assert.IsTrue(mid.Info.status == QPStatus.Optimal, $"allocation QP returned {mid.Info.status}");
            Assert.IsTrue(math.abs(mid.Achieved[2] - drive) < 0.02f * AllocWeight,
                $"drive {mid.Achieved[2]} N does not track the commanded {drive} N");
            Assert.IsTrue(math.abs(mid.Achieved[3]) < 0.02f * AllocTorqueScale,
                $"driving pitched the hull by {mid.Achieved[3]} Nm (scale {AllocTorqueScale} Nm)");

            float midSplit = LiftSplit(mid.Controls);
            Assert.IsTrue(math.abs(midSplit) < 0.01f * AllocWeight,
                $"mid-height mounts still split lift front-to-rear by {midSplit} N");
            Assert.IsTrue(PitchServoSplit(mid.Controls) < math.radians(0.5f),
                $"front and rear pitch servos still differ by {math.degrees(PitchServoSplit(mid.Controls))} deg");

            // Same demand with the mounts back under the hull, where the coupling is geometric.
            const float belowY = -0.5f;
            AllocResult below = RunAllocation(in desired, new float4(1f), belowY);
            Assert.IsTrue(below.Info.status == QPStatus.Optimal, $"allocation QP returned {below.Info.status}");

            float expected = belowY * drive / AllocHalfLength;
            float belowSplit = LiftSplit(below.Controls);
            Assert.IsTrue(math.abs(belowSplit - expected) < 0.3f * math.abs(expected),
                $"bottom-mounted rig split lift by {belowSplit} N, geometry demands about {expected} N");
            Assert.IsTrue(PitchServoSplit(below.Controls) > math.radians(5f),
                $"bottom-mounted rig should have to split its pitch servos, got {math.degrees(PitchServoSplit(below.Controls))} deg");
        }

        // ---- allocation test rig ----

        const float AllocMass = 1500f, AllocGravity = 9.81f;
        const float AllocHalfWidth = 2f, AllocHalfLength = 3f;
        const float AllocDt = 1f / 60f;
        const int AllocSteps = 400;

        static float AllocWeight => AllocMass * AllocGravity;
        static float AllocArm => math.max(AllocHalfWidth, AllocHalfLength);
        static float AllocTorqueScale => AllocWeight * AllocArm;

        /// <summary>What <see cref="GimbalAllocationJob"/> settled on, copied out of its buffers.</summary>
        struct AllocResult
        {
            /// <summary>Achieved lateral, lift, drive (N) then pitch, yaw, roll (Nm).</summary>
            public float[] Achieved;

            /// <summary>Converged controls: 4 pitch angles, 4 yaw angles (rad), 4 throttles.</summary>
            public float[] Controls;

            public QPInfo Info;
            public float MinEig, MaxEig;
            public bool EigConverged;
        }

        /// <summary>
        /// Runs the allocation to convergence for one demanded wrench, from a level rig trimmed to
        /// carry the hull. <paramref name="mountY"/> is the mount height above the center of mass.
        /// </summary>
        static AllocResult RunAllocation(in GimbalWrench desired, float4 health, float mountY)
        {
            GimbalSettings settings = GimbalSettings.Default;

            var controls = new NativeArray<float>(GimbalAllocation.ControlCount, Allocator.TempJob);
            var readout = new NativeArray<float>(9, Allocator.TempJob);
            var info = new NativeArray<QPInfo>(1, Allocator.TempJob);

            float trim = math.clamp(AllocWeight / (4f * settings.maxThrust),
                                    settings.minThrust / settings.maxThrust, 1f);
            for (int i = 0; i < 4; i++) { controls[i] = 0f; controls[4 + i] = 0f; controls[8 + i] = trim; }

            var job = new GimbalAllocationJob
            {
                Controls = controls, Out = readout, Info = info,
                Settings = settings, Desired = desired, Health = health,
                MountX = new float4(-AllocHalfWidth, AllocHalfWidth, -AllocHalfWidth, AllocHalfWidth),
                MountY = new float4(mountY),
                MountZ = new float4(AllocHalfLength, AllocHalfLength, -AllocHalfLength, -AllocHalfLength),
                MountArm = AllocArm, Weight = AllocWeight, Dt = AllocDt, Steps = AllocSteps,
            };
            IJobExtensions.RunByRef(ref job);

            var r = new AllocResult
            {
                Achieved = new float[GimbalAllocation.WrenchRows],
                Controls = new float[GimbalAllocation.ControlCount],
                Info = info[0],
                MinEig = readout[6], MaxEig = readout[7], EigConverged = readout[8] == 1f,
            };
            for (int i = 0; i < GimbalAllocation.WrenchRows; i++) r.Achieved[i] = readout[i];
            for (int i = 0; i < GimbalAllocation.ControlCount; i++) r.Controls[i] = controls[i];

            controls.Dispose(); readout.Dispose(); info.Dispose();
            return r;
        }

        /// <summary>How far the front pair's mean pitch servo angle sits from the rear pair's, radians.</summary>
        static float PitchServoSplit(float[] z)
            => math.abs(0.5f * (z[0] + z[1]) - 0.5f * (z[2] + z[3]));

        /// <summary>Front-pair lift minus rear-pair lift for a converged control vector, newtons.</summary>
        static float LiftSplit(float[] z)
        {
            float maxThrust = GimbalSettings.Default.maxThrust;
            float front = 0f, rear = 0f;
            for (int i = 0; i < 4; i++)
            {
                float lift = z[8 + i] * maxThrust * GimbalAllocation.ForceDirection(z[i], z[4 + i]).y;
                if (i < 2) front += lift; else rear += lift;
            }
            return front - rear;
        }

        static float StateNorm(NativeArray<float> x, int n)
        {
            float s = 0f;
            for (int i = 0; i < n; i++) s += x[i] * x[i];
            return math.sqrt(s);
        }
    }

    /// <summary>
    /// Drives <see cref="GimbalAllocation"/> on its own — no hover loop, no driver — for
    /// <see cref="Steps"/> fixed steps against one constant demanded wrench, then reports the wrench
    /// it settled on and the spectrum of the allocation Hessian at those controls. Controls carry the
    /// starting rig in and the converged one out. Out is [lateral, lift, drive, pitch, yaw, roll,
    /// min eig(Q), max eig(Q), 1 if the eigensolve converged].
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct GimbalAllocationJob : IJob
    {
        public NativeArray<float> Controls;
        public NativeArray<float> Out;
        public NativeArray<QPInfo> Info;

        public GimbalSettings Settings;
        public GimbalWrench Desired;
        public float4 Health, MountX, MountY, MountZ;
        public float MountArm, Weight, Dt;

        /// <summary>Fixed steps to run. At least 1: the rig is rebuilt around the controls each step.</summary>
        public int Steps;

        public void Execute()
        {
            var z = new floatN(Controls);   // view, no copy

            GimbalRig rig = GimbalAllocation.BuildRig(in Settings, MountX, MountY, MountZ, Health,
                                                     in z, Dt, Weight, MountArm);
            for (int s = 0; s < Steps; s++)
            {
                Info[0] = GimbalAllocation.Solve(in rig, in Desired, ref z, 0);
                rig = GimbalAllocation.BuildRig(in Settings, MountX, MountY, MountZ, Health,
                                                in z, Dt, Weight, MountArm);
            }

            GimbalWrench w = GimbalAllocation.Wrench(in rig, in z);
            Out[0] = w.Lateral; Out[1] = w.Lift; Out[2] = w.Drive;
            Out[3] = w.Pitch; Out[4] = w.Yaw; Out[5] = w.Roll;

            const int n = GimbalAllocation.ControlCount;
            var J = new floatMxN(GimbalAllocation.WrenchRows, n, Allocator.Temp, true);
            GimbalAllocation.ScaledJacobian(in rig, in z, ref J);

            var Q = new floatMxN(n, n, Allocator.Temp, true);
            GimbalAllocation.Hessian(in rig, in J, ref Q);

            // valuesSymmetricInPlace DESTROYS Q and sorts descending, so the smallest is last.
            var ev = new floatN(n, Allocator.Temp, true);
            EigenInfo eig = Eigen.valuesSymmetricInPlace(ref Q, ref ev);
            Out[6] = ev[n - 1];
            Out[7] = ev[0];
            Out[8] = eig ? 1f : 0f;

            ev.Dispose(); Q.Dispose(); J.Dispose();
        }
    }

    /// <summary>
    /// Sweeps <see cref="TerrainField.Height"/> over the whole field and reduces it to the numbers the
    /// terrain's shape contract is stated in. Out is
    /// [max |h| on the apron, max |h| anywhere, max gradient off the features, max gradient on the
    /// escarpment, max gradient on the wall, wall prominence over its own flanks].
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct TerrainSampleJob : IJob
    {
        public NativeArray<float> Out;

        /// <summary>Sample spacing, metres.</summary>
        public float Step;

        /// <summary>Central-difference half-step for the gradient, metres.</summary>
        public float Delta;

        public void Execute()
        {
            float apronMax = 0f, absMax = 0f;
            float gentleSlope = 0f, escarpSlope = 0f, wallSlope = 0f;

            const float half = TerrainField.Size * 0.5f;
            int n = (int)(TerrainField.Size / Step);

            // Feature bands are widened by a metre so the gradient stencil never straddles a boundary
            // and reports a feature's slope against the gentle budget.
            const float wallBandX = TerrainField.WallHalfLength + TerrainField.WallFade + 1f;
            const float wallBandZ = TerrainField.WallHalfThick + TerrainField.WallFade + 1f;

            for (int j = 0; j <= n; j++)
            {
                float z = -half + j * Step;
                for (int i = 0; i <= n; i++)
                {
                    float x = -half + i * Step;

                    float h = TerrainField.Height(x, z);
                    absMax = math.max(absMax, math.abs(h));
                    if (math.sqrt(x * x + z * z) <= TerrainField.ApronRadius)
                        apronMax = math.max(apronMax, math.abs(h));

                    float gx = (TerrainField.Height(x + Delta, z) - TerrainField.Height(x - Delta, z)) / (2f * Delta);
                    float gz = (TerrainField.Height(x, z + Delta) - TerrainField.Height(x, z - Delta)) / (2f * Delta);
                    float slope = math.sqrt(gx * gx + gz * gz);

                    bool onWall = math.abs(x - TerrainField.WallCenterX) <= wallBandX
                               && math.abs(z - TerrainField.WallCenterZ) <= wallBandZ;
                    bool onEscarp = z >= TerrainField.EscarpStartZ - 1f && z <= TerrainField.EscarpEndZ + 1f;

                    if (onWall) wallSlope = math.max(wallSlope, slope);
                    else if (onEscarp) escarpSlope = math.max(escarpSlope, slope);
                    else gentleSlope = math.max(gentleSlope, slope);
                }
            }

            // Measured against the MEAN of the two flanks, so any linear hill gradient running through
            // the wall cancels and what is left is the wall itself.
            float prominence = float.MaxValue;
            for (int k = -3; k <= 3; k++)
            {
                float x = TerrainField.WallCenterX + k * 4f;
                float crest = TerrainField.Height(x, TerrainField.WallCenterZ);
                float flank = 0.5f * (TerrainField.Height(x, TerrainField.WallCenterZ - 8f)
                                    + TerrainField.Height(x, TerrainField.WallCenterZ + 8f));
                prominence = math.min(prominence, crest - flank);
            }

            Out[0] = apronMax; Out[1] = absMax;
            Out[2] = gentleSlope; Out[3] = escarpSlope; Out[4] = wallSlope;
            Out[5] = prominence;
        }
    }
}
