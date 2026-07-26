using BULA;
using BULA.Control;

using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebraDemos;

namespace LinearAlgebraDemos.Tests
{
    /// <summary>
    /// The three things demo 15's MPC does that the LQR cascade it replaced structurally could not.
    /// Each is written as a DISCRIMINATING PAIR — two runs differing in exactly one input — because
    /// every one of these claims is about a command that appears when nothing in the regulated error
    /// has changed, and only the pair can show that.
    ///
    /// 1. Terrain preview. Ride-height error is EXACTLY ZERO in both runs of
    ///    <see cref="Preview_CommandsClimb_BeforeAnyClearanceErrorExists"/>; the only difference is
    ///    the fitted ground normal. A controller on clearance error commands the same thing in both,
    ///    by construction, so a nonzero difference is the anticipation itself.
    /// 2. Anti-collision. The velocity reference is full-ahead in both runs of
    ///    <see cref="Proximity_BrakesForAWall_ThatTheReferenceKnowsNothingAbout"/>; only the ranger
    ///    reading moves. Nothing in the cost function mentions the wall.
    /// 3. Input bounds. The commanded acceleration stays inside the rig's authority even when the
    ///    reference asks for far more than it can deliver.
    ///
    /// Nothing here touches Physics, Rigidbody or raycasts: the step job is driven directly against a
    /// synthetic estimate.
    /// </summary>
    public class HoverTankMPCControlTests
    {
        const float Dt = 1f / 60f;
        const int Horizon = 15;
        const float Mass = 1500f, Gravity = 9.81f;
        const float RollInertia = 2100f, PitchInertia = 4600f, YawInertia = 6500f;
        const float HalfWidth = 2f, HalfLength = 3f;
        const float RideHeight = 2f, RayLength = 8f, Margin = 1.5f;

        /// <summary>
        /// The demo's own exact-penalty weight, and the reason it is not the library default. A metre
        /// of predicted intrusion has to out-price the velocity tracking driving the tank at the
        /// obstacle: slowing enough to give up a metre of displacement inside the horizon costs on the
        /// order of qVel * (2 * vErr * dV) * Horizon, a few thousand at these weights. The library's
        /// 1e3 default assumes cost matrices at O(1); at 1e3 the tank accelerates into a wall at full
        /// throttle and merely REPORTS the violation, which is what
        /// <see cref="Proximity_BrakesForAWall_ThatTheReferenceKnowsNothingAbout"/> first caught.
        /// </summary>
        const float Penalty = 1e5f;

        /// <summary>Steps each case is held at a FIXED estimate for, so the reported command is the
        /// converged one rather than a first-solve transient.</summary>
        const int SettleSteps = 40;

        /// <summary>
        /// Rising ground ahead must produce a CLIMB command while the tank is exactly on its commanded
        /// ride height. The level run is the control: same speed, same attitude, same zero clearance
        /// error, flat normal. Any vertical command in the sloped run therefore came from the preview
        /// and from nothing else — which is the whole claim, since an LQR reading clearance error sees
        /// an identical (zero) input in both.
        ///
        /// Terrain rising at gradient s along +Z has surface normal proportional to (0, 1, -s), which
        /// is where the sloped run's normal comes from.
        /// </summary>
        [Test]
        public void Preview_CommandsClimb_BeforeAnyClearanceErrorExists()
        {
            const float speed = 8f, gradient = 0.25f;   // ~14 degrees

            float3 rising = math.normalize(new float3(0f, 1f, -gradient));
            var sloped = RunFixed(fwdSpeed: speed, groundNormal: rising, groundValid: true,
                                  clearance: RideHeight, proxFwd: RayLength, driveInput: 0f,
                                  out float slopedRise, out _, out MPCInfo slopedInfo);
            var level = RunFixed(fwdSpeed: speed, groundNormal: new float3(0f, 1f, 0f), groundValid: true,
                                 clearance: RideHeight, proxFwd: RayLength, driveInput: 0f,
                                 out float levelRise, out _, out MPCInfo levelInfo);

            Assert.IsTrue(slopedInfo.status != MPCStatus.Fallback, $"sloped run fell back: {slopedInfo.status}");
            Assert.IsTrue(levelInfo.status != MPCStatus.Fallback, $"level run fell back: {levelInfo.status}");

            // The preview itself: at 8 m/s over a 0.25 s horizon the ground rises 0.5 m.
            float expectedRise = gradient * speed * (Horizon * Dt);
            Assert.IsTrue(math.abs(slopedRise - expectedRise) < 0.05f * expectedRise,
                $"predicted rise at the horizon end {slopedRise} m, expected {expectedRise} m");
            Assert.IsTrue(math.abs(levelRise) < 1e-4f,
                $"level ground must predict no rise, got {levelRise} m");

            // The command that follows from it. Climbing at gradient*speed = 2 m/s from rest inside the
            // horizon needs a real acceleration, so this is not a rounding-level difference.
            Assert.IsTrue(level[HoverTankMPCDemo.AVert] < 0.05f,
                $"level ground at zero clearance error must not command a climb, got {level[HoverTankMPCDemo.AVert]} m/s^2");
            Assert.IsTrue(sloped[HoverTankMPCDemo.AVert] > 0.5f,
                $"rising ground must command a climb before the clearance error appears, got {sloped[HoverTankMPCDemo.AVert]} m/s^2");

            // Ground FALLING away must command the opposite sign, or the test would also pass on a
            // controller that simply climbs whenever the normal is not vertical.
            float3 falling = math.normalize(new float3(0f, 1f, gradient));
            var down = RunFixed(fwdSpeed: speed, groundNormal: falling, groundValid: true,
                                clearance: RideHeight, proxFwd: RayLength, driveInput: 0f,
                                out float downRise, out _, out _);
            Assert.IsTrue(downRise < 0f, $"falling ground must predict a drop, got {downRise} m");
            Assert.IsTrue(down[HoverTankMPCDemo.AVert] < -0.5f,
                $"falling ground must command a descent, got {down[HoverTankMPCDemo.AVert]} m/s^2");

            sloped.Dispose(); level.Dispose(); down.Dispose();
        }

        /// <summary>
        /// A wall ahead must produce a BRAKING command while the driver is holding full throttle. The
        /// clear run is the control: identical velocity reference, identical state, only the forward
        /// ranger moves. The cost function never mentions the wall — it enters purely as a soft row on
        /// predicted displacement, moved by <c>MPC.setSoftBound</c>.
        ///
        /// At 10 m/s the tank covers 2.5 m inside the horizon against 0.5 m of room, and the input box
        /// cannot stop it in time, so the row is genuinely unavoidable and the reported slack is
        /// positive. That is what makes this a test of a BINDING constraint rather than a dormant one.
        /// </summary>
        [Test]
        public void Proximity_BrakesForAWall_ThatTheReferenceKnowsNothingAbout()
        {
            const float speed = 10f;

            var clear = RunFixed(fwdSpeed: speed, groundNormal: new float3(0f, 1f, 0f), groundValid: true,
                                 clearance: RideHeight, proxFwd: RayLength, driveInput: 1f,
                                 out _, out float clearRoom, out MPCInfo clearInfo);
            var walled = RunFixed(fwdSpeed: speed, groundNormal: new float3(0f, 1f, 0f), groundValid: true,
                                  clearance: RideHeight, proxFwd: 2f, driveInput: 1f,
                                  out _, out float walledRoom, out MPCInfo walledInfo);

            Assert.IsTrue(clearInfo.status != MPCStatus.Fallback, $"clear run fell back: {clearInfo.status}");
            Assert.IsTrue(walledInfo.status != MPCStatus.Fallback, $"walled run fell back: {walledInfo.status}");

            Assert.IsTrue(math.abs(clearRoom - (RayLength - Margin)) < 1e-4f,
                $"an empty ranger should leave {RayLength - Margin} m of room, got {clearRoom} m");
            Assert.IsTrue(math.abs(walledRoom - 0.5f) < 1e-4f,
                $"a wall at 2 m behind a {Margin} m margin should leave 0.5 m of room, got {walledRoom} m");

            // The constraint is unavoidable at this closing speed, so it must be REPORTED, not silently
            // dropped: a soft row that never shows slack here would mean it is not in the QP at all.
            Assert.IsTrue(walledInfo.maxSlackViolation > 0.1,
                $"the wall did not bind: slack {walledInfo.maxSlackViolation} m");
            Assert.IsTrue(clearInfo.maxSlackViolation < 1e-3,
                $"open ground must not violate anything, slack {clearInfo.maxSlackViolation} m");

            Assert.IsTrue(clear[HoverTankMPCDemo.AFwd] > 0f,
                $"full throttle on open ground should accelerate, got {clear[HoverTankMPCDemo.AFwd]} m/s^2");
            Assert.IsTrue(walled[HoverTankMPCDemo.AFwd] < 0f,
                $"full throttle at a wall should brake, got {walled[HoverTankMPCDemo.AFwd]} m/s^2");

            clear.Dispose(); walled.Dispose();
        }

        /// <summary>
        /// The input box is a hard bound, not a suggestion: a velocity reference far beyond the rig's
        /// authority must still produce a command inside it. This is the capability the cascade lacked
        /// — it would have demanded a wrench and left the allocation to clip it, with the controller
        /// none the wiser.
        /// </summary>
        [Test]
        public void InputBounds_HoldUnderAnImpossibleReference()
        {
            var floor = new floatN(HoverTankMPCDemo.InputCount, Allocator.TempJob, true);
            var ceil = new floatN(HoverTankMPCDemo.InputCount, Allocator.TempJob, true);
            for (int i = 0; i < HoverTankMPCDemo.InputCount; i++) { floor[i] = -3f; ceil[i] = 3f; }

            var u = RunFixed(fwdSpeed: 0f, groundNormal: new float3(0f, 1f, 0f), groundValid: true,
                             clearance: RideHeight, proxFwd: RayLength, driveInput: 1f,
                             out _, out _, out MPCInfo info, in floor, in ceil, maxFwdSpeed: 400f);

            Assert.IsTrue(info.status != MPCStatus.Fallback, $"run fell back: {info.status}");
            for (int i = 0; i < HoverTankMPCDemo.InputCount; i++)
                Assert.IsTrue(u[i] >= -3f - 1e-4f && u[i] <= 3f + 1e-4f,
                    $"input {i} left its box at {u[i]}, bounds [-3, 3]");

            // Non-vacuity: an unreachable reference must actually push the forward channel to its
            // limit, or the box was never tested.
            Assert.IsTrue(u[HoverTankMPCDemo.AFwd] > 3f - 1e-2f,
                $"a 400 m/s reference should saturate the forward input, got {u[HoverTankMPCDemo.AFwd]}");

            u.Dispose(); floor.Dispose(); ceil.Dispose();
        }

        // ---------------------------------------------------------------------------------------

        static NativeArray<float> RunFixed(float fwdSpeed, float3 groundNormal, bool groundValid,
                                           float clearance, float proxFwd, float driveInput,
                                           out float riseAtHorizon, out float tightestRoom, out MPCInfo info)
        {
            var lo = new floatN(HoverTankMPCDemo.InputCount, Allocator.TempJob, true);
            var hi = new floatN(HoverTankMPCDemo.InputCount, Allocator.TempJob, true);
            for (int i = 0; i < HoverTankMPCDemo.InputCount; i++) { lo[i] = -12f; hi[i] = 12f; }
            lo[HoverTankMPCDemo.AVert] = -Gravity;
            var u = RunFixed(fwdSpeed, groundNormal, groundValid, clearance, proxFwd, driveInput,
                             out riseAtHorizon, out tightestRoom, out info, in lo, in hi, 14f);
            lo.Dispose(); hi.Dispose();
            return u;
        }

        /// <summary>
        /// Runs the demo's own step job for <see cref="SettleSteps"/> steps against an estimate that
        /// never changes, and returns the converged first input. Holding the estimate fixed is what
        /// makes the comparison between two cases exact: the ONLY difference between any pair here is
        /// the argument that differs.
        /// </summary>
        static NativeArray<float> RunFixed(float fwdSpeed, float3 groundNormal, bool groundValid,
                                           float clearance, float proxFwd, float driveInput,
                                           out float riseAtHorizon, out float tightestRoom, out MPCInfo info,
                                           in floatN uLo, in floatN uHi, float maxFwdSpeed)
        {
            const int n = HoverTankMPCDemo.StateCount, m = HoverTankMPCDemo.InputCount;
            GimbalSettings settings = GimbalSettings.Default;

            HoverTankMPCDemo.BuildMpcModel(Dt, 0.05f, 12f, 120f, 14f, 90f, 8f, 10f, 0.02f, 0.4f,
                                           Allocator.TempJob, out var A, out var B, out var Q, out var R);

            var C = new floatMxN(HoverTankMPCDemo.SoftRows, n, Allocator.TempJob);
            C[0, HoverTankMPCDemo.SFwd] = 1f; C[1, HoverTankMPCDemo.SFwd] = -1f;
            C[2, HoverTankMPCDemo.SLat] = 1f; C[3, HoverTankMPCDemo.SLat] = -1f;
            var d = new floatN(HoverTankMPCDemo.SoftRows, Allocator.TempJob, true);
            for (int i = 0; i < HoverTankMPCDemo.SoftRows; i++) d[i] = RayLength;

            var mpc = new floatMPCState(n, m, Horizon, Allocator.TempJob,
                                        in A, in B, in Q, in R, in uLo, in uHi, in C, in d, Penalty, 1f);
            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); C.Dispose(); d.Dispose();

            var mpcX0 = new NativeArray<float>(n, Allocator.TempJob);
            var mpcRef = new NativeArray<float>(Horizon * n, Allocator.TempJob);
            var mpcU0 = new NativeArray<float>(m, Allocator.TempJob);
            var mpcSoft = new NativeArray<float>(HoverTankMPCDemo.SoftRows, Allocator.TempJob);
            var mpcOut = new NativeArray<MPCInfo>(1, Allocator.TempJob);
            var previewOut = new NativeArray<float>(4, Allocator.TempJob);

            var prox = new NativeArray<float>(ProximityRig.Rays, Allocator.TempJob);
            prox[0] = proxFwd;
            for (int i = 1; i < ProximityRig.Rays; i++) prox[i] = RayLength;

            var controls = new NativeArray<float>(GimbalAllocation.ControlCount, Allocator.TempJob);
            var allocOut = new NativeArray<QPInfo>(1, Allocator.TempJob);
            var wrenchOut = new NativeArray<float>(2 * GimbalAllocation.WrenchRows, Allocator.TempJob);
            var groundOut = new NativeArray<float>(GimbalAllocation.Thrusters, Allocator.TempJob);

            float trim = math.clamp(Mass * Gravity / (4f * settings.maxThrust),
                                    settings.minThrust / settings.maxThrust, 1f);
            for (int i = 0; i < 4; i++) { controls[i] = 0f; controls[4 + i] = 0f; controls[8 + i] = trim; }

            var job = new HoverTankMPCStepJob
            {
                Mpc = mpc,
                MpcX0 = mpcX0, MpcRef = mpcRef, MpcU0 = mpcU0, MpcSoft = mpcSoft,
                MpcOut = mpcOut, PreviewOut = previewOut, Horizon = Horizon,
                Mass = Mass, RollInertia = RollInertia, PitchInertia = PitchInertia,
                YawInertia = YawInertia, Gravity = Gravity,
                Dt = Dt,

                Rpy = float3.zero,
                GroundNormal = groundNormal,
                VelWorld = float3.zero,
                ForwardSpeed = fwdSpeed, LateralSpeed = 0f, YawRate = 0f, RollRate = 0f, PitchRate = 0f,
                Clearance = clearance, TiltCos = 1f, TargetRideHeight = RideHeight,
                GroundValid = groundValid,

                ProxSensed = prox, CollisionMargin = Margin,

                Controls = controls, AllocOut = allocOut, WrenchOut = wrenchOut,
                Settings = settings, Health = new float4(1f),
                MountX = new float4(-HalfWidth, HalfWidth, -HalfWidth, HalfWidth),
                MountY = float4.zero,
                MountZ = new float4(HalfLength, HalfLength, -HalfLength, -HalfLength),
                MountArm = math.max(HalfWidth, HalfLength),
                DriveInput = driveInput, StrafeInput = 0f, SteerInput = 0f, BrakeInput = false,
                MaxFwdSpeed = maxFwdSpeed, MaxLatSpeed = 9f, MaxYawRate = 1.4f,

                NozzleHeights = new float4(2f), NozzleRadius = 0f, GroundOut = groundOut,
            };

            for (int s = 0; s < SettleSteps; s++) IJobExtensions.RunByRef(ref job);

            riseAtHorizon = previewOut[0];
            tightestRoom = previewOut[3];
            info = mpcOut[0];

            var result = new NativeArray<float>(m, Allocator.Persistent);
            result.CopyFrom(mpcU0);

            job.Mpc.Dispose();
            mpcX0.Dispose(); mpcRef.Dispose(); mpcU0.Dispose(); mpcSoft.Dispose();
            mpcOut.Dispose(); previewOut.Dispose(); prox.Dispose();
            controls.Dispose(); allocOut.Dispose(); wrenchOut.Dispose(); groundOut.Dispose();
            return result;
        }
    }
}
