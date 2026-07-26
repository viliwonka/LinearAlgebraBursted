using BULA;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

namespace LinearAlgebraDemos.Tests
{
    /// <summary>
    /// Headless tests for the SENSING AND ESTIMATION half of <see cref="HoverTankMPCDemo"/>:
    /// <see cref="Attitude"/>'s algebra, the analytic Jacobians of <see cref="TankInsModel"/> and the
    /// two measurement models, <see cref="GroundFit.Plane"/>, and <see cref="TankEstimatorJob"/> driven
    /// end to end against an analytic trajectory. Nothing here touches Physics, Rigidbody or raycasts —
    /// every range is synthesized from a known plane.
    ///
    /// THE JACOBIAN COMPARISON IS THE LOAD-BEARING TEST. A wrong analytic Jacobian leaves every
    /// convergence scenario below intact to three decimals — the filter still tracks, it just stops
    /// being an optimal estimator — so the numeric-vs-analytic check is the only thing that sees it.
    /// It therefore carries its own DISCRIMINATION PROBES: deliberately wrong Jacobians measured
    /// against the same numeric reference, which must exceed the tolerance by a wide margin. Without
    /// them a comparison that had quietly become incapable of failing would still read as coverage.
    /// </summary>
    public class HoverTankMPCEstimatorTests
    {
        // ================================ 1. Attitude algebra =======================================

        /// <summary>
        /// <see cref="Attitude.Matrix"/> and <see cref="Attitude.FromBasis"/> are each other's inverse,
        /// and the matrix is a rotation. FromBasis is how truth is read for the error plot, so a
        /// disagreement between the two would show up as estimator error that is not there.
        ///
        /// Pitch is drawn clear of ±90°, where the three-parameter attitude is singular by construction
        /// and roll/yaw stop being separately defined.
        /// </summary>
        [Test]
        public void Attitude_Matrix_RoundTripsThroughFromBasis_AndIsOrthonormal()
        {
            var rng = new Random(0x5E45E1u);
            float maxRoundTrip = 0f, maxOrtho = 0f, worstPitch = 0f;

            for (int k = 0; k < 200; k++)
            {
                // Roll and yaw turn freely; ±3.0 rad keeps atan2's branch cut out of the comparison.
                float3 rpy = new float3(rng.NextFloat(-3f, 3f),
                                        rng.NextFloat(-1.35f, 1.35f),
                                        rng.NextFloat(-3f, 3f));

                float3x3 R = Attitude.Matrix(rpy);
                float3 back = Attitude.FromBasis(R.c0, R.c1, R.c2);

                float err = math.cmax(math.abs(back - rpy));
                if (err > maxRoundTrip) { maxRoundTrip = err; worstPitch = rpy.y; }

                float3x3 rtr = math.mul(math.transpose(R), R);
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        maxOrtho = math.max(maxOrtho, math.abs(rtr[j][i] - (i == j ? 1f : 0f)));
            }

            Assert.IsTrue(maxRoundTrip < 1e-4f,
                $"Matrix -> FromBasis round trip is off by {maxRoundTrip} rad (worst at pitch {math.degrees(worstPitch)} deg)");
            Assert.IsTrue(maxOrtho < 1e-5f,
                $"Attitude.Matrix is not orthonormal: max |RᵀR - I| = {maxOrtho}");
        }

        /// <summary>
        /// <see cref="Attitude.RateMatrix"/> and <see cref="Attitude.Matrix"/> describe the same
        /// kinematics: stepping the Euler angles by their own rates must move the rotation the way a
        /// body rate does, R(t+h) = R·(I + [w]x·h). This is the cheapest way to prove the two agree,
        /// and the Euler-rate map is what the filter's attitude propagation and its Jacobian are both
        /// built on.
        ///
        /// Pitch is held under 1 rad: the O(h²) term of the comparison grows with the Euler-rate
        /// magnitude, which is what diverges at the pole.
        /// </summary>
        [Test]
        public void Attitude_RateMatrix_AgreesWithMatrixDerivative()
        {
            const float h = 1e-4f;
            var rng = new Random(0x13579BDu);
            float maxDev = 0f;

            for (int k = 0; k < 200; k++)
            {
                float3 rpy = new float3(rng.NextFloat(-3f, 3f),
                                        rng.NextFloat(-1f, 1f),
                                        rng.NextFloat(-3f, 3f));
                float3 w = rng.NextFloat3(new float3(-1f), new float3(1f));

                float3x3 R = Attitude.Matrix(rpy);
                float3x3 stepped = Attitude.Matrix(rpy + Attitude.Rates(rpy, w) * h);

                // Column i of R·(I + [w]x·h) is R·(e_i + h·(w x e_i)) — the cross product spares the
                // test its own skew-matrix sign convention.
                float3 ex = new float3(1f, 0f, 0f), ey = new float3(0f, 1f, 0f), ez = new float3(0f, 0f, 1f);
                float3x3 expect = new float3x3(
                    math.mul(R, ex + h * math.cross(w, ex)),
                    math.mul(R, ey + h * math.cross(w, ey)),
                    math.mul(R, ez + h * math.cross(w, ez)));

                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        maxDev = math.max(maxDev, math.abs(stepped[j][i] - expect[j][i]));
            }

            Assert.IsTrue(maxDev < 1e-6f,
                $"stepping the Euler angles by RateMatrix·w disagrees with R·(I + [w]x·h) by {maxDev} at h = {h}");
        }

        // ================================ 2. Jacobians ==============================================

        /// <summary>
        /// <see cref="TankInsModel.JacobianF"/> against central differences of its own
        /// <see cref="TankInsModel.F"/>, at four states with roll, pitch and yaw all well away from
        /// zero — both faults this model has actually carried vanish at roll = 0, which is exactly
        /// where a level hover trajectory lives.
        ///
        /// Then the two discrimination probes. Neither is an assertion about the model: they are
        /// assertions about the COMPARISON, that a Jacobian off by a factor of two in dt, or with the
        /// attitude coupling sign-flipped, would be caught by it with two orders of magnitude to spare.
        /// </summary>
        [Test]
        public void InsModel_AnalyticJacobianF_MatchesNumeric_AndProbesBite()
        {
            var stats = new NativeArray<float>(4, Allocator.TempJob);
            var job = new TankJacobianJob
            {
                Which = TankJacobianJob.Kind.Transition,
                Dt = 1f / 60f, Gravity = 9.81f, Eps = 1e-2f, Out = stats,
            };
            IJobExtensions.RunByRef(ref job);

            float maxErr = stats[0], probeDt = stats[1], probeAtt = stats[2], states = stats[3];
            stats.Dispose();

            Assert.IsTrue(states == 4f, $"the Jacobian job tested {states} states, expected 4");
            Assert.IsTrue(maxErr < 2e-3f,
                $"analytic JacobianF differs from central differences by {maxErr} (eps = 1e-2)");

            Assert.IsTrue(probeDt > 0.05f,
                $"a JacobianF built at twice the model's dt deviates by only {probeDt} — the comparison cannot see a wrong Jacobian");
            Assert.IsTrue(probeAtt > 0.05f,
                $"negating the attitude coupling deviates by only {probeAtt} — the comparison cannot see a wrong Jacobian");
        }

        /// <summary>
        /// <see cref="TankVectorMeasurement.JacobianH"/> (the gravity reference and the magnetometer,
        /// which share a shape) and <see cref="TankPositionMeasurement.JacobianH"/> (the beacon, an
        /// exact identity block) against central differences, with the same kind of probes: an analytic
        /// Jacobian taken at the WRONG reference direction, and one with the attitude block negated.
        /// </summary>
        [Test]
        public void Measurements_AnalyticJacobianH_MatchNumeric_AndProbesBite()
        {
            var stats = new NativeArray<float>(5, Allocator.TempJob);
            var job = new TankJacobianJob
            {
                Which = TankJacobianJob.Kind.Measurement,
                Dt = 1f / 60f, Gravity = 9.81f, Eps = 1e-2f, Out = stats,
            };
            IJobExtensions.RunByRef(ref job);

            float maxVec = stats[0], maxPos = stats[1], probeRef = stats[2], probeNeg = stats[3];
            float states = stats[4];
            stats.Dispose();

            Assert.IsTrue(states == 4f, $"the Jacobian job tested {states} states, expected 4");
            Assert.IsTrue(maxVec < 2e-3f,
                $"analytic direction-measurement JacobianH differs from central differences by {maxVec}");
            Assert.IsTrue(maxPos < 2e-3f,
                $"analytic beacon JacobianH differs from central differences by {maxPos}");

            Assert.IsTrue(probeRef > 0.05f,
                $"a JacobianH taken at the wrong reference direction deviates by only {probeRef} — the comparison cannot see a wrong Jacobian");
            Assert.IsTrue(probeNeg > 0.05f,
                $"negating the attitude block deviates by only {probeNeg} — the comparison cannot see a wrong Jacobian");
        }

        // ================================ 3. ground plane fit =======================================

        /// <summary>
        /// Level hull over level ground: the fit has to come back with hull +y and the exact
        /// perpendicular distance from the lidar mount. Ranges are synthesized from the plane, so this
        /// is the noise-free closed form.
        /// </summary>
        [Test]
        public void GroundFit_LevelHull_RecoversNormalAndClearance()
        {
            var dirs = new NativeArray<float3>(LidarGrid.Rays, Allocator.Temp);
            var ranges = new NativeArray<float>(LidarGrid.Rays, Allocator.Temp);
            LidarGrid.Directions(dirs);

            float3x3 R = Attitude.Matrix(float3.zero);
            SynthRanges(dirs, LidarOrigin, R, new float3(0f, 3f, 0f),
                        float3.zero, new float3(0f, 1f, 0f), 12f, ranges,
                        out float3 expectNormal, out float expectClearance);

            GroundPlane g = GroundFit.Plane(ranges, dirs, LidarOrigin);

            Assert.IsTrue(g.Valid, "the fit was refused on 25 exact returns");
            Assert.IsTrue(g.Returns == LidarGrid.Rays, $"{g.Returns} of {LidarGrid.Rays} beams returned");
            AssertPlane(g, expectNormal, expectClearance, 0.05f, 1e-3f, "level hull, level ground");

            dirs.Dispose(); ranges.Dispose();
        }

        /// <summary>
        /// The claim the demo's whole slope-versus-tilt separation rests on: with the hull rolled,
        /// pitched and yawed over sloping ground, the fitted normal comes back in HULL axes as
        /// Rᵀ·n_world. Ranges are synthesized in the WORLD frame — world plane, world beam directions —
        /// so recovering Rᵀ·n_world is a genuine round trip through the pose and not a restatement of
        /// the input.
        /// </summary>
        [Test]
        public void GroundFit_TiltedHull_ReturnsGroundNormalInHullAxes()
        {
            var dirs = new NativeArray<float3>(LidarGrid.Rays, Allocator.Temp);
            var ranges = new NativeArray<float>(LidarGrid.Rays, Allocator.Temp);
            LidarGrid.Directions(dirs);

            float3 rpy = new float3(0.28f, -0.19f, 1.1f);           // 16 deg roll, -11 deg pitch, 63 deg yaw
            float3x3 R = Attitude.Matrix(rpy);
            float3 nWorld = math.normalize(new float3(0.2f, 1f, -0.15f));   // 14 deg terrain slope

            // The range cap is opened wide on purpose: at 19 deg of hull tilt the outermost beams of a
            // ±28° fan graze the slope and run tens of metres, and this case is about the FIT, not the
            // range budget. Beams that find nothing are covered by the contaminated case below.
            SynthRanges(dirs, LidarOrigin, R, new float3(7f, 4.2f, -3f),
                        new float3(7f, 0f, -3f), nWorld, 60f, ranges,
                        out float3 expectNormal, out float expectClearance);

            GroundPlane g = GroundFit.Plane(ranges, dirs, LidarOrigin);

            Assert.IsTrue(g.Valid, "the fit was refused on 25 exact returns from a tilted hull");
            Assert.IsTrue(g.Returns == LidarGrid.Rays, $"{g.Returns} of {LidarGrid.Rays} beams returned");

            // The fit is pose-free, so its answer must be the terrain normal seen from the hull...
            AssertPlane(g, expectNormal, expectClearance, 0.05f, 1e-3f, "tilted hull, sloping ground");

            // ...and rotating it back out by the hull's own attitude must land on the world slope,
            // which is what TankEstimatorJob reports as GroundNormal.
            float worldDeg = AngleDeg(math.mul(R, g.Normal), nWorld);
            Assert.IsTrue(worldDeg < 0.05f,
                $"the fitted normal rotated to world is {worldDeg} deg off the terrain slope");

            // And the hull normal must NOT be world up: otherwise the test would pass on a fit that
            // ignored the terrain entirely.
            float tiltDeg = AngleDeg(expectNormal, new float3(0f, 1f, 0f));
            Assert.IsTrue(tiltDeg > 5f,
                $"test setup: the ground normal is only {tiltDeg} deg off hull up, so this case discriminates nothing");

            dirs.Dispose(); ranges.Dispose();
        }

        /// <summary>
        /// The case the consensus step exists for: four beams find nothing and a five-beam cluster
        /// ranges a surface 3 m lower, so a fifth of the returns belong to a different plane. A single
        /// reweighted fit can only walk downhill from a start the contamination has already moved, so
        /// the naive fit over the same points is measured too — a probe on the test, proving the
        /// contamination really is there to be rejected.
        /// </summary>
        [Test]
        public void GroundFit_SurvivesMissesAndCliffCluster()
        {
            var dirs = new NativeArray<float3>(LidarGrid.Rays, Allocator.Temp);
            var ranges = new NativeArray<float>(LidarGrid.Rays, Allocator.Temp);
            var cliff = new NativeArray<float>(LidarGrid.Rays, Allocator.Temp);
            LidarGrid.Directions(dirs);

            float3 rpy = new float3(0.12f, -0.08f, -0.6f);
            float3x3 R = Attitude.Matrix(rpy);
            float3 nWorld = math.normalize(new float3(-0.12f, 1f, 0.08f));
            float3 hull = new float3(-2f, 4f, 5f), p0 = new float3(-2f, 0f, 5f);

            SynthRanges(dirs, LidarOrigin, R, hull, p0, nWorld, 20f, ranges,
                        out float3 expectNormal, out float expectClearance);

            // The cliff: the same ground dropped 3 m along its own normal.
            SynthRanges(dirs, LidarOrigin, R, hull, p0 - 3f * nWorld, nWorld, 20f, cliff, out _, out _);

            const int cliffLo = 20, cliffHi = 24;       // the far-forward row of the 5x5 fan
            for (int k = cliffLo; k <= cliffHi; k++) ranges[k] = cliff[k];
            ranges[0] = TankSensorRig.NoReturn;         // four beams find nothing at all
            ranges[1] = TankSensorRig.NoReturn;
            ranges[5] = TankSensorRig.NoReturn;
            ranges[6] = TankSensorRig.NoReturn;

            const int expectReturns = LidarGrid.Rays - 4, expectInliers = expectReturns - 5;

            GroundPlane g = GroundFit.Plane(ranges, dirs, LidarOrigin);

            Assert.IsTrue(g.Valid, "the fit was refused on a contaminated but clearly majority-ground fan");
            Assert.IsTrue(g.Returns == expectReturns,
                $"{g.Returns} beams returned, {expectReturns} expected (4 misses must be dropped, not ranged)");
            Assert.IsTrue(g.Inliers == expectInliers,
                $"{g.Inliers} beams agreed with the fitted plane, {expectInliers} expected (the 5 cliff returns must be rejected)");

            AssertPlane(g, expectNormal, expectClearance, 3f, 0.15f, "misses plus a 3 m cliff cluster");

            // Non-vacuity: the same points under a single robust fit, which is what the consensus step
            // replaced. If this came back clean the case would not be testing anything.
            var pts = new NativeArray<float3>(expectReturns, Allocator.Temp);
            int n = 0;
            for (int k = 0; k < LidarGrid.Rays; k++)
                if (ranges[k] > 0f) pts[n++] = LidarOrigin + dirs[k] * ranges[k];
            var loss = new floatHuberLoss(GroundFit.HuberScale);
            bool fitted = Fit.plane(pts, in loss, out _, out float3 naive);
            if (naive.y < 0f) naive = -naive;
            float naiveDeg = AngleDeg(naive, expectNormal);
            pts.Dispose();

            Assert.IsTrue(fitted, "test setup: the naive plane fit did not converge");
            Assert.IsTrue(naiveDeg > 0.5f,
                $"test setup: a single robust fit over the same points is only {naiveDeg} deg off, so the cliff is not contaminating anything");

            dirs.Dispose(); ranges.Dispose(); cliff.Dispose();
        }

        /// <summary>
        /// Too few returns is REFUSED, not fitted: the caller holds its last plane instead. Checked at
        /// one beam below the threshold and with the fan entirely blind, which is what flying off an
        /// escarpment looks like.
        /// </summary>
        [Test]
        public void GroundFit_RefusesBelowMinReturns()
        {
            var dirs = new NativeArray<float3>(LidarGrid.Rays, Allocator.Temp);
            var ranges = new NativeArray<float>(LidarGrid.Rays, Allocator.Temp);
            LidarGrid.Directions(dirs);

            float3x3 R = Attitude.Matrix(float3.zero);
            SynthRanges(dirs, LidarOrigin, R, new float3(0f, 3f, 0f),
                        float3.zero, new float3(0f, 1f, 0f), 12f, ranges, out _, out _);

            for (int k = GroundFit.MinReturns - 1; k < LidarGrid.Rays; k++)
                ranges[k] = TankSensorRig.NoReturn;

            GroundPlane thin = GroundFit.Plane(ranges, dirs, LidarOrigin);
            Assert.IsTrue(!thin.Valid,
                $"{GroundFit.MinReturns - 1} returns is below MinReturns = {GroundFit.MinReturns} and must be refused");
            Assert.IsTrue(thin.Returns == GroundFit.MinReturns - 1, $"counted {thin.Returns} returns");
            Assert.IsTrue(math.all(thin.Normal == new float3(0f, 1f, 0f)),
                $"a refused fit must report hull up, got {thin.Normal}");
            Assert.IsTrue(thin.Clearance == 0f, $"a refused fit must not report a clearance, got {thin.Clearance}");

            for (int k = 0; k < LidarGrid.Rays; k++) ranges[k] = TankSensorRig.NoReturn;
            GroundPlane blind = GroundFit.Plane(ranges, dirs, LidarOrigin);
            Assert.IsTrue(!blind.Valid && blind.Returns == 0,
                $"a blind fan must report 0 returns and no fit, got {blind.Returns} / valid={blind.Valid}");

            dirs.Dispose(); ranges.Dispose();
        }

        // ================================ 4. the estimator, end to end ==============================

        /// <summary>
        /// <see cref="TankEstimatorJob"/> over 3000 fixed steps (50 s) of an analytic manoeuvre —
        /// translating, heaving, yawing and rocking over sloping ground — fed the full default sensor
        /// suite: a cheap MEMS IMU with an unknown turn-on bias, a noisy magnetometer, a lidar fan and
        /// a beacon at one fix per 50 steps. Scored over the last 500 steps, because BIAS LEARNING
        /// takes roughly a minute of sim time and an average over the transient would be measuring the
        /// wrong thing.
        ///
        /// The peak-drift check is a guard on the test, not on the filter: the inertial solution has to
        /// visibly walk away between beacon fixes, or the position bound below would be passing on a
        /// filter that had quietly stopped propagating anything.
        /// </summary>
        [Test]
        public void Estimator_Converges_OnExcitedTrajectory()
        {
            float[] m = RunEstimator(TankSensorSpec.Default, excited: true);
            string all = Describe(m);

            Assert.IsTrue(m[7] == 1f, $"an EKF update reported a status other than Ok. {all}");
            Assert.IsTrue(m[10] == 0f, $"the ground fit was refused on {m[10]} steps. {all}");
            Assert.IsTrue(m[8] >= 20f, $"only {m[8]} lidar beams returned on the worst step. {all}");

            Assert.IsTrue(m[0] < 2f, $"mean position error {m[0]} m (beacon noise is 1.5 / 2.5 m). {all}");
            Assert.IsTrue(m[1] < 3f, $"mean attitude error {m[1]} deg. {all}");
            Assert.IsTrue(m[2] < 1f, $"mean velocity error {m[2]} m/s. {all}");
            Assert.IsTrue(m[3] < 0.05f, $"mean ride-height error {m[3]} m. {all}");
            Assert.IsTrue(m[9] < 6f, $"mean terrain-slope error {m[9]} deg. {all}");

            // A filter that learned nothing would sit at the true bias magnitude, 0.18 m/s^2.
            Assert.IsTrue(m[4] < 0.18f,
                $"the accelerometer bias is no better known than at start: gap {m[4]} m/s^2 against a true magnitude of 0.18. {all}");

            Assert.IsTrue(m[6] > 0.05f,
                $"peak position drift over the scoring window is only {m[6]} m — the inertial solution is not being propagated between beacon fixes, so the position bound proves nothing. {all}");
        }

        /// <summary>
        /// The same filter standing still over level ground with the sensor noise turned down to
        /// near-nothing. Everything is then observable and there is nothing left to average, so the
        /// estimate has to be RIGHT rather than merely well tuned — this is what separates a
        /// structurally correct filter from one whose covariances happen to suit the demo.
        /// </summary>
        [Test]
        public void Estimator_StaticAndNearNoiseless_PinsPosition()
        {
            // The turn-on biases stay at their default magnitude: they are the one error source this
            // case is NOT allowed to turn off, since learning them is the whole job.
            var quiet = TankSensorSpec.Default;
            quiet.accelNoise = 0.01f;
            quiet.gyroNoise = 0.001f;
            quiet.magNoise = 0.002f;
            quiet.lidarNoise = 0.003f;
            quiet.lidarCorrelation = 0f;
            quiet.gpsNoiseXZ = 0.02f;
            quiet.gpsNoiseY = 0.02f;
            quiet.proxNoise = 0.005f;
            quiet.tiltSigma = 0.005f;

            float[] m = RunEstimator(quiet, excited: false);
            string all = Describe(m);

            Assert.IsTrue(m[7] == 1f, $"an EKF update reported a status other than Ok. {all}");
            Assert.IsTrue(m[10] == 0f, $"the ground fit was refused on {m[10]} steps. {all}");

            Assert.IsTrue(m[0] < 0.05f, $"mean position error {m[0]} m at rest with near-noiseless sensors. {all}");
            Assert.IsTrue(m[2] < 0.1f, $"mean velocity error {m[2]} m/s at rest. {all}");
            Assert.IsTrue(m[3] < 0.02f, $"mean ride-height error {m[3]} m at rest. {all}");
        }

        // ================================ shared rig ================================================

        /// <summary>Just below the hull's bottom face, as <see cref="HoverTankMPCDemo"/> mounts it.</summary>
        static float3 LidarOrigin => new float3(0f, -0.55f, 0f);

        /// <summary>Every metric of an estimator run on one line, so any failure names the others.</summary>
        static string Describe(float[] m)
            => $"[pos {m[0]:F4} m | att {m[1]:F3} deg | vel {m[2]:F4} m/s | ride {m[3]:F4} m | "
             + $"accelBias gap {m[4]:F4} | gyroBias gap {m[5]:F5} | peak pos {m[6]:F4} m | "
             + $"allOk {m[7]} | min returns {m[8]} | slope {m[9]:F3} deg | refused fits {m[10]}]";

        /// <summary>
        /// Fills <paramref name="ranges"/> with the exact distance from the lidar mount to a WORLD
        /// plane through <paramref name="p0"/> with unit normal <paramref name="n"/>, for a hull at
        /// <paramref name="hullPos"/> with rotation <paramref name="R"/>. Beams that would run past
        /// <paramref name="rayLength"/>, or point away from the surface, report
        /// <see cref="TankSensorRig.NoReturn"/>. Also returns what the fit is then obliged to say: the
        /// same plane in HULL axes.
        /// </summary>
        static void SynthRanges(NativeArray<float3> dirs, float3 lidarLocal, float3x3 R, float3 hullPos,
                                float3 p0, float3 n, float rayLength, NativeArray<float> ranges,
                                out float3 hullNormal, out float clearance)
        {
            float3 originW = hullPos + math.mul(R, lidarLocal);
            clearance = math.dot(originW - p0, n);
            hullNormal = math.mul(math.transpose(R), n);

            for (int k = 0; k < dirs.Length; k++)
            {
                float3 dw = math.mul(R, dirs[k]);
                float den = math.dot(dw, n);
                float t = den < -1e-4f ? -clearance / den : -1f;
                ranges[k] = (t > 0f && t <= rayLength) ? t : TankSensorRig.NoReturn;
            }
        }

        static float AngleDeg(float3 a, float3 b)
            => math.degrees(math.acos(math.clamp(math.dot(math.normalize(a), math.normalize(b)), -1f, 1f)));

        static void AssertPlane(GroundPlane g, float3 expectNormal, float expectClearance,
                                float normalDegTol, float clearanceTol, string what)
        {
            float deg = AngleDeg(g.Normal, expectNormal);
            Assert.IsTrue(deg < normalDegTol,
                $"{what}: fitted normal {g.Normal} is {deg} deg off the true {expectNormal}");
            Assert.IsTrue(math.abs(math.lengthsq(g.Normal) - 1f) < 1e-4f,
                $"{what}: fitted normal is not unit, |n|^2 = {math.lengthsq(g.Normal)}");
            Assert.IsTrue(math.abs(g.Clearance - expectClearance) < clearanceTol,
                $"{what}: clearance {g.Clearance} m against a true {expectClearance} m");
        }

        /// <summary>
        /// Allocates every buffer <see cref="TankEstimatorJob"/> needs, seeds the filter the way
        /// <c>HoverTankMPCDemo.SeedEstimator</c> does (spawn pose, zero biases, a covariance saying how
        /// little of that is trusted), runs <see cref="TankEstimatorDriveJob"/> and returns its
        /// metrics. See that job for the layout of the returned array.
        /// </summary>
        static float[] RunEstimator(TankSensorSpec spec, bool excited)
        {
            const float dt = 1f / 60f, gravity = 9.81f, rideHeight = 3f;
            const int steps = 3000, window = 500;
            const int n = TankInsModel.N;

            // Bias random walk, matching HoverTankMPCDemo's own AccelBiasWalk / GyroBiasWalk.
            const float accelBiasWalk = 0.06f, gyroBiasWalk = 0.004f;

            var dirs = new NativeArray<float3>(LidarGrid.Rays, Allocator.Persistent);
            LidarGrid.Directions(dirs);
            var lidarTrue = new NativeArray<float>(LidarGrid.Rays, Allocator.Persistent);
            var lidarSensed = new NativeArray<float>(LidarGrid.Rays, Allocator.Persistent);
            var proxTrue = new NativeArray<float>(ProximityRig.Rays, Allocator.Persistent);
            var proxSensed = new NativeArray<float>(ProximityRig.Rays, Allocator.Persistent);

            var kf = new floatKFState(n, 3, Allocator.Persistent);
            var Q = new floatMxN(n, n, Allocator.Persistent);
            var RMag = new floatMxN(3, 3, Allocator.Persistent);
            var RGps = new floatMxN(3, 3, Allocator.Persistent);
            var kfOut = new NativeArray<KFInfo>(3, Allocator.Persistent);
            var estOut = new NativeArray<TankEstimate>(1, Allocator.Persistent);
            var hoverState = new NativeArray<float>(6, Allocator.Persistent);
            var ground = new NativeArray<GroundPlane>(1, Allocator.Persistent);
            var gpsAge = new NativeArray<int>(1, Allocator.Persistent);
            var stats = new NativeArray<float>(11, Allocator.Persistent);

            TankSensorNoise noise = TankSensorNoise.Build(in spec, 0x5E45E1u, Allocator.Persistent);
            Assert.IsTrue(noise.Factored, "a sensor covariance did not factor — the noise model is broken");

            // Filter covariances, exactly HoverTankMPCDemo.BuildFilterNoise at a fixed dt.
            float qv = spec.accelNoise * dt, qa = spec.gyroNoise * dt;
            float qba = accelBiasWalk * dt, qbg = gyroBiasWalk * dt;
            for (int i = 0; i < 3; i++)
            {
                Q[TankInsModel.Pos + i, TankInsModel.Pos + i] = 1e-6f;
                Q[TankInsModel.Vel + i, TankInsModel.Vel + i] = qv * qv;
                Q[TankInsModel.Att + i, TankInsModel.Att + i] = qa * qa;
                Q[TankInsModel.AccelBias + i, TankInsModel.AccelBias + i] = qba * qba;
                Q[TankInsModel.GyroBias + i, TankInsModel.GyroBias + i] = qbg * qbg;

                RMag[i, i] = spec.magNoise * spec.magNoise;
            }
            RGps[0, 0] = spec.gpsNoiseXZ * spec.gpsNoiseXZ;
            RGps[1, 1] = spec.gpsNoiseY * spec.gpsNoiseY;
            RGps[2, 2] = spec.gpsNoiseXZ * spec.gpsNoiseXZ;

            // A cold start: both IMU biases at zero, so the filter has to find them.
            for (int i = 0; i < 3; i++)
            {
                kf.P[TankInsModel.Pos + i, TankInsModel.Pos + i] = 4f;
                kf.P[TankInsModel.Vel + i, TankInsModel.Vel + i] = 1f;
                kf.P[TankInsModel.Att + i, TankInsModel.Att + i] = 0.01f;
                kf.P[TankInsModel.AccelBias + i, TankInsModel.AccelBias + i] = 0.25f;
                kf.P[TankInsModel.GyroBias + i, TankInsModel.GyroBias + i] = 2.5e-3f;
            }

            ground[0] = new GroundPlane
            {
                Normal = new float3(0f, 1f, 0f),
                Clearance = rideHeight,
                Valid = false,
            };

            var job = new TankEstimatorDriveJob
            {
                Steps = steps, Window = window,
                Dt = dt, Gravity = gravity, TargetRideHeight = rideHeight, RayLength = 20f,

                Center = excited ? new float3(5f, 3.5f, -4f) : new float3(0f, 3.5f, 0f),
                GroundPoint = excited ? new float3(5f, 0f, -4f) : float3.zero,
                GroundNormalWorld = excited
                    ? math.normalize(new float3(-0.15f, 1f, 0.10f))   // 9.9 deg terrain slope
                    : new float3(0f, 1f, 0f),
                PosAmpX = excited ? 3f : 0f,
                PosAmpY = excited ? 0.3f : 0f,
                PosAmpZ = excited ? 3f : 0f,
                PosOmega = 0.25f,
                RollAmp = excited ? 0.10f : 0f,
                PitchAmp = excited ? 0.08f : 0f,
                AttOmega = 0.5f,
                YawRate = excited ? 0.18f : 0f,

                // Same magnitudes the demo draws its turn-on biases at, in fixed directions.
                TrueAccelBias = math.normalize(new float3(0.6f, -0.3f, 0.75f)) * spec.accelBias,
                TrueGyroBias = math.normalize(new float3(-0.4f, 0.8f, 0.45f)) * spec.gyroBias,
                Spec = spec,

                LidarDirs = dirs, LidarOrigin = LidarOrigin,
                LidarTrue = lidarTrue, LidarSensed = lidarSensed,
                ProxTrue = proxTrue, ProxSensed = proxSensed,

                Noise = noise, Kf = kf, Q = Q, RMag = RMag, RGps = RGps,
                KfOut = kfOut, EstOut = estOut, HoverState = hoverState, Ground = ground,
                GpsAge = gpsAge, Out = stats,
            };
            IJobExtensions.RunByRef(ref job);

            var m = new float[stats.Length];
            for (int i = 0; i < stats.Length; i++) m[i] = stats[i];

            dirs.Dispose(); lidarTrue.Dispose(); lidarSensed.Dispose();
            proxTrue.Dispose(); proxSensed.Dispose();
            kf.Dispose(); Q.Dispose(); RMag.Dispose(); RGps.Dispose();
            kfOut.Dispose(); estOut.Dispose(); hoverState.Dispose(); ground.Dispose();
            gpsAge.Dispose(); stats.Dispose();
            job.Noise.Dispose();

            return m;
        }
    }

    /// <summary>
    /// Compares the demo's analytic Jacobians against <c>Kalman.numericJacobianF/H</c> at four states,
    /// and measures how far three deliberately WRONG Jacobians land from the same numeric reference.
    ///
    /// <see cref="Eps"/> must be passed explicitly and kept well above the default: the state carries
    /// metre-scale positions, and at the default step the differencing noise floor alone reads as a
    /// Jacobian error of several parts in a thousand.
    ///
    /// Out, Transition: [max |numeric - analytic|, deviation of a Jacobian built at 2·dt, deviation
    /// with the attitude coupling negated, states tested].
    /// Out, Measurement: [max deviation on the direction measurement, max on the beacon, deviation at
    /// the wrong reference direction, deviation with the attitude block negated, states tested].
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct TankJacobianJob : IJob
    {
        public enum Kind { Transition, Measurement }

        public Kind Which;
        public float Dt, Gravity, Eps;
        public NativeArray<float> Out;

        const int Cases = 4;

        public void Execute()
        {
            if (Which == Kind.Transition) Transition();
            else Measurement();
        }

        /// <summary>
        /// One test state and its IMU sample. Roll, pitch and yaw are all far from zero in every case:
        /// an error in a rotation derivative can vanish identically at roll = 0, which is where a level
        /// hover sits, so a level state would not see it. Pitch stays clear of
        /// <see cref="Attitude.MinPitchCos"/>, past which the Euler rates SATURATE by design and the
        /// analytic Jacobian of the saturated map is deliberately not the derivative of it.
        /// </summary>
        static void Sample(int c, ref floatN x, ref floatN u)
        {
            float3 p, v, a, ba, bg, f, w;
            if (c == 0)
            {
                p = new float3(12.5f, 6.25f, -30f); v = new float3(3.2f, -0.7f, 1.4f);
                a = new float3(0.35f, -0.22f, 1.1f);
                ba = new float3(0.05f, -0.11f, 0.02f); bg = new float3(0.003f, -0.002f, 0.004f);
                f = new float3(0.4f, 9.6f, -0.8f); w = new float3(0.12f, -0.25f, 0.4f);
            }
            else if (c == 1)
            {
                p = new float3(-45f, 3.1f, 18.7f); v = new float3(-2f, 0.5f, -4f);
                a = new float3(-0.9f, 0.55f, -2.3f);
                ba = new float3(-0.2f, 0.08f, 0.14f); bg = new float3(-0.005f, 0.006f, -0.001f);
                f = new float3(-1.2f, 9.2f, 2.1f); w = new float3(-0.5f, 0.3f, -0.8f);
            }
            else if (c == 2)
            {
                // Steep but regular: cos(pitch) = 0.32, well above the MinPitchCos floor of 0.1.
                p = new float3(0.5f, 2f, 0.25f); v = new float3(0.1f, 0.2f, -0.3f);
                a = new float3(1.3f, -1.25f, 0.2f);
                ba = new float3(0.12f, 0.03f, -0.07f); bg = new float3(0.002f, 0.001f, -0.003f);
                f = new float3(2.5f, 8.8f, 1.5f); w = new float3(0.9f, -0.6f, 0.2f);
            }
            else
            {
                p = new float3(-3.75f, 1.5f, 8f); v = new float3(-0.4f, 0.05f, 0.9f);
                a = new float3(0.6f, 0.4f, -2.9f);
                ba = new float3(-0.02f, -0.15f, 0.09f); bg = new float3(-0.001f, -0.004f, 0.002f);
                f = new float3(-0.7f, 9.9f, 0.3f); w = new float3(0.05f, 0.8f, -0.15f);
            }

            TankInsModel.Write(ref x, TankInsModel.Pos, p);
            TankInsModel.Write(ref x, TankInsModel.Vel, v);
            TankInsModel.Write(ref x, TankInsModel.Att, a);
            TankInsModel.Write(ref x, TankInsModel.AccelBias, ba);
            TankInsModel.Write(ref x, TankInsModel.GyroBias, bg);
            TankInsModel.Write(ref u, 0, f);
            TankInsModel.Write(ref u, 3, w);
        }

        void Transition()
        {
            const int n = TankInsModel.N;

            var model = new TankInsModel { Dt = Dt, Gravity = Gravity };
            var coarse = new TankInsModel { Dt = 2f * Dt, Gravity = Gravity };

            var x = new floatN(n, Allocator.Temp);
            var u = new floatN(6, Allocator.Temp);
            var Ja = new floatMxN(n, n, Allocator.Temp);
            var Jn = new floatMxN(n, n, Allocator.Temp);
            var Jp = new floatMxN(n, n, Allocator.Temp);

            float maxErr = 0f, probeDt = 0f, probeAtt = 0f;

            for (int c = 0; c < Cases; c++)
            {
                Sample(c, ref x, ref u);

                model.JacobianF(in x, in u, ref Ja);
                Kalman.numericJacobianF(in model, in x, in u, ref Jn, Eps);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        maxErr = math.max(maxErr, math.abs(Jn[i, j] - Ja[i, j]));

                // Probe 1: the same model differentiated at twice the propagation interval. Every
                // dt-scaled entry moves, and none of the identity structure does.
                coarse.JacobianF(in x, in u, ref Jp);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        probeDt = math.max(probeDt, math.abs(Jn[i, j] - Jp[i, j]));

                // Probe 2: the attitude columns' DERIVATIVE CONTENT negated, with their identity part
                // left alone. Flipping the whole column would also be caught, but only by the 1 on the
                // diagonal — this version proves the coupling itself is being compared.
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        Jp[i, j] = Ja[i, j];
                for (int j = 0; j < 3; j++)
                    for (int i = 0; i < n; i++)
                    {
                        int col = TankInsModel.Att + j;
                        float id = i == col ? 1f : 0f;
                        Jp[i, col] = id - (Ja[i, col] - id);
                    }
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        probeAtt = math.max(probeAtt, math.abs(Jn[i, j] - Jp[i, j]));
            }

            Out[0] = maxErr; Out[1] = probeDt; Out[2] = probeAtt; Out[3] = Cases;

            x.Dispose(); u.Dispose(); Ja.Dispose(); Jn.Dispose(); Jp.Dispose();
        }

        void Measurement()
        {
            const int n = TankInsModel.N;

            var x = new floatN(n, Allocator.Temp);
            var u = new floatN(6, Allocator.Temp);
            var Ha = new floatMxN(3, n, Allocator.Temp);
            var Hn = new floatMxN(3, n, Allocator.Temp);
            var Hp = new floatMxN(3, n, Allocator.Temp);

            float3 up = new float3(0f, 1f, 0f);
            var spec = TankSensorSpec.Default;
            float3 field = spec.MagField();

            float maxVec = 0f, maxPos = 0f, probeRef = 0f, probeNeg = 0f;

            for (int c = 0; c < Cases; c++)
            {
                Sample(c, ref x, ref u);

                for (int r = 0; r < 2; r++)
                {
                    float3 reference = r == 0 ? up : field;
                    float3 other = r == 0 ? field : up;

                    var meas = new TankVectorMeasurement { Reference = reference };
                    meas.JacobianH(in x, ref Ha);
                    Kalman.numericJacobianH(in meas, in x, ref Hn, Eps);
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < n; j++)
                            maxVec = math.max(maxVec, math.abs(Hn[i, j] - Ha[i, j]));

                    // Probe 1: the analytic Jacobian of the OTHER reference direction. Both are unit
                    // vectors of the same shape, so this is the shape of a reference wired to the
                    // wrong stream rather than an obviously broken matrix.
                    var wrong = new TankVectorMeasurement { Reference = other };
                    wrong.JacobianH(in x, ref Hp);
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < n; j++)
                            probeRef = math.max(probeRef, math.abs(Hn[i, j] - Hp[i, j]));

                    // Probe 2: the three attitude columns negated — the only nonzero block there is.
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < n; j++)
                            Hp[i, j] = j >= TankInsModel.Att && j < TankInsModel.Att + 3
                                     ? -Ha[i, j] : Ha[i, j];
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < n; j++)
                            probeNeg = math.max(probeNeg, math.abs(Hn[i, j] - Hp[i, j]));
                }

                var beacon = new TankPositionMeasurement();
                beacon.JacobianH(in x, ref Ha);
                Kalman.numericJacobianH(in beacon, in x, ref Hn, Eps);
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < n; j++)
                        maxPos = math.max(maxPos, math.abs(Hn[i, j] - Ha[i, j]));
            }

            Out[0] = maxVec; Out[1] = maxPos; Out[2] = probeRef; Out[3] = probeNeg; Out[4] = Cases;

            x.Dispose(); u.Dispose(); Ha.Dispose(); Hn.Dispose(); Hp.Dispose();
        }
    }

    /// <summary>
    /// Drives <see cref="TankEstimatorJob"/> for <see cref="Steps"/> fixed steps against an ANALYTIC
    /// trajectory, scoring the estimate over the last <see cref="Window"/> of them. Truth is
    /// differentiated in closed form rather than differenced, so the specific force and body rate the
    /// simulated IMU is handed are exact.
    ///
    /// Running it inside a job is what forces Burst to compile the whole estimator path; the inner
    /// job's <c>Execute</c> is called directly so the 15-state covariance and the sensor random stream
    /// carry across steps — both live in structs the job advances and both are copied back each step.
    ///
    /// Out is [mean |position error| (m), mean attitude error (deg), mean |velocity error| (m/s), mean
    /// |ride-height error| (m), mean accelerometer-bias gap (m/s²), mean gyro-bias gap (rad/s), PEAK
    /// position error inside the window (m), 1 if every EKF update reported Ok, fewest lidar returns on
    /// any step, mean terrain-slope error (deg), steps whose ground fit was refused].
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct TankEstimatorDriveJob : IJob
    {
        public int Steps, Window;
        public float Dt, Gravity, TargetRideHeight, RayLength;

        /// <summary>Trajectory centre, and the world ground plane the lidar ranges against.</summary>
        public float3 Center, GroundPoint, GroundNormalWorld;

        /// <summary>Translation amplitudes (m) and their angular frequency (rad/s).</summary>
        public float PosAmpX, PosAmpY, PosAmpZ, PosOmega;

        /// <summary>Rocking amplitudes (rad), their frequency, and a constant yaw rate (rad/s).</summary>
        public float RollAmp, PitchAmp, AttOmega, YawRate;

        /// <summary>True IMU turn-on biases. The filter is never told them.</summary>
        public float3 TrueAccelBias, TrueGyroBias;

        public TankSensorSpec Spec;

        [ReadOnly] public NativeArray<float3> LidarDirs;
        public float3 LidarOrigin;
        public NativeArray<float> LidarTrue, LidarSensed, ProxTrue, ProxSensed;

        public TankSensorNoise Noise;
        public floatKFState Kf;
        public floatMxN Q, RMag, RGps;

        public NativeArray<KFInfo> KfOut;
        public NativeArray<TankEstimate> EstOut;
        public NativeArray<float> HoverState;
        public NativeArray<GroundPlane> Ground;
        public NativeArray<int> GpsAge;

        public NativeArray<float> Out;

        public void Execute()
        {
            TankSensorNoise noise = Noise;
            floatKFState kf = Kf;

            // The filter starts at the spawn pose, which is a design constant of the demo rather than
            // a reading off the vehicle. Everything else stays at the cold-start seed.
            TruthAt(0f, out TankTruth spawn, out _);
            TankInsModel.Write(ref kf.x, TankInsModel.Pos, spawn.Position);

            double aPos = 0, aAtt = 0, aVel = 0, aClr = 0, aBa = 0, aBg = 0, aSlope = 0;
            int scored = 0, refused = 0, minReturns = int.MaxValue;
            float peakPos = 0f;
            bool allOk = true;

            for (int s = 0; s < Steps; s++)
            {
                TruthAt(s * Dt, out TankTruth truth, out float3 rpyTrue);

                float3 originW = truth.Position
                               + truth.Right * LidarOrigin.x
                               + truth.Up * LidarOrigin.y
                               + truth.Fwd * LidarOrigin.z;
                float clearanceTrue = math.dot(originW - GroundPoint, GroundNormalWorld);

                for (int k = 0; k < LidarTrue.Length; k++)
                {
                    float3 d = LidarDirs[k];
                    float3 dw = truth.Right * d.x + truth.Up * d.y + truth.Fwd * d.z;
                    float den = math.dot(dw, GroundNormalWorld);
                    float t = den < -1e-4f ? -clearanceTrue / den : -1f;
                    LidarTrue[k] = t > 0f && t <= RayLength ? t : TankSensorRig.NoReturn;
                }
                // The proximity rangers are wired but unread by the control law; they still have to be
                // driven, because their draw advances the same random stream.
                for (int k = 0; k < ProxTrue.Length; k++) ProxTrue[k] = 5f;

                var est = new TankEstimatorJob
                {
                    Truth = truth,
                    TrueAccelBias = TrueAccelBias, TrueGyroBias = TrueGyroBias,
                    LidarDirs = LidarDirs, LidarOrigin = LidarOrigin,
                    LidarTrue = LidarTrue, LidarSensed = LidarSensed,
                    ProxTrue = ProxTrue, ProxSensed = ProxSensed,
                    Noise = noise, Spec = Spec,
                    Kf = kf, Q = Q, RMag = RMag, RGps = RGps,
                    KfOut = KfOut, Out = EstOut, HoverState = HoverState, Ground = Ground,
                    GpsAge = GpsAge,
                    Dt = Dt, Gravity = Gravity, TargetRideHeight = TargetRideHeight,
                    Step = s,
                };
                est.Execute();

                // Both carry state the job advanced: the covariance and the random stream.
                noise = est.Noise;
                kf = est.Kf;

                TankEstimate e = EstOut[0];
                if (e.TiltFix && KfOut[0].status != KFStatus.Ok) allOk = false;
                if (e.MagFix && KfOut[1].status != KFStatus.Ok) allOk = false;
                if (e.GpsFix && KfOut[2].status != KFStatus.Ok) allOk = false;
                if (!e.GroundValid) refused++;
                minReturns = math.min(minReturns, e.LidarReturns);

                if (s >= Steps - Window)
                {
                    float pe = math.length(e.Position - truth.Position);
                    peakPos = math.max(peakPos, pe);

                    aPos += pe;
                    aAtt += math.length(Attitude.Difference(e.Rpy, rpyTrue));
                    aVel += math.length(e.Velocity - truth.Velocity);
                    aClr += math.abs(e.Clearance - clearanceTrue);
                    aBa += math.length(e.AccelBias - TrueAccelBias);
                    aBg += math.length(e.GyroBias - TrueGyroBias);
                    aSlope += math.acos(math.clamp(
                        math.dot(math.normalize(e.GroundNormal), GroundNormalWorld), -1f, 1f));
                    scored++;
                }
            }

            Noise = noise;
            Kf = kf;

            double inv = scored > 0 ? 1.0 / scored : 0.0;
            Out[0] = (float)(aPos * inv);
            Out[1] = math.degrees((float)(aAtt * inv));
            Out[2] = (float)(aVel * inv);
            Out[3] = (float)(aClr * inv);
            Out[4] = (float)(aBa * inv);
            Out[5] = (float)(aBg * inv);
            Out[6] = peakPos;
            Out[7] = allOk ? 1f : 0f;
            Out[8] = minReturns == int.MaxValue ? 0f : minReturns;
            Out[9] = math.degrees((float)(aSlope * inv));
            Out[10] = refused;
        }

        /// <summary>
        /// The hull's exact state at time <paramref name="t"/>: a lissajous translation with a heave,
        /// a constant yaw rate and a slow roll/pitch rock. Velocity and acceleration are the analytic
        /// derivatives, and the body rate is the Euler rate mapped back through
        /// <see cref="Attitude.RateMatrix"/>, so the IMU sample the estimator is handed is consistent
        /// with the pose it is asked to recover rather than approximately so.
        /// </summary>
        void TruthAt(float t, out TankTruth truth, out float3 rpy)
        {
            float wp = PosOmega, wa = AttOmega;

            rpy = new float3(RollAmp * math.sin(wa * t),
                             PitchAmp * math.sin(0.7f * wa * t + 0.9f),
                             YawRate * t);
            float3 rpyDot = new float3(RollAmp * wa * math.cos(wa * t),
                                       PitchAmp * 0.7f * wa * math.cos(0.7f * wa * t + 0.9f),
                                       YawRate);

            float3 p = Center + new float3(PosAmpX * math.sin(wp * t),
                                           PosAmpY * math.sin(0.5f * wp * t),
                                           PosAmpZ * math.cos(wp * t));
            float3 v = new float3(PosAmpX * wp * math.cos(wp * t),
                                  PosAmpY * 0.5f * wp * math.cos(0.5f * wp * t),
                                  -PosAmpZ * wp * math.sin(wp * t));
            float3 a = new float3(-PosAmpX * wp * wp * math.sin(wp * t),
                                  -PosAmpY * 0.25f * wp * wp * math.sin(0.5f * wp * t),
                                  -PosAmpZ * wp * wp * math.cos(wp * t));

            float3x3 R = Attitude.Matrix(rpy);

            truth = new TankTruth
            {
                Position = p,
                Velocity = v,
                Right = R.c0, Up = R.c1, Fwd = R.c2,
                AngularRate = math.mul(math.inverse(Attitude.RateMatrix(rpy)), rpyDot),
            };
            truth.SpecificForce = truth.ToBody(a - new float3(0f, -Gravity, 0f));
        }
    }
}
