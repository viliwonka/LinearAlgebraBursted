using BULA;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// Hull attitude as three Euler angles [roll, pitch, yaw] in radians, composed
    /// R = Ry(yaw)·Rx(pitch)·Rz(roll) — Unity's own order, so the state can be read against
    /// <c>Quaternion.Euler(pitch, yaw, roll)</c> without a conversion. Angles turn hull axes
    /// (x right, y up, z forward) into world axes.
    ///
    /// WHY THREE PARAMETERS AND WHERE THEY BREAK. A three-parameter attitude always has a singular
    /// pose; the choice is only WHICH pose. This one degenerates at PITCH = ±90°, nose straight up or
    /// straight down, and is regular everywhere else — roll and yaw both turn a full circle freely,
    /// which is what a vehicle steered by the mouse needs. A rotation-vector parametrization would
    /// instead degenerate at a half turn, which yaw reaches every few seconds of driving. The pitch
    /// pole is a crashed hover tank, so the filter carries the singularity where the vehicle cannot
    /// go. <see cref="MinPitchCos"/> is what keeps the arithmetic finite if it is driven there anyway.
    ///
    /// Positive pitch points the nose down and positive roll lifts the hull's right side, matching the
    /// torque signs in <see cref="GimbalWrench"/>.
    /// </summary>
    public static class Attitude
    {
        /// <summary>
        /// Floor on |cos(pitch)| in the two places it divides. Reached at 84° of pitch: past that the
        /// Euler rates and their Jacobian saturate instead of blowing up, and neither is meaningful.
        /// </summary>
        public const float MinPitchCos = 0.1f;

        /// <summary>Rotation about hull +x by <paramref name="a"/> radians.</summary>
        public static float3x3 RotX(float a)
        {
            math.sincos(a, out float s, out float c);
            return new float3x3(1f, 0f, 0f,
                                0f, c, -s,
                                0f, s, c);
        }

        /// <summary>Rotation about hull +y by <paramref name="a"/> radians.</summary>
        public static float3x3 RotY(float a)
        {
            math.sincos(a, out float s, out float c);
            return new float3x3(c, 0f, s,
                                0f, 1f, 0f,
                                -s, 0f, c);
        }

        /// <summary>Rotation about hull +z by <paramref name="a"/> radians.</summary>
        public static float3x3 RotZ(float a)
        {
            math.sincos(a, out float s, out float c);
            return new float3x3(c, -s, 0f,
                                s, c, 0f,
                                0f, 0f, 1f);
        }

        /// <summary>World-from-body rotation for <paramref name="rpy"/> = [roll, pitch, yaw].</summary>
        public static float3x3 Matrix(float3 rpy)
            => math.mul(RotY(rpy.z), math.mul(RotX(rpy.y), RotZ(rpy.x)));

        /// <summary>
        /// [roll, pitch, yaw] of the rotation whose columns are the hull axes in world space — the
        /// inverse of <see cref="Matrix"/>, and how truth is read for the error plot.
        /// </summary>
        public static float3 FromBasis(float3 right, float3 up, float3 fwd)
        {
            // R[1,0] = cos(pitch)·sin(roll), R[1,1] = cos(pitch)·cos(roll), R[1,2] = -sin(pitch),
            // R[0,2] = sin(yaw)·cos(pitch), R[2,2] = cos(yaw)·cos(pitch).
            float pitch = math.asin(math.clamp(-fwd.y, -1f, 1f));
            float roll = math.atan2(right.y, up.y);
            float yaw = math.atan2(fwd.x, fwd.z);
            return new float3(roll, pitch, yaw);
        }

        /// <summary>
        /// Body angular rate to Euler-angle rate, so that d(rpy)/dt = <see cref="RateMatrix"/>·w.
        /// Singular at pitch = ±90° (see <see cref="MinPitchCos"/>).
        /// </summary>
        public static float3x3 RateMatrix(float3 rpy)
        {
            math.sincos(rpy.x, out float sr, out float cr);
            float cp = math.max(math.cos(rpy.y), MinPitchCos);
            float tp = math.sin(rpy.y) / cp;

            return new float3x3(tp * sr, tp * cr, 1f,
                                cr, -sr, 0f,
                                sr / cp, cr / cp, 0f);
        }

        /// <summary>Euler-angle rates at attitude <paramref name="rpy"/> for body rate <paramref name="w"/>.</summary>
        public static float3 Rates(float3 rpy, float3 w) => math.mul(RateMatrix(rpy), w);

        /// <summary>
        /// d(<see cref="Matrix"/>(rpy)·v) / d(rpy) — columns in [roll, pitch, yaw] order.
        /// </summary>
        public static float3x3 DRotate(float3 rpy, float3 v)
        {
            float3x3 rz = RotZ(rpy.x), rx = RotX(rpy.y), ry = RotY(rpy.z);
            float3x3 rxz = math.mul(rx, rz);
            float3x3 r = math.mul(ry, rxz);

            // dR/d(angle) = R with the elementary generator of that angle's axis spliced in at its
            // own position in the product, so each column is one cross product in the right frame.
            float3 dRoll = math.mul(r, math.cross(new float3(0f, 0f, 1f), v));
            float3 dPitch = math.mul(ry, math.cross(new float3(1f, 0f, 0f), math.mul(rxz, v)));
            float3 dYaw = math.cross(new float3(0f, 1f, 0f), math.mul(r, v));

            return new float3x3(dRoll, dPitch, dYaw);
        }

        /// <summary>
        /// d(<see cref="Matrix"/>(rpy)ᵀ·v) / d(rpy) — the world-vector-into-hull-axes direction, which
        /// is the form every direction measurement takes. Columns in [roll, pitch, yaw] order.
        /// </summary>
        public static float3x3 DRotateT(float3 rpy, float3 v)
        {
            float3x3 rz = RotZ(rpy.x), rx = RotX(rpy.y), ry = RotY(rpy.z);
            float3x3 rxz = math.mul(rx, rz);
            float3x3 r = math.mul(ry, rxz);

            float3 dRoll = -math.cross(new float3(0f, 0f, 1f), math.mul(math.transpose(r), v));
            float3 dPitch = -math.mul(math.transpose(rxz),
                                      math.cross(new float3(1f, 0f, 0f), math.mul(math.transpose(ry), v)));
            float3 dYaw = -math.mul(math.transpose(r), math.cross(new float3(0f, 1f, 0f), v));

            return new float3x3(dRoll, dPitch, dYaw);
        }

        /// <summary>
        /// d(<see cref="Rates"/>(rpy, w)) / d(rpy) — columns in [roll, pitch, yaw] order. The yaw
        /// column is zero: the Euler rates do not depend on heading.
        /// </summary>
        public static float3x3 DRates(float3 rpy, float3 w)
        {
            math.sincos(rpy.x, out float sr, out float cr);
            float cp = math.max(math.cos(rpy.y), MinPitchCos);
            float sp = math.sin(rpy.y);
            float tp = sp / cp;
            float sec2 = 1f / (cp * cp);

            float s = w.x * sr + w.y * cr;     // the combination every row is built from
            float c = w.x * cr - w.y * sr;     // and its derivative in roll

            float3 dRoll = new float3(tp * c, -s, c / cp);
            float3 dPitch = new float3(s * sec2, 0f, s * sp * sec2);
            return new float3x3(dRoll, dPitch, float3.zero);
        }

        /// <summary>
        /// The small rotation that takes attitude <paramref name="a"/> onto attitude
        /// <paramref name="b"/>, as a body-frame rotation vector (radians). Wrap-free, so it is the
        /// honest way to report attitude error even when yaw is near half a turn.
        /// </summary>
        public static float3 Difference(float3 a, float3 b)
        {
            float3x3 e = math.mul(math.transpose(Matrix(a)), Matrix(b));

            // Rotation vector of e from its skew part, scaled by the angle. Exact at small angles,
            // which is the regime an attitude error lives in.
            float3 axis = 0.5f * new float3(e[1][2] - e[2][1], e[2][0] - e[0][2], e[0][1] - e[1][0]);
            float sin = math.length(axis);
            float cos = 0.5f * (e[0][0] + e[1][1] + e[2][2] - 1f);
            float angle = math.atan2(sin, cos);
            return sin > 1e-8f ? axis * (angle / sin) : axis;
        }
    }

    /// <summary>
    /// The 15-state strapdown inertial model the EKF propagates:
    /// [position(3), velocity(3), attitude(3), accelerometer bias(3), gyro bias(3)], world axes for
    /// the first two, <see cref="Attitude"/> Euler angles for the third, hull axes for the biases.
    /// The control input is the IMU sample [specific force(3), angular rate(3)] in hull axes — an IMU
    /// drives the propagation, it does not measure the state.
    ///
    /// Both bias triples are random-walk (identity dynamics); their process noise is what lets the
    /// filter learn them.
    /// </summary>
    public struct TankInsModel : IfloatKFModel
    {
        /// <summary>Propagation interval, seconds.</summary>
        public float Dt;

        /// <summary>Gravitational acceleration magnitude, m/s². World gravity is (0, -Gravity, 0).</summary>
        public float Gravity;

        /// <summary>State index of the first component of each block.</summary>
        public const int Pos = 0, Vel = 3, Att = 6, AccelBias = 9, GyroBias = 12, N = 15;

        public void F(in floatN x, in floatN u, ref floatN xNext)
        {
            float3 p = Read(in x, Pos), v = Read(in x, Vel), a = Read(in x, Att);
            float3 ba = Read(in x, AccelBias), bg = Read(in x, GyroBias);

            float3 f = Read(in u, 0) - ba;
            float3 w = Read(in u, 3) - bg;

            float3 accel = math.mul(Attitude.Matrix(a), f) + new float3(0f, -Gravity, 0f);

            Write(ref xNext, Pos, p + v * Dt);
            Write(ref xNext, Vel, v + accel * Dt);
            Write(ref xNext, Att, a + Attitude.Rates(a, w) * Dt);
            Write(ref xNext, AccelBias, ba);
            Write(ref xNext, GyroBias, bg);
        }

        public void JacobianF(in floatN x, in floatN u, ref floatMxN J)
        {
            float3 a = Read(in x, Att);
            float3 f = Read(in u, 0) - Read(in x, AccelBias);
            float3 w = Read(in u, 3) - Read(in x, GyroBias);

            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    J[i, j] = i == j ? 1f : 0f;

            // position <- velocity
            for (int i = 0; i < 3; i++) J[Pos + i, Vel + i] = Dt;

            // velocity <- attitude (the gravity-compensated specific force turns with the hull)
            // and velocity <- accelerometer bias.
            float3x3 dv = Attitude.DRotate(a, f);
            float3x3 r = Attitude.Matrix(a);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    J[Vel + i, Att + j] = Dt * dv[j][i];
                    J[Vel + i, AccelBias + j] = -Dt * r[j][i];
                }

            // attitude <- attitude (the Euler-rate map itself turns with the hull) and
            // attitude <- gyro bias.
            float3x3 da = Attitude.DRates(a, w);
            float3x3 e = Attitude.RateMatrix(a);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    J[Att + i, Att + j] += Dt * da[j][i];
                    J[Att + i, GyroBias + j] = -Dt * e[j][i];
                }
        }

        /// <summary>Reads the 3-vector starting at <paramref name="at"/>.</summary>
        public static float3 Read(in floatN x, int at) => new float3(x[at], x[at + 1], x[at + 2]);

        /// <summary>Writes the 3-vector starting at <paramref name="at"/>.</summary>
        public static void Write(ref floatN x, int at, float3 v)
        {
            x[at] = v.x; x[at + 1] = v.y; x[at + 2] = v.z;
        }
    }

    /// <summary>
    /// A known world direction seen in hull axes: h(x) = R(attitude)ᵀ·<see cref="Reference"/>.
    ///
    /// Two of the demo's three measurement streams are this shape. The MAGNETOMETER reads the local
    /// field, whose world direction is fixed and near horizontal, so it is what pins heading. The
    /// GRAVITY REFERENCE reads the accelerometer's bias-corrected output when the hull is not
    /// manoeuvring, where it points along world up, so it is what levels roll and pitch. Neither
    /// observes rotation about its own reference direction, which is why the demo runs both.
    ///
    /// Position, velocity and the biases are unobserved by this measurement, so all but the three
    /// attitude columns of the Jacobian are zero.
    /// </summary>
    public struct TankVectorMeasurement : IfloatKFMeasurement
    {
        /// <summary>The measured direction expressed in WORLD axes. Unit length.</summary>
        public float3 Reference;

        public void H(in floatN x, ref floatN z)
        {
            float3 b = math.mul(math.transpose(Attitude.Matrix(TankInsModel.Read(in x, TankInsModel.Att))),
                                Reference);
            z[0] = b.x; z[1] = b.y; z[2] = b.z;
        }

        public void JacobianH(in floatN x, ref floatMxN J)
        {
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < TankInsModel.N; j++) J[i, j] = 0f;

            float3x3 d = Attitude.DRotateT(TankInsModel.Read(in x, TankInsModel.Att), Reference);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) J[i, TankInsModel.Att + j] = d[j][i];
        }
    }

    /// <summary>
    /// The beacon: absolute world position, h(x) = position. Slow and noisy, and the ONLY thing that
    /// bounds horizontal position — an inertial solution alone integrates its own accelerometer error
    /// twice and walks away without limit.
    /// </summary>
    public struct TankPositionMeasurement : IfloatKFMeasurement
    {
        public void H(in floatN x, ref floatN z)
        {
            z[0] = x[TankInsModel.Pos]; z[1] = x[TankInsModel.Pos + 1]; z[2] = x[TankInsModel.Pos + 2];
        }

        public void JacobianH(in floatN x, ref floatMxN J)
        {
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < TankInsModel.N; j++) J[i, j] = 0f;
            for (int i = 0; i < 3; i++) J[i, TankInsModel.Pos + i] = 1f;
        }
    }

    /// <summary>
    /// The ground under the hull as the lidar sees it, in HULL AXES — no pose, no world frame, so
    /// nothing about it is circular on the estimate that later reads it.
    /// </summary>
    public struct GroundPlane
    {
        /// <summary>Ground normal in hull axes, unit, pointing up out of the surface.</summary>
        public float3 Normal;

        /// <summary>Perpendicular distance from the lidar origin to the plane, metres.</summary>
        public float Clearance;

        /// <summary>Beams that found ground this step.</summary>
        public int Returns;

        /// <summary>Beams that agreed with the fitted plane. Well under <see cref="Returns"/> means the
        /// fan is straddling a feature and part of it is ranging a different surface.</summary>
        public int Inliers;

        /// <summary>False when the fit was refused; the caller should hold its last plane.</summary>
        public bool Valid;
    }

    /// <summary>
    /// Fits the ground plane to the lidar returns with <c>Fit.plane</c>.
    ///
    /// This is what separates HULL TILT from TERRAIN SLOPE. A ride-height-only estimate reads a corner
    /// height difference the same way whether the hull is tilted or the ground is sloping, and levels
    /// the hull to the ground. Here the plane gives the ground's own orientation in hull axes, the
    /// filter gives the hull's orientation in world axes from gravity and the field, and the two
    /// together give both quantities separately: the hull can be held level while the terrain rolls
    /// underneath it.
    /// </summary>
    public static class GroundFit
    {
        /// <summary>Beams that must return before a fit is attempted.</summary>
        public const int MinReturns = 8;

        /// <summary>
        /// How far from the ground plane a beam may land and still belong to it, metres. Several
        /// times the range noise and far below a terrain feature, so it separates "the ground under
        /// the hull" from "the floor of the escarpment".
        /// </summary>
        public const float InlierBand = 0.15f;

        /// <summary>
        /// Huber transition residual for the polish, metres — the scale at which a surviving beam
        /// stops being weighted as noise.
        /// </summary>
        public const float HuberScale = 0.08f;

        /// <summary>Consensus draws per fit, and the seed that makes them repeatable.</summary>
        public const int RansacIter = 48;
        public const uint RansacSeed = 0x9E3779B1u;

        /// <summary>
        /// Fits the plane through the beams that returned. <paramref name="ranges"/> carries
        /// <see cref="TankSensorRig.NoReturn"/> for a beam that found nothing, which is DROPPED rather
        /// than treated as a range — a saturated miss would pull the plane by metres.
        /// Allocates <c>Allocator.Temp</c> only.
        ///
        /// CONSENSUS FIRST, then a robust polish. When the fan straddles the escarpment lip or the
        /// wall, a quarter of the beams are ranging a DIFFERENT surface metres away, and a reweighted
        /// single fit can only walk downhill from a start that contamination has already moved — it
        /// settles between the two surfaces at an angle that is not either of them. <c>Fit.ransac</c>
        /// picks the majority surface by consensus instead, and <c>Fit.plane</c> under a Huber loss
        /// then polishes the beams that agreed with it.
        /// </summary>
        public static GroundPlane Plane(NativeArray<float> ranges, NativeArray<float3> dirs, float3 origin)
        {
            var g = new GroundPlane { Normal = new float3(0f, 1f, 0f) };

            for (int k = 0; k < ranges.Length; k++)
                if (ranges[k] > 0f) g.Returns++;

            if (g.Returns < MinReturns) return g;

            var pts = new NativeArray<float3>(g.Returns, Allocator.Temp);
            int n = 0;
            for (int k = 0; k < ranges.Length; k++)
                if (ranges[k] > 0f) pts[n++] = origin + dirs[k] * ranges[k];

            var model = new Fit.floatPlane();
            RansacInfo info = Fit.ransac(pts, ref model, InlierBand, RansacIter, RansacSeed);
            if (!info.found) { pts.Dispose(); return g; }

            float3 centroid = model.Point, normal = model.Normal;

            if (info.inliers >= MinReturns && info.inliers < n)
            {
                var kept = new NativeArray<float3>(info.inliers, Allocator.Temp);
                int m = 0;
                for (int i = 0; i < n && m < info.inliers; i++)
                    if (model.Distance(pts[i]) <= InlierBand) kept[m++] = pts[i];

                var loss = new floatHuberLoss(HuberScale);
                if (m >= MinReturns && Fit.plane(kept, in loss, out float3 c, out float3 nrm))
                {
                    centroid = c; normal = nrm;
                }
                kept.Dispose();
            }
            pts.Dispose();

            // The fitted normal's sign is arbitrary; the ground is below the sensor, so the outward
            // normal is the one pointing up in hull axes.
            if (normal.y < 0f) normal = -normal;
            if (math.abs(math.lengthsq(normal) - 1f) > 1e-2f) return g;

            g.Normal = normal;
            g.Clearance = math.dot(origin - centroid, normal);
            g.Inliers = info.inliers;
            g.Valid = true;
            return g;
        }
    }

    /// <summary>
    /// What the controller is allowed to see: the filter's estimate plus the sensed ground, and
    /// nothing measured off the rigid body.
    /// </summary>
    public struct TankEstimate
    {
        /// <summary>Estimated world position, m.</summary>
        public float3 Position;

        /// <summary>Estimated world velocity, m/s.</summary>
        public float3 Velocity;

        /// <summary>Estimated [roll, pitch, yaw], rad.</summary>
        public float3 Rpy;

        /// <summary>Estimated IMU biases in hull axes.</summary>
        public float3 AccelBias, GyroBias;

        /// <summary>Fitted ground normal in WORLD axes — the terrain slope, separated from hull tilt.</summary>
        public float3 GroundNormal;

        /// <summary>Perpendicular ride height above the fitted ground, m, and its rate.</summary>
        public float Clearance, ClearanceRate;

        /// <summary>Cosine of the hull's tilt from world up, for the gravity feedforward.</summary>
        public float TiltCos;

        /// <summary>Hull-frame velocity components the driver's damping terms are written against.</summary>
        public float ForwardSpeed, LateralSpeed;

        /// <summary>Bias-corrected hull yaw rate, rad/s.</summary>
        public float YawRate;

        /// <summary>Bias-corrected Euler rates for the two axes the hover loop regulates, rad/s.</summary>
        public float RollRate, PitchRate;

        /// <summary>Beams that returned this step, how many of them agreed with the fitted plane, and
        /// whether the fit was accepted at all.</summary>
        public int LidarReturns, LidarInliers;
        public bool GroundValid;

        /// <summary>Which measurement streams updated the filter this step.</summary>
        public bool TiltFix, MagFix, GpsFix;

        /// <summary>Noise the gravity reference was trusted at this step — its base value while the
        /// hull is coasting, several times that under a manoeuvre.</summary>
        public float TiltSigma;

        /// <summary>Steps since the last accepted position fix.</summary>
        public int StepsSinceGps;
    }

    /// <summary>
    /// One fixed step of SENSING and ESTIMATION, upstream of the control law.
    ///
    /// Takes ground truth and the raw ranges the rays came back with, corrupts them into what each
    /// simulated sensor would actually report, runs the 15-state EKF over them, fits the ground plane
    /// to the lidar, and writes the estimate plus the six-state hover vector
    /// <see cref="HoverTankMPCStepJob"/> consumes. Truth enters HERE and nowhere downstream.
    ///
    /// The IMU drives <c>Kalman.ekfPredict</c> every step; the other streams call
    /// <c>Kalman.ekfUpdate</c> on their own periods, so a step may run zero, one or three updates
    /// against a single prediction. That multi-rate mix is what puts the sawtooth in the position
    /// error: the inertial solution walks away between beacon fixes and is pulled back by each one.
    ///
    /// Caller must RunByRef and copy <see cref="Kf"/> and <see cref="Noise"/> back — both carry state
    /// the job advances (the covariance, and the random stream).
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct TankEstimatorJob : IJob
    {
        /// <summary>What the vehicle actually is. Read by the sensor simulation only.</summary>
        public TankTruth Truth;

        /// <summary>True IMU biases, hull axes. The filter is not told these; it has to find them.</summary>
        public float3 TrueAccelBias, TrueGyroBias;

        /// <summary>Beam directions in hull axes and the mount they fire from.</summary>
        [ReadOnly] public NativeArray<float3> LidarDirs;
        public float3 LidarOrigin;

        /// <summary>True ranges in, reported (noisy) ranges out. <see cref="TankSensorRig.NoReturn"/> passes through.</summary>
        [ReadOnly] public NativeArray<float> LidarTrue;
        public NativeArray<float> LidarSensed;

        /// <summary>Proximity ranger truth in, reported ranges out. Not read by the control law yet.</summary>
        [ReadOnly] public NativeArray<float> ProxTrue;
        public NativeArray<float> ProxSensed;

        /// <summary>Noise generators, pre-factored. Carried back: the random stream advances.</summary>
        public TankSensorNoise Noise;
        public TankSensorSpec Spec;

        /// <summary>Filter state. Carried back: x and P advance.</summary>
        public floatKFState Kf;

        /// <summary>Process noise (15x15) and the two fixed measurement covariances (3x3 each). The
        /// gravity reference sizes its own per sample, so it has none here.</summary>
        public floatMxN Q, RMag, RGps;

        /// <summary>[0] tilt [1] magnetometer [2] beacon, for whichever ran this step.</summary>
        public NativeArray<KFInfo> KfOut;

        /// <summary>The estimate, length 1.</summary>
        public NativeArray<TankEstimate> Out;

        /// <summary>[height error, height rate, roll, roll rate, pitch, pitch rate] for the hover LQR.</summary>
        public NativeArray<float> HoverState;

        /// <summary>Held ground plane, length 1 — what a refused fit falls back on.</summary>
        public NativeArray<GroundPlane> Ground;

        public float Dt, Gravity, TargetRideHeight;

        /// <summary>Fixed steps since the demo started; drives every sensor's own period.</summary>
        public int Step;

        /// <summary>Steps since the last accepted position fix, carried in and out.</summary>
        public NativeArray<int> GpsAge;

        public void Execute()
        {
            var est = new TankEstimate();

            // ---- IMU: bias + white noise on truth, then straight into the propagation ----
            Noise.DrawImu();
            var u = new floatN(6, Allocator.Temp);
            float3 fMeas = Truth.SpecificForce + TrueAccelBias
                         + new float3(Noise.Draw6[0], Noise.Draw6[1], Noise.Draw6[2]);
            float3 wMeas = Truth.AngularRate + TrueGyroBias
                         + new float3(Noise.Draw6[3], Noise.Draw6[4], Noise.Draw6[5]);
            TankInsModel.Write(ref u, 0, fMeas);
            TankInsModel.Write(ref u, 3, wMeas);

            var model = new TankInsModel { Dt = Dt, Gravity = Gravity };
            Kalman.ekfPredict(ref Kf, in model, in u, in Q);

            var z = new floatN(3, Allocator.Temp);
            float3 rpyNow = TankInsModel.Read(in Kf.x, TankInsModel.Att);
            float3x3 Rnow = Attitude.Matrix(rpyNow);
            float3 velBody = math.mul(math.transpose(Rnow), TankInsModel.Read(in Kf.x, TankInsModel.Vel));
            float3 rateNow = wMeas - TankInsModel.Read(in Kf.x, TankInsModel.GyroBias);

            // ---- gravity reference: the accelerometer levelled, when it is reading gravity ----
            // The CENTRIPETAL TERM is removed first. A vehicle turning flat at speed reads a lateral
            // specific force of |w x v| while the magnitude of gravity barely changes, so nothing that
            // looks only at magnitude can see it and the filter would lean the whole hull into every
            // turn. Both factors are already estimated, so subtracting w x v costs one cross product
            // and takes the largest systematic error out of the reading.
            //
            // What is left — a surge, a brake, a kick — is handled by SIZING THE READING'S OWN NOISE
            // from how far it disagrees with gravity, in both magnitude and direction. A hard manoeuvre
            // then weighs almost nothing while a persistent lean, which disagrees just as much but does
            // not go away, is still corrected sample after sample. A hard accept/reject gate would
            // instead lock out the one measurement that could recover a large attitude error.
            //
            // The bias correction uses the filter's own estimate, which is what keeps a turn-on bias
            // from being read as a permanent lean.
            float3 corrected = fMeas - TankInsModel.Read(in Kf.x, TankInsModel.AccelBias)
                             - math.cross(rateNow, velBody);
            float tiltMag = math.length(corrected);
            float3 tiltDir = tiltMag > 1e-3f ? corrected / tiltMag : new float3(0f, 1f, 0f);
            float3 upBody = math.mul(math.transpose(Rnow), new float3(0f, 1f, 0f));
            float disagree = math.abs(tiltMag - Gravity) / Gravity + math.length(tiltDir - upBody);
            est.TiltSigma = Spec.tiltSigma + Spec.tiltSlack * disagree;

            if (tiltMag > 1e-3f && Step % math.max(Spec.tiltPeriod, 1) == 0)
            {
                var RTilt = new floatMxN(3, 3, Allocator.Temp);
                for (int i = 0; i < 3; i++) RTilt[i, i] = est.TiltSigma * est.TiltSigma;

                z[0] = tiltDir.x; z[1] = tiltDir.y; z[2] = tiltDir.z;
                var up = new TankVectorMeasurement { Reference = new float3(0f, 1f, 0f) };
                KfOut[0] = Kalman.ekfUpdate(ref Kf, in up, in RTilt, in z);
                RTilt.Dispose();
                est.TiltFix = true;
            }

            // ---- magnetometer: the field in hull axes ----
            if (Step % math.max(Spec.magPeriod, 1) == 0)
            {
                float3 field = Spec.MagField();
                Noise.DrawMag();
                float3 body = Truth.ToBody(field)
                            + new float3(Noise.Draw3[0], Noise.Draw3[1], Noise.Draw3[2]);
                z[0] = body.x; z[1] = body.y; z[2] = body.z;
                var mag = new TankVectorMeasurement { Reference = field };
                KfOut[1] = Kalman.ekfUpdate(ref Kf, in mag, in RMag, in z);
                est.MagFix = true;
            }

            // ---- beacon: absolute position, slow ----
            int age = GpsAge[0] + 1;
            if (Step % math.max(Spec.gpsPeriod, 1) == 0)
            {
                Noise.DrawGps();
                float3 fix = Truth.Position + new float3(Noise.Draw3[0], Noise.Draw3[1], Noise.Draw3[2]);
                z[0] = fix.x; z[1] = fix.y; z[2] = fix.z;
                var beacon = new TankPositionMeasurement();
                KfOut[2] = Kalman.ekfUpdate(ref Kf, in beacon, in RGps, in z);
                est.GpsFix = true;
                age = 0;
            }
            GpsAge[0] = age;
            est.StepsSinceGps = age;

            z.Dispose(); u.Dispose();

            // ---- ranging sensors: correlated range noise, misses passed through as misses ----
            Noise.DrawLidar();
            for (int k = 0; k < LidarTrue.Length; k++)
                LidarSensed[k] = LidarTrue[k] > 0f
                    ? math.max(LidarTrue[k] + Noise.Draw25[k], 0f)
                    : TankSensorRig.NoReturn;

            Noise.DrawProx();
            for (int k = 0; k < ProxTrue.Length; k++)
                ProxSensed[k] = ProxTrue[k] > 0f
                    ? math.max(ProxTrue[k] + Noise.Draw4[k], 0f)
                    : TankSensorRig.NoReturn;

            GroundPlane ground = GroundFit.Plane(LidarSensed, LidarDirs, LidarOrigin);
            est.LidarReturns = ground.Returns;
            est.LidarInliers = ground.Inliers;
            est.GroundValid = ground.Valid;
            if (ground.Valid) Ground[0] = ground;
            else ground = Ground[0];

            // ---- assemble what the controller may read ----
            float3 rpy = TankInsModel.Read(in Kf.x, TankInsModel.Att);
            float3x3 R = Attitude.Matrix(rpy);
            float3 v = TankInsModel.Read(in Kf.x, TankInsModel.Vel);
            float3 w = wMeas - TankInsModel.Read(in Kf.x, TankInsModel.GyroBias);
            float3 rates = Attitude.Rates(rpy, w);
            float3 normalWorld = math.mul(R, ground.Normal);

            est.Position = TankInsModel.Read(in Kf.x, TankInsModel.Pos);
            est.Velocity = v;
            est.Rpy = rpy;
            est.AccelBias = TankInsModel.Read(in Kf.x, TankInsModel.AccelBias);
            est.GyroBias = TankInsModel.Read(in Kf.x, TankInsModel.GyroBias);
            est.GroundNormal = normalWorld;
            est.Clearance = ground.Clearance;

            // Closing rate against a locally planar ground: the vehicle's own velocity resolved along
            // the ground normal. Over a slope the ground rises under the hull as it drives, and the
            // horizontal part of this term is exactly that.
            est.ClearanceRate = math.dot(v, normalWorld);
            est.TiltCos = R.c1.y;
            est.ForwardSpeed = math.dot(v, R.c2);
            est.LateralSpeed = math.dot(v, R.c0);
            est.YawRate = w.y;
            est.RollRate = rates.x;
            est.PitchRate = rates.y;
            Out[0] = est;

            // A refused fit falls back on the held plane, which the caller seeds at the setpoint: a
            // ground the lidar has never seen still has to hand the hover loop something, and the
            // setpoint is the one clearance that commands nothing.
            HoverState[0] = ground.Clearance - TargetRideHeight;
            HoverState[1] = est.ClearanceRate;
            HoverState[2] = rpy.x;
            HoverState[3] = rates.x;
            HoverState[4] = rpy.y;
            HoverState[5] = rates.y;
        }
    }
}
