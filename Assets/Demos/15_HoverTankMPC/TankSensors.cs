using BULA;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using Random = Unity.Mathematics.Random;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// One fixed step of hull ground truth. The SENSOR SIMULATION is the only consumer: a simulated
    /// sensor has to be told what it is looking at, and the UI reads it to plot estimate error.
    /// Nothing on the control path may touch it — see <see cref="TankEstimatorJob"/> for what the
    /// controller is allowed to see instead.
    /// </summary>
    public struct TankTruth
    {
        /// <summary>Hull origin in world space, metres.</summary>
        public float3 Position;

        /// <summary>Hull velocity in world space, m/s.</summary>
        public float3 Velocity;

        /// <summary>Hull axes in world space — the columns of the world-from-body rotation.</summary>
        public float3 Right, Up, Fwd;

        /// <summary>Hull-frame specific force, m/s²: what an ideal accelerometer reads (a - g).</summary>
        public float3 SpecificForce;

        /// <summary>Hull-frame angular rate, rad/s.</summary>
        public float3 AngularRate;

        /// <summary>Hull-frame components of a world vector.</summary>
        public float3 ToBody(float3 world)
            => new float3(math.dot(world, Right), math.dot(world, Up), math.dot(world, Fwd));
    }

    /// <summary>
    /// Noise levels, sampling rates and error sources of the simulated sensor suite — the inspector
    /// face of "what the tank can know about itself".
    ///
    /// The IMU numbers are deliberately those of a CHEAP MEMS part. A good one would let the strapdown
    /// solution coast between position fixes with an error far below the beacon's own noise, and the
    /// multi-rate signature the demo exists to show — drift out, snap back — would not be visible.
    ///
    /// Rates are in FIXED STEPS between samples, so they follow <c>Time.fixedDeltaTime</c>.
    /// </summary>
    [System.Serializable]
    public struct TankSensorSpec
    {
        [Tooltip("Accelerometer white noise, m/s^2 per axis (1 sigma).")]
        [Range(0.01f, 2f)] public float accelNoise;

        [Tooltip("Rate gyro white noise, rad/s per axis (1 sigma).")]
        [Range(0.001f, 0.2f)] public float gyroNoise;

        [Tooltip("Accelerometer turn-on bias, m/s^2. The filter has to learn this: it is not told.")]
        [Range(0f, 1f)] public float accelBias;

        [Tooltip("Rate gyro turn-on bias, rad/s. Also learned, not told.")]
        [Range(0f, 0.1f)] public float gyroBias;

        [Tooltip("Magnetometer noise per axis, in units of the field (1 sigma).")]
        [Range(0.001f, 0.2f)] public float magNoise;

        [Tooltip("Field elevation below horizontal, degrees. A near-horizontal field is what makes heading observable; a near-vertical one would leave yaw as the poorly-seen axis.")]
        [Range(0f, 70f)] public float magDipDeg;

        [Tooltip("Lidar range noise, metres (1 sigma).")]
        [Range(0.005f, 0.5f)] public float lidarNoise;

        [Tooltip("Range-error correlation between neighbouring beams. 0 makes the 25 beams independent.")]
        [Range(0f, 0.9f)] public float lidarCorrelation;

        [Tooltip("Beacon horizontal position noise, metres (1 sigma).")]
        [Range(0.1f, 10f)] public float gpsNoiseXZ;

        [Tooltip("Beacon vertical position noise, metres (1 sigma). Worse than horizontal, as it is for the real thing.")]
        [Range(0.1f, 15f)] public float gpsNoiseY;

        [Tooltip("Proximity ranger noise, metres (1 sigma).")]
        [Range(0.005f, 0.5f)] public float proxNoise;

        [Tooltip("Fixed steps between position fixes. Large is the point: this is the slow sensor.")]
        [Range(1, 200)] public int gpsPeriod;

        [Tooltip("Fixed steps between magnetometer samples.")]
        [Range(1, 30)] public int magPeriod;

        [Tooltip("Fixed steps between gravity-reference (levelling) samples.")]
        [Range(1, 30)] public int tiltPeriod;

        [Tooltip("Direction noise of the gravity reference at rest, 1 sigma (it measures a unit vector). Kept above the accelerometer's own noise over g, because the same reading already drives the prediction.")]
        [Range(0.005f, 0.5f)] public float tiltSigma;

        [Tooltip("How much a reading that does not look like gravity inflates its own noise. 0 trusts every sample equally, which lets a hard manoeuvre lean the whole hull.")]
        [Range(0f, 6f)] public float tiltSlack;

        public static TankSensorSpec Default => new TankSensorSpec
        {
            accelNoise = 0.35f,
            gyroNoise = 0.012f,
            accelBias = 0.18f,
            gyroBias = 0.008f,
            magNoise = 0.02f,
            magDipDeg = 20f,
            lidarNoise = 0.03f,
            lidarCorrelation = 0.35f,
            gpsNoiseXZ = 1.5f,
            gpsNoiseY = 2.5f,
            proxNoise = 0.05f,
            gpsPeriod = 50,
            magPeriod = 3,
            tiltPeriod = 2,
            tiltSigma = 0.05f,
            tiltSlack = 1.5f,
        };

        /// <summary>
        /// Unit magnetic field in world axes: north is +z, tipped <see cref="magDipDeg"/> below
        /// horizontal.
        /// </summary>
        public float3 MagField()
        {
            math.sincos(math.radians(magDipDeg), out float s, out float c);
            return new float3(0f, -s, c);
        }
    }

    /// <summary>
    /// The 5x5 forward-biased lidar fan, as unit directions in hull axes.
    ///
    /// One grid serves both jobs the demo needs from a downward look: the ground plane under the hull
    /// (ride height and terrain slope, see <see cref="GroundFit"/>) and the terrain ahead. Biasing the
    /// whole fan forward is what buys the second without a second sensor.
    /// </summary>
    public static class LidarGrid
    {
        /// <summary>Beams per axis.</summary>
        public const int Side = 5;

        /// <summary>Total beams.</summary>
        public const int Rays = Side * Side;

        /// <summary>Half-angle of the fan on either axis, degrees.</summary>
        public const float HalfFanDeg = 28f;

        /// <summary>Tilt of the whole fan toward hull +z, degrees.</summary>
        public const float ForwardBiasDeg = 18f;

        /// <summary>
        /// Fills <paramref name="dirs"/> (length <see cref="Rays"/>) with the beam directions in hull
        /// axes, straight down at the fan centre. Beam (i, j) is at column i, row j of the grid.
        /// </summary>
        public static void Directions(NativeArray<float3> dirs)
        {
            for (int j = 0; j < Side; j++)
                for (int i = 0; i < Side; i++)
                {
                    float u = (i - (Side - 1) * 0.5f) / ((Side - 1) * 0.5f);   // -1 .. +1 across
                    float v = (j - (Side - 1) * 0.5f) / ((Side - 1) * 0.5f);   // -1 .. +1 along

                    float ax = math.radians(HalfFanDeg) * u;
                    float az = math.radians(HalfFanDeg) * v + math.radians(ForwardBiasDeg);

                    math.sincos(ax, out float sx, out float cx);
                    math.sincos(az, out float sz, out float cz);
                    dirs[j * Side + i] = math.normalize(new float3(sx * cz, -cx * cz, sz));
                }
        }
    }

    /// <summary>
    /// The four proximity rangers, as unit directions in hull axes: forward, back, left, right.
    /// Wired and displayed; the controller does not read them yet.
    /// </summary>
    public static class ProximityRig
    {
        public const int Rays = 4;

        public static readonly string[] Names = { "fwd", "back", "left", "right" };

        public static void Directions(NativeArray<float3> dirs)
        {
            dirs[0] = new float3(0f, 0f, 1f);
            dirs[1] = new float3(0f, 0f, -1f);
            dirs[2] = new float3(-1f, 0f, 0f);
            dirs[3] = new float3(1f, 0f, 0f);
        }

        /// <summary>
        /// Where each ranger sits, in hull axes: on the hull face it looks out of, cleared by
        /// <paramref name="pad"/> metres. A ray starting inside the hull collider would report the
        /// world beyond it, so the offset is what makes the reading mean "distance to the obstacle".
        /// </summary>
        public static void Origins(NativeArray<float3> origins, float halfWidth, float halfLength, float pad)
        {
            origins[0] = new float3(0f, 0f, halfLength + pad);
            origins[1] = new float3(0f, 0f, -halfLength - pad);
            origins[2] = new float3(-halfWidth - pad, 0f, 0f);
            origins[3] = new float3(halfWidth + pad, 0f, 0f);
        }
    }

    /// <summary>
    /// Every sensor's noise generator, held together: one Cholesky factor per covariance plus the
    /// zero means and scratch the draws need, so <c>Rand.multivariateNormalInPlace</c> can be called
    /// per step without allocating and WITHOUT re-factoring — the factor is the expensive part and it
    /// is computed once, in <see cref="Build"/>.
    ///
    /// <see cref="Rng"/> advances on every draw, so a caller running this inside an <c>IJob</c> must
    /// copy the struct back after the run or the stream restarts each step.
    /// </summary>
    public struct TankSensorNoise
    {
        /// <summary>Lower Cholesky factors of each sensor's covariance.</summary>
        public floatMxN LImu, LMag, LLidar, LGps, LProx;

        /// <summary>Zero means, one per draw size.</summary>
        public floatN Mean3, Mean4, Mean6, Mean25;

        /// <summary>Draw destinations, one per size.</summary>
        public floatN Draw3, Draw4, Draw6, Draw25;

        /// <summary>N(0,1) scratch, one per size.</summary>
        public floatN Z3, Z4, Z6, Z25;

        /// <summary>Caller-owned stream. Mutated by every draw.</summary>
        public Random Rng;

        /// <summary>False if any covariance failed to factor — no sensor may then be sampled.</summary>
        public bool Factored;

        /// <summary>
        /// Factors every sensor covariance once and allocates the draw buffers.
        /// Covariances are diagonal except the lidar's, whose beams share a range error that decays
        /// with grid separation (a separable AR(1), so it is positive definite for any correlation
        /// below 1).
        /// </summary>
        public static TankSensorNoise Build(in TankSensorSpec spec, uint seed, Allocator allocator)
        {
            var n = new TankSensorNoise
            {
                LImu = new floatMxN(6, 6, allocator),
                LMag = new floatMxN(3, 3, allocator),
                LLidar = new floatMxN(LidarGrid.Rays, LidarGrid.Rays, allocator),
                LGps = new floatMxN(3, 3, allocator),
                LProx = new floatMxN(ProximityRig.Rays, ProximityRig.Rays, allocator),

                Mean3 = new floatN(3, allocator),
                Mean4 = new floatN(ProximityRig.Rays, allocator),
                Mean6 = new floatN(6, allocator),
                Mean25 = new floatN(LidarGrid.Rays, allocator),

                Draw3 = new floatN(3, allocator),
                Draw4 = new floatN(ProximityRig.Rays, allocator),
                Draw6 = new floatN(6, allocator),
                Draw25 = new floatN(LidarGrid.Rays, allocator),

                Z3 = new floatN(3, allocator),
                Z4 = new floatN(ProximityRig.Rays, allocator),
                Z6 = new floatN(6, allocator),
                Z25 = new floatN(LidarGrid.Rays, allocator),

                Rng = new Random(seed == 0u ? 1u : seed),
            };

            var imu = new floatMxN(6, 6, Allocator.Temp);
            for (int i = 0; i < 3; i++) imu[i, i] = spec.accelNoise * spec.accelNoise;
            for (int i = 3; i < 6; i++) imu[i, i] = spec.gyroNoise * spec.gyroNoise;

            var mag = new floatMxN(3, 3, Allocator.Temp);
            for (int i = 0; i < 3; i++) mag[i, i] = spec.magNoise * spec.magNoise;

            var gps = new floatMxN(3, 3, Allocator.Temp);
            gps[0, 0] = spec.gpsNoiseXZ * spec.gpsNoiseXZ;
            gps[1, 1] = spec.gpsNoiseY * spec.gpsNoiseY;
            gps[2, 2] = spec.gpsNoiseXZ * spec.gpsNoiseXZ;

            var prox = new floatMxN(ProximityRig.Rays, ProximityRig.Rays, Allocator.Temp);
            for (int i = 0; i < ProximityRig.Rays; i++) prox[i, i] = spec.proxNoise * spec.proxNoise;

            var lidar = new floatMxN(LidarGrid.Rays, LidarGrid.Rays, Allocator.Temp);
            float rho = math.clamp(spec.lidarCorrelation, 0f, 0.9f);
            float lv = spec.lidarNoise * spec.lidarNoise;
            for (int a = 0; a < LidarGrid.Rays; a++)
                for (int b = 0; b < LidarGrid.Rays; b++)
                {
                    int di = math.abs(a % LidarGrid.Side - b % LidarGrid.Side);
                    int dj = math.abs(a / LidarGrid.Side - b / LidarGrid.Side);
                    lidar[a, b] = lv * math.pow(rho, di + dj);
                }

            n.Factored = CHO.decomp(in imu, ref n.LImu).Solved
                       & CHO.decomp(in mag, ref n.LMag).Solved
                       & CHO.decomp(in gps, ref n.LGps).Solved
                       & CHO.decomp(in prox, ref n.LProx).Solved
                       & CHO.decomp(in lidar, ref n.LLidar).Solved;

            imu.Dispose(); mag.Dispose(); gps.Dispose(); prox.Dispose(); lidar.Dispose();
            return n;
        }

        /// <summary>Draws one 6-vector of IMU error (3 accelerometer axes then 3 gyro axes).</summary>
        public void DrawImu() => Rand.multivariateNormalInPlace(ref Rng, in LImu, in Mean6, ref Draw6, ref Z6);

        /// <summary>Draws one 3-vector of magnetometer error, in units of the field.</summary>
        public void DrawMag() => Rand.multivariateNormalInPlace(ref Rng, in LMag, in Mean3, ref Draw3, ref Z3);

        /// <summary>Draws one 3-vector of beacon position error, metres. Shares Draw3 with the magnetometer.</summary>
        public void DrawGps() => Rand.multivariateNormalInPlace(ref Rng, in LGps, in Mean3, ref Draw3, ref Z3);

        /// <summary>Draws one correlated 25-vector of lidar range error, metres.</summary>
        public void DrawLidar() => Rand.multivariateNormalInPlace(ref Rng, in LLidar, in Mean25, ref Draw25, ref Z25);

        /// <summary>Draws one 4-vector of proximity range error, metres.</summary>
        public void DrawProx() => Rand.multivariateNormalInPlace(ref Rng, in LProx, in Mean4, ref Draw4, ref Z4);

        public void Dispose()
        {
            if (LImu.IsCreated) LImu.Dispose();
            if (LMag.IsCreated) LMag.Dispose();
            if (LLidar.IsCreated) LLidar.Dispose();
            if (LGps.IsCreated) LGps.Dispose();
            if (LProx.IsCreated) LProx.Dispose();

            if (Mean3.IsCreated) Mean3.Dispose();
            if (Mean4.IsCreated) Mean4.Dispose();
            if (Mean6.IsCreated) Mean6.Dispose();
            if (Mean25.IsCreated) Mean25.Dispose();

            if (Draw3.IsCreated) Draw3.Dispose();
            if (Draw4.IsCreated) Draw4.Dispose();
            if (Draw6.IsCreated) Draw6.Dispose();
            if (Draw25.IsCreated) Draw25.Dispose();

            if (Z3.IsCreated) Z3.Dispose();
            if (Z4.IsCreated) Z4.Dispose();
            if (Z6.IsCreated) Z6.Dispose();
            if (Z25.IsCreated) Z25.Dispose();
        }
    }

    /// <summary>
    /// The ranging half of the sensor suite: it fires rays against the world collider, which needs
    /// <c>UnityEngine.Physics</c> and therefore the main thread.
    ///
    /// Rays leave the TRUE hull pose because a ranger is bolted to the hull — where the beam actually
    /// goes is a fact about the vehicle, not about what it believes. What leaves this class is a range
    /// per beam in the SENSOR's own frame; turning those into anything world-referenced is the
    /// estimator's job and uses the estimate.
    /// </summary>
    public static class TankSensorRig
    {
        /// <summary>A beam that found nothing inside its range, as reported in the range array.</summary>
        public const float NoReturn = -1f;

        /// <summary>
        /// Fires one ray per direction from <paramref name="originLocal"/> (hull axes) and writes each
        /// range into <paramref name="ranges"/>, or <see cref="NoReturn"/> where nothing was hit.
        ///
        /// A miss is FLAGGED, not saturated to <paramref name="rayLength"/>: over terrain with a
        /// drop-off, misses are ordinary, and a saturated range is a metres-long lie that an estimator
        /// consuming it would be dragged by. <see cref="GroundFit"/> drops flagged beams instead.
        ///
        /// Rotation only, no scale: the offsets are metric and stay that way even if the hull root is
        /// scaled later.
        /// </summary>
        public static void Range(Transform hull, float3 originLocal, NativeArray<float3> dirs,
                                 float rayLength, NativeArray<float> ranges)
        {
            Vector3 origin = hull.position + hull.rotation * new Vector3(originLocal.x, originLocal.y, originLocal.z);

            for (int k = 0; k < dirs.Length; k++)
            {
                float3 d = dirs[k];
                Vector3 world = hull.rotation * new Vector3(d.x, d.y, d.z);
                ranges[k] = Physics.Raycast(origin, world, out RaycastHit hit, rayLength)
                    ? hit.distance
                    : NoReturn;
            }
        }

        /// <summary>
        /// As <see cref="Range"/>, with a separate hull-frame origin per ray — what a set of rangers
        /// mounted on different faces needs.
        /// </summary>
        public static void Range(Transform hull, NativeArray<float3> originsLocal, NativeArray<float3> dirs,
                                 float rayLength, NativeArray<float> ranges)
        {
            for (int k = 0; k < dirs.Length; k++)
            {
                float3 o = originsLocal[k], d = dirs[k];
                Vector3 origin = hull.position + hull.rotation * new Vector3(o.x, o.y, o.z);
                Vector3 world = hull.rotation * new Vector3(d.x, d.y, d.z);
                ranges[k] = Physics.Raycast(origin, world, out RaycastHit hit, rayLength)
                    ? hit.distance
                    : NoReturn;
            }
        }
    }
}
