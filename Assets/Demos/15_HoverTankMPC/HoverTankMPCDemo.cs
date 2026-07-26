using System.Diagnostics;
using BULA;
using BULA.Control;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// Player-driven hover tank on FOUR TWO-AXIS GIMBALLED THRUSTERS, each with a pitch servo, a yaw
    /// servo and a throttle, flying over procedural <see cref="TerrainField"/>. Drop on an empty
    /// GameObject and press play. No turret: this demo is about the thruster rig.
    ///
    /// Mouse X turns the hull and mouse Y raises or lowers the commanded ride height, with the cursor
    /// captured; ESC releases it to the on-screen panel and back. W/S drive · Q/E strafe · A/D yaw as
    /// well · SPACE brakes forward speed, sideways speed and yaw rate. Hands off, weak idle damping
    /// settles the tank instead of letting it free-float.
    ///
    /// THE CONTROLLER NEVER READS THE RIGID BODY. Everything the control law is handed comes out of
    /// <see cref="TankEstimatorJob"/>: a 15-state strapdown EKF (position, velocity, attitude, and
    /// both IMU biases) driven by a simulated IMU, magnetometer and position beacon, plus a ground
    /// plane fitted to a 25-beam lidar. Truth is read in exactly two places — the sensor simulation,
    /// which has to be told what it is looking at, and the panel, which plots how far the estimate has
    /// drifted from it. The rays themselves leave the TRUE hull pose because a ranger is bolted to the
    /// hull; what comes back is a range in the sensor's own frame, and turning that into anything
    /// world-referenced uses the estimate.
    ///
    /// Every axis reaches the thrusters through one control-allocation solve:
    ///
    /// 1. A 6-state discrete LQR (ride-height error, closing rate, roll, roll rate, pitch, pitch rate)
    ///    read off the estimate, producing vertical/roll/pitch acceleration commands.
    /// 2. A control-allocation QP that turns those commands plus the driver's forward/strafe/yaw
    ///    demand — and the braking and idle-damping terms, which are wrench demands like everything
    ///    else rather than forces written onto the rigid body — into the 12 thruster controls, under
    ///    servo range/rate and thrust range/rate limits. See <see cref="GimbalAllocation"/>: 12
    ///    controls against 6 wrench components, so the rig is over-actuated and the solve is what
    ///    decides how the work is shared.
    ///
    /// Thrust is augmented near the ground by <see cref="GroundEffect"/>, per nozzle, from a downward
    /// ray at each exhaust plane — so a tilted hull is pushed harder on its low side. The same four
    /// factors scale the force applied to the rigid body AND the allocation's Jacobian, so the two
    /// agree exactly and the demanded wrench is delivered at any height. Their mean also enters the
    /// hover model's vertical input column, which softens the hover gain slightly near the ground; see
    /// <see cref="HoverTankMPCStepJob.Execute"/> for why that is a choice and not a measurement.
    ///
    /// HULL TILT AND TERRAIN SLOPE ARE SEPARATE QUANTITIES HERE, and the panel prints both. The
    /// attitude the hover loop regulates comes from gravity and the magnetic field, so the tank is
    /// held LEVEL across a hillside; the slope comes from the lidar's fitted plane, in hull axes, and
    /// only says which way the ground under it is running. A ride-height-only estimate cannot tell the
    /// two apart and levels the hull to the ground instead.
    ///
    /// The four THRUST MOUNTS sit on the side flanks at hull mid-height (x = ±halfWidth, y = 0,
    /// z = ±halfLength). A mount at y = 0 turns forward thrust into pure yaw, with no pitch moment for
    /// the hover loop to fight.
    ///
    /// Self-assembles terrain, hull and a chase camera in <see cref="Start"/> (sceneless, like the
    /// other demos), including one <see cref="ParticleSystem"/> plume per nozzle.
    /// </summary>
    public class HoverTankMPCDemo : MonoBehaviour
    {
        /// <summary>Commanded ride height range, metres. Mouse Y and the panel slider share it.</summary>
        public const float RideHeightMin = 0.5f, RideHeightMax = 6f;

        [Header("Hull")]
        [Range(300f, 5000f)] public float hullMass = 1500f;
        [Range(1f, 4f)] public float hullHalfWidth = 2f;
        [Range(1f, 6f)] public float hullHalfLength = 3f;
        [Range(0.5f, 2f)] public float hullHeight = 1f;
        [Range(RideHeightMin, RideHeightMax)] public float targetRideHeight = 2f;
        [Range(2f, 20f)] public float rayLength = 8f;

        [Header("Hover LQR")]
        [Range(500f, 6000f)] public float rollInertia = 2100f;
        [Range(500f, 8000f)] public float pitchInertia = 4600f;
        [Range(1f, 200f)] public float qHeight = 40f;
        [Range(0.1f, 50f)] public float qHeightRate = 6f;
        [Range(1f, 300f)] public float qTilt = 90f;
        [Range(0.1f, 50f)] public float qTiltRate = 8f;
        [Range(0.001f, 5f)] public float rThrust = 0.02f;   // LQR cost on commanded vertical accel (m/s^2), not force
        [Range(0.01f, 20f)] public float rTorque = 0.4f;    // LQR cost on commanded angular accel (rad/s^2), not torque

        [Header("Gimballed thrusters (allocated by QP)")]
        public GimbalSettings thrusters = GimbalSettings.Default;

        [Header("Sensors and state estimation")]
        public TankSensorSpec sensors = TankSensorSpec.Default;
        [Tooltip("Seed of the sensor noise stream. A fixed seed makes a play session repeatable.")]
        public uint sensorSeed = 0x5E45E1u;

        [Header("Ground effect")]
        [Tooltip("Off makes every thruster deliver exactly what it was commanded at any height, which is what the rig would feel with no ground under it.")]
        public bool groundEffect = true;
        [Tooltip("Effective nozzle radius R in the Cheeseman-Bennett model, metres. The augmentation is worth 1.07x at a nozzle height of R and 1.33x at R/2, so this sets the ride height band over which the effect is felt.")]
        [Range(0.5f, 4f)] public float nozzleRadius = 2f;

        [Header("Driver demand (routed through the allocation)")]
        [Range(1000f, 40000f)] public float driveForce = 9000f;
        [Tooltip("Peak sideways force, newtons. Lateral authority is bought out of the same thrust that carries the hull, so this stays under driveForce.")]
        [Range(1000f, 40000f)] public float strafeForce = 7000f;
        [Range(1000f, 40000f)] public float steerTorque = 9000f;
        [Tooltip("Peak braking force. Kept below driveForce so the brake cannot starve attitude control.")]
        [Range(1000f, 30000f)] public float brakeForce = 8000f;
        [Tooltip("Braking force per m/s of along-track speed, so the brake eases off as the tank slows.")]
        [Range(200f, 20000f)] public float brakeGain = 3000f;
        [Tooltip("Braking yaw moment per rad/s of yaw rate. Capped at steerTorque, so the brake can never out-demand the stick.")]
        [Range(500f, 30000f)] public float brakeYawGain = 12000f;
        [Tooltip("Steer command per unit of accumulated mouse X. A locked cursor gives unbounded deltas, so this wants to stay small.")]
        [Range(0.01f, 1f)] public float lookSensitivity = 0.12f;
        [Tooltip("Metres of commanded ride height per unit of mouse Y. Push forward to climb.")]
        [Range(0.01f, 1f)] public float climbSensitivity = 0.15f;
        [Tooltip("Idle forward damping, newtons per m/s. Fades out as W/S is pressed; 0 lets the tank free-float.")]
        [Range(0f, 5000f)] public float idleLinearGain = 1500f;
        [Tooltip("Idle yaw damping, newton-metres per rad/s. Fades out as steering appears; 0 lets the tank free-float.")]
        [Range(0f, 15000f)] public float idleAngularGain = 6500f;

        [Header("Chase camera")]
        [Range(3f, 30f)] public float camDistance = 12f;
        [Range(1f, 15f)] public float camHeight = 5f;
        [Range(1f, 30f)] public float camLag = 6f;

        /// <summary>
        /// Nozzle exit plane below its mount, metres — matches the nozzle mesh and its plume, and is
        /// where the ground-effect height is measured. Being a hull-local point strictly outside the
        /// hull collider is what lets the ground-effect ray fire from it without hitting the hull.
        /// </summary>
        const float NozzleExitDrop = 0.7f;

        /// <summary>Bias random-walk driving noise, m/s³ and rad/s² — how fast the filter lets the two
        /// IMU biases wander, which is what makes them learnable at all.</summary>
        const float AccelBiasWalk = 0.06f, GyroBiasWalk = 0.004f;

        /// <summary>Samples of estimate error kept for the panel's trace — 220 fixed steps.</summary>
        const int TraceLength = 220;

        static readonly string[] MountNames = { "FL", "FR", "BL", "BR" };
        static readonly Rect PanelRect = new Rect(10, 10, 580, 806);

        // self-assembled scene objects (Start)
        GameObject groundGO, hullGO, hullVisualGO;
        readonly Transform[] thrusterPivots = new Transform[4];
        readonly ParticleSystem[] plumes = new ParticleSystem[4];
        Material plumeMaterial;
        Texture2D plumeTexture;
        Camera chaseCam;
        Rigidbody rb;
        Vector3[] mountLocal;      // FL, FR, BL, BR thrust mounts on the side flanks
        float4 mountX, mountY, mountZ;
        float mountArm;            // lever arm the torque residual scale is measured against
        float4 thrusterHealth = new float4(1f);
        Vector3 spawnPosition;
        float lookX;               // mouse X accumulated since the last fixed step
        bool mouseCaptured = true; // driving mode; ESC releases the cursor to the panel
        float4 nozzleHeights;      // each nozzle exit's range to the ground, metres
        bool4 nozzleReturn;        // whether each nozzle ray found ground this step
        float lastSteer, lastStrafe;
        bool lastBrake;

        // hover loop buffers (persistent — never allocated inside the job)
        floatMxN hoverK;
        floatLQRState hoverLqr;
        NativeArray<float> hoverState;    // [height err, closing rate, roll, roll rate, pitch, pitch rate]
        NativeArray<float> hoverOut;      // [0] iters [1] converged [2] residual [3] rank-deficient

        // allocation buffers
        NativeArray<float> controls;      // 4 pitch angles, 4 yaw angles (rad), then 4 throttles
        NativeArray<QPInfo> allocOut;
        NativeArray<float> wrenchOut;     // 6 demanded then 6 achieved (N, N*m)
        NativeArray<float> groundOut;     // per-nozzle thrust augmentation this step

        // sensing and estimation buffers (persistent)
        NativeArray<float3> lidarDirs, proxDirs, proxOrigins;
        NativeArray<float> lidarTrue, lidarSensed, proxTrue, proxSensed;
        NativeArray<TankEstimate> estimateOut;
        NativeArray<GroundPlane> groundHold;
        NativeArray<KFInfo> kfOut;
        NativeArray<int> gpsAge;
        floatKFState kf;
        floatMxN kfQ, kfRMag, kfRGps;
        TankSensorNoise noise;
        float3 lidarOrigin;
        float3 trueAccelBias, trueGyroBias;
        int stepCount;

        // truth, captured once per fixed step for the sensor simulation and the error trace
        TankTruth truth;
        Vector3 prevVelocity;
        TankEstimate estimate;

        // error trace (UI only — this is the one place estimate and truth are compared)
        readonly float[] traceHoriz = new float[TraceLength];
        readonly float[] traceVert = new float[TraceLength];
        readonly bool[] traceFix = new bool[TraceLength];
        int traceHead;
        float3 posError, attError;
        float clearanceError, trueSlopeDeg;

        bool hoverDivergedLogged, allocFailedLogged, estimatorFailedLogged;
        float estMs, ctrlMs;

        void Start()
        {
            BuildScene();

            hoverK = new floatMxN(3, 6, Allocator.Persistent);
            hoverLqr = new floatLQRState(6, Allocator.Persistent);
            hoverState = new NativeArray<float>(6, Allocator.Persistent);
            hoverOut = new NativeArray<float>(4, Allocator.Persistent);

            controls = new NativeArray<float>(GimbalAllocation.ControlCount, Allocator.Persistent);
            allocOut = new NativeArray<QPInfo>(1, Allocator.Persistent);
            wrenchOut = new NativeArray<float>(2 * GimbalAllocation.WrenchRows, Allocator.Persistent);
            groundOut = new NativeArray<float>(GimbalAllocation.Thrusters, Allocator.Persistent);
            for (int i = 0; i < 4; i++)
                groundOut[i] = 1f;   // out of ground effect until the first step measures otherwise
            nozzleHeights = new float4(rayLength);

            lidarDirs = new NativeArray<float3>(LidarGrid.Rays, Allocator.Persistent);
            lidarTrue = new NativeArray<float>(LidarGrid.Rays, Allocator.Persistent);
            lidarSensed = new NativeArray<float>(LidarGrid.Rays, Allocator.Persistent);
            LidarGrid.Directions(lidarDirs);
            // Just below the bottom face, so a beam leaves the hull instead of starting inside it.
            lidarOrigin = new float3(0f, -hullHeight * 0.5f - 0.05f, 0f);

            proxDirs = new NativeArray<float3>(ProximityRig.Rays, Allocator.Persistent);
            proxOrigins = new NativeArray<float3>(ProximityRig.Rays, Allocator.Persistent);
            proxTrue = new NativeArray<float>(ProximityRig.Rays, Allocator.Persistent);
            proxSensed = new NativeArray<float>(ProximityRig.Rays, Allocator.Persistent);
            ProximityRig.Directions(proxDirs);
            ProximityRig.Origins(proxOrigins, hullHalfWidth, hullHalfLength, 0.05f);

            estimateOut = new NativeArray<TankEstimate>(1, Allocator.Persistent);
            groundHold = new NativeArray<GroundPlane>(1, Allocator.Persistent);
            kfOut = new NativeArray<KFInfo>(3, Allocator.Persistent);
            gpsAge = new NativeArray<int>(1, Allocator.Persistent);
            kf = new floatKFState(TankInsModel.N, 3, Allocator.Persistent);
            kfQ = new floatMxN(TankInsModel.N, TankInsModel.N, Allocator.Persistent);
            kfRMag = new floatMxN(3, 3, Allocator.Persistent);
            kfRGps = new floatMxN(3, 3, Allocator.Persistent);

            noise = TankSensorNoise.Build(in sensors, sensorSeed, Allocator.Persistent);
            if (!noise.Factored)
                UnityEngine.Debug.LogError("HoverTankMPCDemo: a sensor covariance is not positive definite, noise is off");

            SeedEstimator();
            ResetControls();
        }

        void OnDestroy()
        {
            if (hoverK.IsCreated) hoverK.Dispose();
            hoverLqr.Dispose();
            if (hoverState.IsCreated) hoverState.Dispose();
            if (hoverOut.IsCreated) hoverOut.Dispose();

            if (controls.IsCreated) controls.Dispose();
            if (allocOut.IsCreated) allocOut.Dispose();
            if (wrenchOut.IsCreated) wrenchOut.Dispose();
            if (groundOut.IsCreated) groundOut.Dispose();

            if (lidarDirs.IsCreated) lidarDirs.Dispose();
            if (lidarTrue.IsCreated) lidarTrue.Dispose();
            if (lidarSensed.IsCreated) lidarSensed.Dispose();
            if (proxDirs.IsCreated) proxDirs.Dispose();
            if (proxOrigins.IsCreated) proxOrigins.Dispose();
            if (proxTrue.IsCreated) proxTrue.Dispose();
            if (proxSensed.IsCreated) proxSensed.Dispose();
            if (estimateOut.IsCreated) estimateOut.Dispose();
            if (groundHold.IsCreated) groundHold.Dispose();
            if (kfOut.IsCreated) kfOut.Dispose();
            if (gpsAge.IsCreated) gpsAge.Dispose();
            kf.Dispose();
            if (kfQ.IsCreated) kfQ.Dispose();
            if (kfRMag.IsCreated) kfRMag.Dispose();
            if (kfRGps.IsCreated) kfRGps.Dispose();
            noise.Dispose();

            if (plumeMaterial != null) Destroy(plumeMaterial);
            if (plumeTexture != null) Destroy(plumeTexture);
        }

        /// <summary>
        /// Puts the filter back where a cold start leaves it: the estimate at the SPAWN POSE, which is
        /// a design constant of the demo and not a reading off the rigid body, both IMU biases at zero
        /// so they have to be learned, and a covariance saying how little of that is trusted. Also
        /// draws the turn-on biases the simulated IMU will actually carry — the filter is never told
        /// them.
        /// </summary>
        void SeedEstimator()
        {
            for (int i = 0; i < TankInsModel.N; i++)
            {
                kf.x[i] = 0f;
                for (int j = 0; j < TankInsModel.N; j++) kf.P[i, j] = 0f;
            }
            kf.x[TankInsModel.Pos] = spawnPosition.x;
            kf.x[TankInsModel.Pos + 1] = spawnPosition.y;
            kf.x[TankInsModel.Pos + 2] = spawnPosition.z;

            for (int i = 0; i < 3; i++)
            {
                kf.P[TankInsModel.Pos + i, TankInsModel.Pos + i] = 4f;
                kf.P[TankInsModel.Vel + i, TankInsModel.Vel + i] = 1f;
                kf.P[TankInsModel.Att + i, TankInsModel.Att + i] = 0.01f;
                kf.P[TankInsModel.AccelBias + i, TankInsModel.AccelBias + i] = 0.25f;
                kf.P[TankInsModel.GyroBias + i, TankInsModel.GyroBias + i] = 2.5e-3f;
            }

            BuildFilterNoise(Time.fixedDeltaTime);

            groundHold[0] = new GroundPlane
            {
                Normal = new float3(0f, 1f, 0f),
                Clearance = targetRideHeight,
                Returns = 0,
                Valid = false,
            };
            gpsAge[0] = 0;
            stepCount = 0;

            uint biasSeed = sensorSeed * 2654435761u + 1u;
            var rng = new Unity.Mathematics.Random(biasSeed == 0u ? 7u : biasSeed);
            trueAccelBias = rng.NextFloat3Direction() * sensors.accelBias;
            trueGyroBias = rng.NextFloat3Direction() * sensors.gyroBias;

            estimate = default;
            for (int i = 0; i < TraceLength; i++) { traceHoriz[i] = 0f; traceVert[i] = 0f; traceFix[i] = false; }
            traceHead = 0;
        }

        /// <summary>
        /// Rebuilds the filter's process and measurement covariances from the current sensor sliders,
        /// so moving one in play mode is felt by the estimator and not only by the sensor.
        /// </summary>
        void BuildFilterNoise(float dt)
        {
            for (int i = 0; i < TankInsModel.N; i++)
                for (int j = 0; j < TankInsModel.N; j++) kfQ[i, j] = 0f;

            float qv = sensors.accelNoise * dt, qa = sensors.gyroNoise * dt;
            float qba = AccelBiasWalk * dt, qbg = GyroBiasWalk * dt;
            for (int i = 0; i < 3; i++)
            {
                kfQ[TankInsModel.Pos + i, TankInsModel.Pos + i] = 1e-6f;
                kfQ[TankInsModel.Vel + i, TankInsModel.Vel + i] = qv * qv;
                kfQ[TankInsModel.Att + i, TankInsModel.Att + i] = qa * qa;
                kfQ[TankInsModel.AccelBias + i, TankInsModel.AccelBias + i] = qba * qba;
                kfQ[TankInsModel.GyroBias + i, TankInsModel.GyroBias + i] = qbg * qbg;
            }

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    kfRMag[i, j] = 0f; kfRGps[i, j] = 0f;
                }
            for (int i = 0; i < 3; i++) kfRMag[i, i] = sensors.magNoise * sensors.magNoise;
            kfRGps[0, 0] = sensors.gpsNoiseXZ * sensors.gpsNoiseXZ;
            kfRGps[1, 1] = sensors.gpsNoiseY * sensors.gpsNoiseY;
            kfRGps[2, 2] = sensors.gpsNoiseXZ * sensors.gpsNoiseXZ;
        }

        void BuildScene()
        {
            groundGO = TerrainField.Build("HoverTankMPC_Terrain");

            spawnPosition = new Vector3(0f, TerrainField.Height(0f, 0f) + targetRideHeight + hullHeight * 0.5f, 0f);
            Vector3 hullSize = new Vector3(2f * hullHalfWidth, hullHeight, 2f * hullHalfLength);

            // The hull root carries NO scale: mounts and thruster pivots hang off it in metres, and the
            // render scale lives on the visual child only.
            hullGO = new GameObject("HoverTankMPC_Hull");
            hullGO.transform.position = spawnPosition;
            hullGO.AddComponent<BoxCollider>().size = hullSize;

            rb = hullGO.AddComponent<Rigidbody>();
            rb.mass = hullMass;
            rb.linearDamping = 0.05f;
            rb.angularDamping = 0.2f;
            rb.interpolation = RigidbodyInterpolation.Interpolate;

            hullVisualGO = GameObject.CreatePrimitive(PrimitiveType.Cube);
            hullVisualGO.name = "HoverTankMPC_HullVisual";
            hullVisualGO.transform.SetParent(hullGO.transform, worldPositionStays: false);
            hullVisualGO.transform.localScale = hullSize;
            hullVisualGO.GetComponent<Renderer>().material.color = new Color(0.25f, 0.55f, 0.3f);
            Destroy(hullVisualGO.GetComponent<Collider>());

            // Thrust mounts sit on the SIDE FLANKS at hull mid-height. y = 0 is the load-bearing part:
            // the moment of a purely forward thrust is then r x F = (0, -x*Fz, 0), pure yaw, so driving
            // no longer pitches the nose and the servos no longer have to trade angle to cancel it.
            mountLocal = new[]
            {
                new Vector3(-hullHalfWidth, 0f, +hullHalfLength),   // FL
                new Vector3(+hullHalfWidth, 0f, +hullHalfLength),   // FR
                new Vector3(-hullHalfWidth, 0f, -hullHalfLength),   // BL
                new Vector3(+hullHalfWidth, 0f, -hullHalfLength),   // BR
            };

            mountX = new float4(mountLocal[0].x, mountLocal[1].x, mountLocal[2].x, mountLocal[3].x);
            mountY = new float4(mountLocal[0].y, mountLocal[1].y, mountLocal[2].y, mountLocal[3].y);
            mountZ = new float4(mountLocal[0].z, mountLocal[1].z, mountLocal[2].z, mountLocal[3].z);
            mountArm = math.max(hullHalfWidth, hullHalfLength);

            BuildPlumeAssets();

            // One pivot per mount, rotated by that nozzle's two gimbal angles every step: local +y is
            // the thrust direction and local -y the exhaust, so the nozzle hanging below it swings the
            // way the allocation is steering the thrust.
            for (int i = 0; i < 4; i++)
            {
                var pivot = new GameObject($"HoverTankMPC_Thruster_{MountNames[i]}");
                pivot.transform.SetParent(hullGO.transform, worldPositionStays: false);
                pivot.transform.localPosition = mountLocal[i];
                thrusterPivots[i] = pivot.transform;

                var nozzle = GameObject.CreatePrimitive(PrimitiveType.Cube);
                nozzle.name = "Nozzle";
                nozzle.transform.SetParent(pivot.transform, worldPositionStays: false);
                nozzle.transform.localScale = new Vector3(0.35f, 0.7f, 0.35f);
                nozzle.transform.localPosition = new Vector3(0f, -0.35f, 0f);
                nozzle.GetComponent<Renderer>().material.color = new Color(0.15f, 0.15f, 0.18f);
                Destroy(nozzle.GetComponent<Collider>());

                plumes[i] = BuildPlume(pivot.transform);
            }

            // Reuse whatever camera and light the scene already has; only build one if there is none.
            // Terrain relief is only readable as shading, so an unlit scene would render as a silhouette.
            chaseCam = Camera.main != null ? Camera.main : FindFirstObjectByType<Camera>();
            if (chaseCam == null)
            {
                var camGO = new GameObject("HoverTankMPC_ChaseCamera");
                chaseCam = camGO.AddComponent<Camera>();
                if (FindFirstObjectByType<AudioListener>() == null) camGO.AddComponent<AudioListener>();
            }
            if (FindFirstObjectByType<Light>() == null)
            {
                var lightGO = new GameObject("HoverTankMPC_Sun");
                var sun = lightGO.AddComponent<Light>();
                sun.type = LightType.Directional;
                lightGO.transform.rotation = Quaternion.Euler(48f, -35f, 0f);
            }
            SnapCamera();
        }

        // Rotation only: the offsets are metric and must stay that way even if someone scales the hull
        // root later.
        Vector3 MountWorld(int i) => hullGO.transform.position + hullGO.transform.rotation * mountLocal[i];

        /// <summary>Nozzle i's exhaust plane in world space — where its ground-effect height is measured.</summary>
        Vector3 NozzleExitWorld(int i)
            => hullGO.transform.position
             + hullGO.transform.rotation * (mountLocal[i] + Vector3.down * NozzleExitDrop);

        void ResetControls()
        {
            float maxThrust = Mathf.Max(thrusters.maxThrust, 1f);
            float trim = Mathf.Clamp(
                hullMass * -Physics.gravity.y / (4f * maxThrust),
                Mathf.Clamp01(thrusters.minThrust / maxThrust), 1f);

            for (int i = 0; i < 4; i++)
            {
                controls[i] = 0f;              // pitch gimbal
                controls[4 + i] = 0f;          // yaw gimbal
                controls[8 + i] = trim;
            }
        }

        // A radial-falloff sprite and one shared unlit material for all four plumes. The demo is
        // sceneless, so there is no asset to reference: the shader is whichever of the candidates the
        // active render pipeline actually has, and a null one leaves the ParticleSystem's own default
        // material in place rather than failing the build.
        void BuildPlumeAssets()
        {
            const int size = 32;
            plumeTexture = new Texture2D(size, size, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
            var px = new Color[size * size];
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    float dx = (x + 0.5f) / size * 2f - 1f, dy = (y + 0.5f) / size * 2f - 1f;
                    float a = Mathf.Clamp01(1f - Mathf.Sqrt(dx * dx + dy * dy));
                    px[y * size + x] = new Color(1f, 1f, 1f, a * a);
                }
            plumeTexture.SetPixels(px);
            plumeTexture.Apply();

            // Sequential, not ??: UnityEngine.Object overloads == for its own null, which ?? bypasses.
            Shader sh = Shader.Find("Sprites/Default");
            if (sh == null) sh = Shader.Find("Universal Render Pipeline/Particles/Unlit");
            if (sh == null) sh = Shader.Find("Particles/Standard Unlit");
            if (sh == null) return;

            plumeMaterial = new Material(sh) { name = "HoverTankMPC_Plume", mainTexture = plumeTexture };
        }

        // World-space so the plume trails the moving hull instead of riding with it. The child is
        // rotated a quarter turn about x because a ParticleSystem emits along its own +z, and the
        // exhaust runs down the pivot's -y.
        ParticleSystem BuildPlume(Transform pivot)
        {
            var go = new GameObject("Plume");
            go.transform.SetParent(pivot, worldPositionStays: false);
            go.transform.localPosition = new Vector3(0f, -0.7f, 0f);
            go.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);

            var ps = go.AddComponent<ParticleSystem>();

            var main = ps.main;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.startLifetime = 0.35f;
            main.startSize = 0.55f;
            main.startSpeed = 0f;
            main.maxParticles = 64;
            main.gravityModifier = 0f;
            main.playOnAwake = true;

            var shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 8f;
            shape.radius = 0.16f;

            var emission = ps.emission;
            emission.rateOverTime = 0f;

            var sol = ps.sizeOverLifetime;
            sol.enabled = true;
            sol.size = new ParticleSystem.MinMaxCurve(1f, AnimationCurve.Linear(0f, 0.5f, 1f, 1.6f));

            // Everything else off: the plume is a readout of one number, not a VFX budget.
            var collision = ps.collision; collision.enabled = false;
            var trails = ps.trails; trails.enabled = false;
            var psNoise = ps.noise; psNoise.enabled = false;
            var lights = ps.lights; lights.enabled = false;

            var psRenderer = go.GetComponent<ParticleSystemRenderer>();
            psRenderer.renderMode = ParticleSystemRenderMode.Billboard;
            psRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            psRenderer.receiveShadows = false;
            if (plumeMaterial != null) psRenderer.sharedMaterial = plumeMaterial;

            ps.Play();
            return ps;
        }

        // The plume IS the commanded throttle: rate, exhaust speed and colour all read off it, so what
        // the allocation decided is visible without opening the panel.
        void UpdatePlume(int i, float throttle, bool live)
        {
            ParticleSystem ps = plumes[i];
            if (ps == null) return;

            float t = live ? Mathf.Clamp01(throttle) : 0f;

            var emission = ps.emission;
            emission.rateOverTime = 90f * t;

            var main = ps.main;
            main.startSpeed = 6f + 26f * t;
            main.startColor = Color.Lerp(new Color(0.35f, 0.65f, 1f, 0.30f),
                                         new Color(1f, 0.55f, 0.20f, 0.85f), t);
        }

        void OnEnable() => ApplyCursor(mouseCaptured);

        // Never leave the editor holding a captured cursor when play mode ends. mouseCaptured itself
        // is left alone, so re-enabling comes back in whichever mode the driver chose.
        void OnDisable() => ApplyCursor(false);

        void SetCapture(bool captured)
        {
            mouseCaptured = captured;
            ApplyCursor(captured);
        }

        void ApplyCursor(bool captured)
        {
            Cursor.lockState = captured ? CursorLockMode.Locked : CursorLockMode.None;
            Cursor.visible = !captured;

            // Locking recentres the cursor, which shows up as one large delta on the next frame.
            // Dropping whatever has accumulated keeps that out of the yaw demand.
            lookX = 0f;
        }

        void Update()
        {
            // Cursor capture IS the driving mode: released, the panel takes the mouse and the hull does
            // not respond to it, which is what keeps the sliders usable.
            if (Input.GetKeyDown(KeyCode.Escape)) SetCapture(!mouseCaptured);

            if (mouseCaptured)
            {
                // Both axes come from the same per-rendered-frame delta source. Mouse X is accumulated
                // for the next fixed step; mouse Y moves the ride-height SETPOINT the hover LQR is
                // already regulating, so climbing costs no extra control channel. A locked cursor makes
                // those deltas unbounded, so the two sensitivities carry all of the feel.
                lookX += Input.GetAxis("Mouse X");
                targetRideHeight += Input.GetAxis("Mouse Y") * climbSensitivity;
            }

            targetRideHeight = Mathf.Clamp(targetRideHeight, RideHeightMin, RideHeightMax);
        }

        void FixedUpdate()
        {
            float dt = Time.fixedDeltaTime;
            rb.mass = hullMass;   // keep the real body in sync with the slider used by the linearization

            // ---- capture truth for the sensor simulation ----
            // World acceleration is differenced from the body's own velocity, so the specific force the
            // accelerometer is handed is the real one and not a restatement of the commanded thrust.
            Transform hull = hullGO.transform;
            Vector3 vel = rb.linearVelocity;
            Vector3 accelWorld = (vel - prevVelocity) / dt;
            prevVelocity = vel;

            truth = new TankTruth
            {
                Position = hull.position,
                Velocity = vel,
                Right = hull.right,
                Up = hull.up,
                Fwd = hull.forward,
            };
            truth.SpecificForce = truth.ToBody(accelWorld - Physics.gravity);
            truth.AngularRate = truth.ToBody(rb.angularVelocity);

            // ---- sense: the 5x5 lidar fan and the four proximity rangers ----
            TankSensorRig.Range(hull, lidarOrigin, lidarDirs, rayLength, lidarTrue);
            TankSensorRig.Range(hull, proxOrigins, proxDirs, rayLength, proxTrue);

            // ---- sense: 4 nozzle-down raycasts, one per thruster, for ground effect ----
            // Fired from the exhaust planes rather than from the lidar mount, because ground effect is a
            // property of where the DOWNWASH meets the ground: a tilted hull puts its four nozzles at
            // four different heights, and the asymmetric augmentation that follows is the whole point.
            //
            // A nozzle whose ray finds nothing reads rayLength rather than being dropped: no ground
            // within rayLength IS the physical answer here (out of ground effect), and nothing
            // differences these four, so a step in one cannot become a phantom tilt. These four also
            // stay NOISE-FREE, because the same numbers scale the force applied to the rigid body and
            // the allocation's Jacobian — they are a plant property, not an estimate.
            for (int i = 0; i < 4; i++)
            {
                Vector3 exit = NozzleExitWorld(i);
                nozzleReturn[i] = Physics.Raycast(exit, Vector3.down, out RaycastHit hit, rayLength);
                nozzleHeights[i] = nozzleReturn[i] ? hit.distance : rayLength;
            }

            // ---- estimate: corrupt the readings into what each sensor reports, then fuse ----
            BuildFilterNoise(dt);
            var estJob = new TankEstimatorJob
            {
                Truth = truth,
                TrueAccelBias = trueAccelBias, TrueGyroBias = trueGyroBias,
                LidarDirs = lidarDirs, LidarOrigin = lidarOrigin,
                LidarTrue = lidarTrue, LidarSensed = lidarSensed,
                ProxTrue = proxTrue, ProxSensed = proxSensed,
                Noise = noise, Spec = sensors,
                Kf = kf, Q = kfQ, RMag = kfRMag, RGps = kfRGps,
                KfOut = kfOut, Out = estimateOut, HoverState = hoverState, Ground = groundHold,
                GpsAge = gpsAge,
                Dt = dt, Gravity = -Physics.gravity.y, TargetRideHeight = targetRideHeight,
                Step = stepCount,
            };

            var sw = Stopwatch.StartNew();
            IJobExtensions.RunByRef(ref estJob);
            sw.Stop();
            estMs = (float)sw.Elapsed.TotalMilliseconds;

            noise = estJob.Noise;   // the random stream advanced
            kf = estJob.Kf;         // x and P advanced
            estimate = estimateOut[0];
            stepCount++;

            RecordError();

            // Mouse X and A/D are two INPUT DEVICES on one axis, so they sum and clamp.
            float mouseSteer = lookX * lookSensitivity;
            lookX = 0f;
            lastSteer = Mathf.Clamp(Input.GetAxis("Horizontal") + mouseSteer, -1f, 1f);
            lastStrafe = (Input.GetKey(KeyCode.E) ? 1f : 0f) - (Input.GetKey(KeyCode.Q) ? 1f : 0f);
            lastBrake = Input.GetKey(KeyCode.Space);

            // Everything the control law is handed below comes out of the estimator. Nothing here
            // reads the rigid body or the transform.
            var job = new HoverTankMPCStepJob
            {
                HoverState = hoverState,
                HoverK = hoverK, HoverLqrState = hoverLqr, HoverOut = hoverOut,
                Mass = hullMass, RollInertia = rollInertia, PitchInertia = pitchInertia,
                Gravity = -Physics.gravity.y,
                QHeight = qHeight, QHeightRate = qHeightRate, QTilt = qTilt, QTiltRate = qTiltRate,
                RThrust = rThrust, RTorque = rTorque,
                Dt = dt,

                Controls = controls, AllocOut = allocOut, WrenchOut = wrenchOut,
                Settings = thrusters, Health = thrusterHealth,
                MountX = mountX, MountY = mountY, MountZ = mountZ, MountArm = mountArm,
                DriveInput = Input.GetAxis("Vertical"), DriveForce = driveForce,
                StrafeInput = lastStrafe, StrafeForce = strafeForce,
                SteerInput = lastSteer, SteerTorque = steerTorque,
                BrakeInput = lastBrake,
                BrakeForce = brakeForce, BrakeGain = brakeGain, BrakeYawGain = brakeYawGain,
                IdleLinearGain = idleLinearGain, IdleAngularGain = idleAngularGain,
                ForwardSpeed = estimate.ForwardSpeed,
                LateralSpeed = estimate.LateralSpeed,
                YawRate = estimate.YawRate,
                TiltCos = estimate.TiltCos,

                NozzleHeights = nozzleHeights,
                NozzleRadius = groundEffect ? nozzleRadius : 0f,
                GroundOut = groundOut,
            };

            sw.Restart();
            IJobExtensions.RunByRef(ref job);
            sw.Stop();
            ctrlMs = (float)sw.Elapsed.TotalMilliseconds;

            hoverLqr = job.HoverLqrState;

            LogOnceIfDiverged(hoverOut[1] == 1f, ref hoverDivergedLogged, "hover LQR");
            LogOnceIfDiverged(allocOut[0].status == QPStatus.Optimal, ref allocFailedLogged, "allocation QP");
            LogOnceIfDiverged(kfOut[0].status == KFStatus.Ok && kfOut[1].status == KFStatus.Ok
                              && kfOut[2].status == KFStatus.Ok, ref estimatorFailedLogged, "state estimator");

            // ---- apply thrust: one force per thruster, at its mount, along its gimbal direction ----
            // AddForceAtPosition reproduces both the force and its moment about the center of mass,
            // which is the wrench the allocation solved for.
            //
            // groundOut is the step job's own ground-effect factor, not a second evaluation of the
            // model: the plant and the allocation's Jacobian must scale by the same number or the hover
            // loop chases an error the allocation cannot see.
            for (int i = 0; i < 4; i++)
            {
                float pitch = controls[i], yaw = controls[4 + i], throttle = controls[8 + i];
                float3 dir = GimbalAllocation.ForceDirection(pitch, yaw);
                float magnitude = throttle * thrusters.maxThrust * thrusterHealth[i] * groundOut[i];
                Vector3 worldDir = hullGO.transform.TransformDirection(new Vector3(dir.x, dir.y, dir.z));
                rb.AddForceAtPosition(worldDir * magnitude, MountWorld(i), ForceMode.Force);

                thrusterPivots[i].localRotation = GimbalRotation(pitch, yaw);
                UpdatePlume(i, throttle, thrusterHealth[i] > 0f);
            }
        }

        /// <summary>
        /// Measures the estimate against truth for the panel and pushes it onto the trace ring. This
        /// is the ONLY comparison of the two in the demo, and it drives a readout — nothing here feeds
        /// back into the estimator or the control law.
        /// </summary>
        void RecordError()
        {
            posError = estimate.Position - truth.Position;
            attError = Attitude.Difference(estimate.Rpy, Attitude.FromBasis(truth.Right, truth.Up, truth.Fwd));

            // True ride height is measured the same way the fit reports it: perpendicular to the local
            // ground plane, which is the vertical gap foreshortened by the terrain's own tilt.
            Vector3 origin = hullGO.transform.position
                           + hullGO.transform.rotation * new Vector3(lidarOrigin.x, lidarOrigin.y, lidarOrigin.z);
            float3 nTrue = TerrainNormal(origin.x, origin.z);
            trueSlopeDeg = Mathf.Acos(Mathf.Clamp(nTrue.y, -1f, 1f)) * Mathf.Rad2Deg;
            float vertical = origin.y - TerrainField.Height(origin.x, origin.z);
            clearanceError = estimate.Clearance - vertical * nTrue.y;

            traceHoriz[traceHead] = math.length(posError.xz);
            traceVert[traceHead] = math.abs(posError.y);
            traceFix[traceHead] = estimate.GpsFix;
            traceHead = (traceHead + 1) % TraceLength;
        }

        /// <summary>
        /// Terrain-truth surface normal at a world XZ point, by central differences on
        /// <see cref="TerrainField.Height"/>. UI only: this is the number the fitted slope is scored
        /// against, and no sensor may call it.
        /// </summary>
        static float3 TerrainNormal(float x, float z)
        {
            const float d = 0.25f;
            float gx = (TerrainField.Height(x + d, z) - TerrainField.Height(x - d, z)) / (2f * d);
            float gz = (TerrainField.Height(x, z + d) - TerrainField.Height(x, z - d)) / (2f * d);
            return math.normalize(new float3(-gx, 1f, -gz));
        }

        // Pitch servo first, yaw servo outboard of it, matching the pitch-then-yaw chain
        // GimbalAllocation.ForceDirection differentiates: the result maps local +y onto that
        // direction, so the nozzle and its plume point down the exhaust.
        static Quaternion GimbalRotation(float pitch, float yaw)
            => Quaternion.AngleAxis(-yaw * Mathf.Rad2Deg, Vector3.forward)
             * Quaternion.AngleAxis(pitch * Mathf.Rad2Deg, Vector3.right);

        void LateUpdate()
        {
            if (chaseCam == null || hullGO == null) return;

            // Frame-rate independent smoothing toward the wanted pose.
            float t = 1f - Mathf.Exp(-camLag * Time.deltaTime);
            chaseCam.transform.position = Vector3.Lerp(chaseCam.transform.position, WantedCameraPosition(), t);
            chaseCam.transform.rotation = Quaternion.LookRotation(
                hullGO.transform.position + Vector3.up * 1.5f - chaseCam.transform.position, Vector3.up);
        }

        // Follows the hull's HEADING only. Inheriting roll and pitch would swim the horizon on every
        // attitude correction, which is exactly what the hover loop spends its time doing.
        Vector3 WantedCameraPosition()
        {
            Vector3 flat = Vector3.ProjectOnPlane(hullGO.transform.forward, Vector3.up);
            if (flat.sqrMagnitude < 1e-6f) flat = Vector3.forward;
            return hullGO.transform.position - flat.normalized * camDistance + Vector3.up * camHeight;
        }

        void SnapCamera()
        {
            if (chaseCam == null) return;
            chaseCam.transform.position = WantedCameraPosition();
            chaseCam.transform.rotation = Quaternion.LookRotation(
                hullGO.transform.position + Vector3.up * 1.5f - chaseCam.transform.position, Vector3.up);
        }

        static void LogOnceIfDiverged(bool converged, ref bool alreadyLogged, string label)
        {
            if (converged) { alreadyLogged = false; return; }
            if (alreadyLogged) return;
            UnityEngine.Debug.LogWarning($"HoverTankMPCDemo: {label} did not converge, holding the last solution");
            alreadyLogged = true;
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying || hullGO == null) return;

            Transform hull = hullGO.transform;
            Vector3 lidarWorld = hull.position + hull.rotation * (Vector3)lidarOrigin;

            // The lidar fan: a beam that returned is drawn to its hit, a beam that did not is drawn
            // dim to the end of its range. Misses are what the plane fit has to drop.
            for (int k = 0; k < LidarGrid.Rays; k++)
            {
                Vector3 dir = hull.rotation * (Vector3)lidarDirs[k];
                float r = lidarSensed[k];
                bool hit = r > 0f;
                Gizmos.color = hit ? new Color(0.25f, 0.85f, 1f, 0.7f) : new Color(0.3f, 0.3f, 0.35f, 0.35f);
                Gizmos.DrawLine(lidarWorld, lidarWorld + dir * (hit ? r : rayLength));
                if (hit) Gizmos.DrawSphere(lidarWorld + dir * r, 0.05f);
            }

            // The fitted ground plane, as a cross lying in it at the sensed clearance below the mount,
            // plus its normal. Over a slope this tips with the ground while the hull stays level.
            GroundPlane held = groundHold[0];
            if (held.Valid)
            {
                Vector3 n = hull.rotation * (Vector3)held.Normal;
                Vector3 foot = lidarWorld - n * held.Clearance;
                Vector3 a = Vector3.Cross(n, hull.forward);
                if (a.sqrMagnitude < 1e-6f) a = Vector3.Cross(n, hull.right);
                a.Normalize();
                Vector3 b = Vector3.Cross(n, a);
                Gizmos.color = new Color(1f, 0.85f, 0.2f);
                Gizmos.DrawLine(foot - a * 3f, foot + a * 3f);
                Gizmos.DrawLine(foot - b * 3f, foot + b * 3f);
                Gizmos.DrawLine(foot, foot + n * 2f);
            }

            // The four proximity rangers, on the hull faces they look out of.
            for (int k = 0; k < ProximityRig.Rays; k++)
            {
                Vector3 o = hull.position + hull.rotation * (Vector3)proxOrigins[k];
                Vector3 dir = hull.rotation * (Vector3)proxDirs[k];
                float r = proxSensed[k];
                bool hit = r > 0f;
                Gizmos.color = hit ? new Color(1f, 0.35f, 0.35f) : new Color(0.3f, 0.25f, 0.25f, 0.35f);
                Gizmos.DrawLine(o, o + dir * (hit ? r : rayLength));
            }

            // The ESTIMATE, as a wire hull at the estimated pose. The gap between it and the real hull
            // is the position and attitude error the panel is plotting.
            Gizmos.color = new Color(1f, 0.4f, 0.9f, 0.9f);
            Gizmos.matrix = Matrix4x4.TRS(estimate.Position,
                Quaternion.Euler(estimate.Rpy.y * Mathf.Rad2Deg, estimate.Rpy.z * Mathf.Rad2Deg,
                                 estimate.Rpy.x * Mathf.Rad2Deg),
                Vector3.one);
            Gizmos.DrawWireCube(Vector3.zero, new Vector3(2f * hullHalfWidth, hullHeight, 2f * hullHalfLength));
            Gizmos.matrix = Matrix4x4.identity;

            // Ground-effect ranges, one per nozzle, warming from grey to orange with that nozzle's
            // augmentation: over sloping ground the four differ, which is what the allocation is
            // trading against.
            for (int i = 0; i < 4; i++)
            {
                Vector3 exit = NozzleExitWorld(i);
                float t = Mathf.Clamp01((groundOut[i] - 1f) / (GroundEffect.MaxFactor - 1f));
                Gizmos.color = nozzleReturn[i]
                    ? Color.Lerp(new Color(0.35f, 0.35f, 0.4f), new Color(1f, 0.55f, 0.1f), t)
                    : new Color(0.22f, 0.22f, 0.26f);
                Gizmos.DrawLine(exit, exit + Vector3.down * nozzleHeights[i]);
            }

            float lim = GimbalAllocation.MaxGimbalDeg;
            float angLo = Mathf.Clamp(Mathf.Min(thrusters.servoMinDeg, thrusters.servoMaxDeg), -lim, lim) * Mathf.Deg2Rad;
            float angHi = Mathf.Clamp(Mathf.Max(thrusters.servoMinDeg, thrusters.servoMaxDeg), -lim, lim) * Mathf.Deg2Rad;

            for (int i = 0; i < 4; i++)
            {
                Vector3 mount = MountWorld(i);
                float pitch = controls[i], yaw = controls[4 + i], throttle = controls[8 + i];

                // The two gimbal travel arcs, each swept on the exhaust side through the other axis'
                // current angle, so the pair shows the cap actually reachable from here.
                Gizmos.color = new Color(0.4f, 0.4f, 0.45f);
                Vector3 prevP = ExhaustPoint(mount, angLo, yaw, 0.9f);
                Vector3 prevY = ExhaustPoint(mount, pitch, angLo, 0.9f);
                for (int k = 1; k <= 10; k++)
                {
                    float a = Mathf.Lerp(angLo, angHi, k / 10f);
                    Vector3 nextP = ExhaustPoint(mount, a, yaw, 0.9f);
                    Vector3 nextY = ExhaustPoint(mount, pitch, a, 0.9f);
                    Gizmos.DrawLine(prevP, nextP);
                    Gizmos.DrawLine(prevY, nextY);
                    prevP = nextP; prevY = nextY;
                }

                // commanded exhaust, length proportional to throttle
                Gizmos.color = thrusterHealth[i] > 0f ? Color.Lerp(Color.green, Color.red, throttle) : Color.gray;
                Gizmos.DrawLine(mount, ExhaustPoint(mount, pitch, yaw, 0.7f + 3f * throttle));
                Gizmos.DrawSphere(mount, 0.09f);
            }
        }

        // The nozzle points opposite the force it produces: down at gimbal angles (0, 0).
        Vector3 ExhaustPoint(Vector3 mount, float pitch, float yaw, float length)
        {
            float3 dir = GimbalAllocation.ForceDirection(pitch, yaw);
            return mount + hullGO.transform.TransformDirection(new Vector3(-dir.x, -dir.y, -dir.z)) * length;
        }

        void OnGUI()
        {
            GUILayout.BeginArea(PanelRect, GUI.skin.box);
            GUILayout.Label($"Hover tank over terrain — {estMs:F3} ms sense+EKF, {ctrlMs:F3} ms control (15-state EKF + 3x6 hover LQR + 12-control allocation QP)");
            GUILayout.Label($"Mouse X turn   Mouse Y climb   W/S drive   Q/E strafe   A/D yaw   SPACE brake   ESC {(mouseCaptured ? "release cursor" : "RESUME DRIVING")}");

            DrawEstimatorPanel();

            GUILayout.Label($"hover: converged={hoverOut[1] == 1f}  iters={hoverOut[0]:F0}  residual={hoverOut[2]:E1}   state: h={hoverState[0]:F2} roll={hoverState[2] * Mathf.Rad2Deg:F1} pitch={hoverState[4] * Mathf.Rad2Deg:F1}");

            QPInfo alloc = allocOut[0];
            GUILayout.Label($"alloc QP: {alloc.status}  pivots={alloc.iterations}  obj={alloc.objective:E2}");
            GUILayout.Label($"force  N   lateral {wrenchOut[6]:F0}/{wrenchOut[0]:F0}   lift {wrenchOut[7]:F0}/{wrenchOut[1]:F0}   drive {wrenchOut[8]:F0}/{wrenchOut[2]:F0}   (achieved/demanded)");
            GUILayout.Label($"torque Nm  pitch {wrenchOut[9]:F0}/{wrenchOut[3]:F0}   yaw {wrenchOut[10]:F0}/{wrenchOut[4]:F0}   roll {wrenchOut[11]:F0}/{wrenchOut[5]:F0}");
            GUILayout.Label($"yaw axis: {YawOwner()}   speed {estimate.ForwardSpeed,5:F1} m/s   strafe {estimate.LateralSpeed,5:F1} m/s   yaw rate {estimate.YawRate * Mathf.Rad2Deg,5:F0} deg/s   (all estimated)");
            GUILayout.Label($"ride height cmd {targetRideHeight:F2} m   sensed {estimate.Clearance:F2} m   lidar {GroundLabel()}   mouse {(mouseCaptured ? "CAPTURED" : "released")}");

            GUILayout.BeginHorizontal();
            groundEffect = GUILayout.Toggle(groundEffect, "ground effect", GUILayout.Width(110));
            GUILayout.Label(groundEffect
                ? $"x{MeanGroundGain():F3} into B   FL {groundOut[0]:F2} FR {groundOut[1]:F2} BL {groundOut[2]:F2} BR {groundOut[3]:F2}   clamp x{GroundEffect.MaxFactor:F2} under {GroundEffect.ClampHeight(nozzleRadius):F2} m"
                : "off — thrusters deliver exactly what they are commanded at any height");
            GUILayout.EndHorizontal();

            for (int i = 0; i < 4; i++)
            {
                float throttle = controls[8 + i];
                GUILayout.Label($"{MountNames[i]}  gimbal pitch {controls[i] * Mathf.Rad2Deg,6:F1} yaw {controls[4 + i] * Mathf.Rad2Deg,6:F1} deg   thrust {throttle * thrusters.maxThrust,7:F0} N ({throttle * 100f:F0}%)"
                    + (thrusterHealth[i] > 0f ? "" : "   DEAD"));
            }

            GUILayout.BeginHorizontal();
            GUILayout.Label("thrusters", GUILayout.Width(70));
            for (int i = 0; i < 4; i++)
            {
                bool live = thrusterHealth[i] > 0f;
                bool now = GUILayout.Toggle(live, MountNames[i], GUILayout.Width(50));
                if (now != live) thrusterHealth[i] = now ? 1f : 0f;
            }
            GUILayout.EndHorizontal();

            targetRideHeight = LabeledSlider($"ride height {targetRideHeight:F2}", targetRideHeight, RideHeightMin, RideHeightMax);
            nozzleRadius = LabeledSlider($"nozzle radius {nozzleRadius:F2}m", nozzleRadius, 0.5f, 4f);
            qTilt = LabeledSlider($"Q tilt {qTilt:F0}", qTilt, 1f, 300f);
            lookSensitivity = LabeledSlider($"mouse sens {lookSensitivity:F2}", lookSensitivity, 0.01f, 1f);
            climbSensitivity = LabeledSlider($"climb sens {climbSensitivity:F2}m", climbSensitivity, 0.01f, 1f);
            strafeForce = LabeledSlider($"strafe force {strafeForce:F0}N", strafeForce, 1000f, 40000f);
            brakeForce = LabeledSlider($"brake force {brakeForce:F0}N", brakeForce, 1000f, 30000f);
            brakeYawGain = LabeledSlider($"brake yaw {brakeYawGain:F0}Nm/(rad/s)", brakeYawGain, 500f, 30000f);
            idleLinearGain = LabeledSlider($"idle linear {idleLinearGain:F0}N/(m/s)", idleLinearGain, 0f, 5000f);
            idleAngularGain = LabeledSlider($"idle yaw {idleAngularGain:F0}Nm/(rad/s)", idleAngularGain, 0f, 15000f);
            thrusters.servoMaxDeg = LabeledSlider($"gimbal range +{thrusters.servoMaxDeg:F0}deg", thrusters.servoMaxDeg, 0f, GimbalAllocation.MaxGimbalDeg);
            thrusters.servoRateDeg = LabeledSlider($"servo rate {thrusters.servoRateDeg:F0}deg/s", thrusters.servoRateDeg, 15f, 720f);

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Kick"))
            {
                rb.AddForce(Vector3.up * 6000f, ForceMode.Impulse);
                rb.AddTorque(Vector3.right * 4000f, ForceMode.Impulse);
            }
            if (GUILayout.Button("Reset"))
            {
                rb.position = spawnPosition;
                rb.rotation = Quaternion.identity;
                rb.linearVelocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
                thrusterHealth = new float4(1f);
                prevVelocity = Vector3.zero;
                SeedEstimator();
                ResetControls();
                SnapCamera();
            }
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        // What is driving the yaw demand this step (see HoverTankMPCStepJob.Execute), and whether the
        // allocation actually delivered it. Yaw authority is bought with forward thrust the hover loop
        // also wants, so a hard mouse flick can exceed what the rig can produce — say so rather than
        // letting the hull drift off heading unexplained.
        string YawOwner()
        {
            string owner;
            if (lastBrake) owner = "BRAKE (space)";
            else if (Mathf.Abs(lastSteer) > HoverTankMPCStepJob.StickDeadzone) owner = "DRIVER (mouse + A/D)";
            else if (!mouseCaptured) owner = "cursor released";
            else owner = idleAngularGain > 0f ? "idle damping" : "free";

            float torqueScale = hullMass * -Physics.gravity.y * mountArm;
            bool shortfall = Mathf.Abs(wrenchOut[10] - wrenchOut[4]) > 0.05f * torqueScale;
            return shortfall ? owner + "  [YAW SATURATED]" : owner;
        }

        /// <summary>Mean of the four nozzle augmentations — the factor the hover model's B carries.</summary>
        float MeanGroundGain() => 0.25f * (groundOut[0] + groundOut[1] + groundOut[2] + groundOut[3]);

        // Beams that came back this step and how many of them the fitted plane kept. Too few returns
        // and the fit is refused, so the hover loop is flying against the last plane the lidar could
        // see; a large gap between the two counts means the fan is straddling a terrain feature.
        string GroundLabel()
            => estimate.GroundValid
                ? $"{estimate.LidarInliers}/{estimate.LidarReturns} of {LidarGrid.Rays} beams"
                : $"{estimate.LidarReturns}/{LidarGrid.Rays} beams  [HOLDING]";

        /// <summary>
        /// The estimator readout: how far the estimate has drifted from truth, what the sensors are
        /// doing about it, and the hull-tilt-versus-terrain-slope split the fitted plane buys.
        ///
        /// The trace is where the multi-rate structure shows: horizontal position walks away under
        /// inertial dead reckoning and is pulled back at every beacon fix, while the attitude row
        /// barely moves because the gravity reference and the magnetometer run two orders of magnitude
        /// more often.
        /// </summary>
        void DrawEstimatorPanel()
        {
            float3 attDeg = attError * Mathf.Rad2Deg;
            GUILayout.Label($"EST vs TRUTH   pos  x {posError.x,6:F2}  y {posError.y,6:F2}  z {posError.z,6:F2} m   |horiz| {math.length(posError.xz),5:F2} m"
                          + $"   ride height {clearanceError,6:F2} m");
            GUILayout.Label($"               att  roll {attDeg.x,6:F2}  pitch {attDeg.y,6:F2}  yaw {attDeg.z,6:F2} deg"
                          + $"   bias a {math.length(estimate.AccelBias):F3}/{math.length(trueAccelBias):F3}  g {math.length(estimate.GyroBias):F4}/{math.length(trueGyroBias):F4}");
            GUILayout.Label($"sensors: beacon fix {estimate.StepsSinceGps * Time.fixedDeltaTime:F2} s ago"
                          + $"   gravity ref sigma {estimate.TiltSigma:F3} ({(estimate.TiltSigma > 3f * sensors.tiltSigma ? "MANOEUVRING" : "coasting")})"
                          + $"   mag {(estimate.MagFix ? "on" : "-")}   {GroundLabel()}");

            float hullTilt = Mathf.Acos(Mathf.Clamp(estimate.TiltCos, -1f, 1f)) * Mathf.Rad2Deg;
            float hullTiltTrue = Mathf.Acos(Mathf.Clamp(truth.Up.y, -1f, 1f)) * Mathf.Rad2Deg;
            float slope = Mathf.Acos(Mathf.Clamp(estimate.GroundNormal.y, -1f, 1f)) * Mathf.Rad2Deg;
            GUILayout.Label($"SEPARATED  hull tilt {hullTilt:F1} deg (truth {hullTiltTrue:F1})   terrain slope {slope:F1} deg (truth {trueSlopeDeg:F1})"
                          + "   — one estimate cannot say both without the fitted plane");

            DrawErrorTrace(GUILayoutUtility.GetRect(TraceLength, 52f));
        }

        /// <summary>
        /// Draws the error ring buffer: horizontal position error in cyan, vertical in amber, one
        /// pixel column per fixed step, with a tick at each beacon fix. Full scale is
        /// <see cref="TraceScale"/> metres.
        /// </summary>
        void DrawErrorTrace(Rect r)
        {
            if (Event.current.type != EventType.Repaint) return;

            Color prev = GUI.color;
            GUI.color = new Color(0.1f, 0.1f, 0.12f, 0.85f);
            GUI.DrawTexture(r, Texture2D.whiteTexture);

            float w = Mathf.Min(r.width, TraceLength);
            for (int k = 0; k < TraceLength; k++)
            {
                int i = (traceHead + k) % TraceLength;
                float x = r.x + k * (w / TraceLength);

                if (traceFix[i])
                {
                    GUI.color = new Color(0.35f, 0.35f, 0.4f);
                    GUI.DrawTexture(new Rect(x, r.y, 1f, r.height), Texture2D.whiteTexture);
                }

                DrawTraceBar(r, x, traceVert[i], new Color(1f, 0.7f, 0.2f, 0.8f));
                DrawTraceBar(r, x, traceHoriz[i], new Color(0.3f, 0.85f, 1f));
            }

            GUI.color = prev;
        }

        /// <summary>Full-scale of the error trace, metres.</summary>
        const float TraceScale = 4f;

        static void DrawTraceBar(Rect r, float x, float value, Color color)
        {
            float h = Mathf.Clamp01(value / TraceScale) * r.height;
            if (h < 1f) h = 1f;
            GUI.color = color;
            GUI.DrawTexture(new Rect(x, r.yMax - h, 1f, h), Texture2D.whiteTexture);
        }

        static float LabeledSlider(string label, float v, float lo, float hi)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, GUILayout.Width(170));
            v = GUILayout.HorizontalSlider(v, lo, hi, GUILayout.Width(220));
            GUILayout.EndHorizontal();
            return v;
        }
    }

    /// <summary>
    /// Per-fixed-step control law, downstream of <see cref="TankEstimatorJob"/> and blind to anything
    /// it did not produce. Warm-solves the 6-state hover LQR over the estimated hover state (3
    /// acceleration commands: vertical, roll, pitch); resolves the driver's forward/strafe/yaw/brake
    /// inputs into the rest of the demanded hull-frame <see cref="GimbalWrench"/>; then allocates that
    /// onto 4 pitch angles, 4 yaw angles and 4 throttles with <see cref="GimbalAllocation.Solve"/>.
    /// The LQR re-runs every step (warm <see cref="floatLQRState"/>, cheap once converged) to showcase
    /// the warm-start path.
    ///
    /// This step's <see cref="GroundEffect"/> augmentation enters the allocation's Jacobian, where it
    /// is exact, and the hover model's vertical input column, where it is a deliberate detune — see
    /// <see cref="Execute"/>.
    ///
    /// Caller must RunByRef and copy HoverLqrState back.
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct HoverTankMPCStepJob : IJob
    {
        /// <summary>Stick deflection below which the driver is considered hands-off the yaw axis.</summary>
        public const float StickDeadzone = 0.02f;

        // hover / attitude
        /// <summary>
        /// [ride-height error, closing rate, roll, roll rate, pitch, pitch rate] — the estimator's
        /// output, read here and never written. Angles are radians and follow
        /// <see cref="Attitude"/>: positive pitch is nose-down, positive roll lifts the right side.
        /// </summary>
        [ReadOnly] public NativeArray<float> HoverState;
        public floatMxN HoverK;
        public floatLQRState HoverLqrState;
        public NativeArray<float> HoverOut;
        public float Mass, RollInertia, PitchInertia, Gravity;
        public float QHeight, QHeightRate, QTilt, QTiltRate, RThrust, RTorque;
        public float Dt;

        // driver
        public float DriveInput, DriveForce;
        /// <summary>Q/E strafe demand in [-1, 1]; positive is to the hull's right.</summary>
        public float StrafeInput;
        public float StrafeForce;
        /// <summary>Mouse X and A/D already summed and clamped to [-1, 1].</summary>
        public float SteerInput;
        public float SteerTorque;
        public bool BrakeInput;
        public float BrakeForce, BrakeGain, BrakeYawGain;
        public float IdleLinearGain, IdleAngularGain;
        /// <summary>Hull velocity along its own forward axis, m/s.</summary>
        public float ForwardSpeed;
        /// <summary>Hull velocity along its own right axis, m/s.</summary>
        public float LateralSpeed;
        /// <summary>Hull yaw rate about its own up axis, rad/s.</summary>
        public float YawRate;

        // thruster allocation
        public NativeArray<float> Controls;
        public NativeArray<QPInfo> AllocOut;
        public NativeArray<float> WrenchOut;
        public GimbalSettings Settings;
        public float4 Health, MountX, MountY, MountZ;
        public float MountArm;
        /// <summary>cos of the hull's tilt from world up, for the gravity feedforward.</summary>
        public float TiltCos;

        // ground effect
        /// <summary>Each nozzle's exit plane above the ground, metres, in mount order.</summary>
        public float4 NozzleHeights;

        /// <summary>Effective nozzle radius, metres. 0 or less turns ground effect off.</summary>
        public float NozzleRadius;

        /// <summary>
        /// The four <see cref="GroundEffect.Factor(float, float)"/> values this step settled on. The
        /// caller must scale the force it applies to the rigid body by these, since they are also what
        /// the allocation sized its throttles against.
        /// </summary>
        public NativeArray<float> GroundOut;

        public void Execute()
        {
            // ---- ground effect: one augmentation per nozzle ----
            // Evaluated once, here, and read back out: the applied force and the allocation's view of
            // what a throttle buys are then the same numbers by construction, not by agreement.
            float4 groundGain = GroundEffect.Factor(NozzleHeights, NozzleRadius);
            for (int i = 0; i < GimbalAllocation.Thrusters; i++) GroundOut[i] = groundGain[i];

            // ---- hover LQR: warm re-solve every step, with the mean augmentation in B ----
            // WHAT THIS IS, PRECISELY: a deliberate mild detune, NOT identification of a varying plant.
            // The allocation is handed the true per-nozzle gain and inverts it exactly, so a unit of
            // commanded vertical acceleration still buys exactly one unit at every height and ground
            // effect is invisible to the controller. Telling the model otherwise only lowers K, so the
            // real closed loop -- which runs against the true unity map, not against this B -- gets
            // slightly SOFTER near the ground rather than tighter. That is the intent: hold station
            // less aggressively where the cushion is already helping.
            //
            // Scheduling would be real identification if the allocation were given only an ESTIMATE of
            // the gain; the plant would then genuinely vary with height and the estimate's error would
            // be a disturbance worth rejecting.
            //
            // Free either way: the model is rebuilt and warm re-solved every step regardless, and the
            // warm floatLQRState re-converges from the previous step's S rather than solving cold.
            BuildHoverModel(Dt, 0.25f * math.csum(groundGain),
                QHeight, QHeightRate, QTilt, QTiltRate, RThrust, RTorque,
                Allocator.Temp, out var A, out var B, out var Q, out var R);

            RiccatiInfo info = LQR.lqr(in A, in B, in Q, in R, ref HoverK, ref HoverLqrState);
            HoverOut[0] = info.iterations;
            HoverOut[1] = info ? 1f : 0f;
            HoverOut[2] = (float)info.residual;
            HoverOut[3] = info.rankDeficient ? 1f : 0f;
            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose();

            // u = -K x  ->  [vertical accel command, roll angular accel command, pitch angular accel command]
            float uVertAccel = 0f, uRollAccel = 0f, uPitchAccel = 0f;
            for (int j = 0; j < 6; j++)
            {
                uVertAccel -= HoverK[0, j] * HoverState[j];
                uRollAccel -= HoverK[1, j] * HoverState[j];
                uPitchAccel -= HoverK[2, j] * HoverState[j];
            }

            // ---- driver demand ----
            // Everything here is a WRENCH DEMAND handed to the allocation, never a force written onto
            // the rigid body: braking and damping are solved for like every other axis, so the servos
            // visibly swing to produce them.
            //
            // The brake and the idle damping COMPOSE AS A LADDER rather than summing. SPACE is the
            // strong explicit brake and subsumes idle damping entirely; otherwise idle damping fades
            // out in proportion to how hard the axis is being commanded. Two damping terms live at
            // once would feel mushy and would make the brake read weaker than it is.
            //
            // The three BODY-RATE axes the driver owns — along-track, lateral and yaw — are damped.
            // Pitch and roll are already regulated by the hover loop, and a second controller on those
            // axes would fight it.
            float driveIn = math.clamp(DriveInput, -1f, 1f);
            float strafeIn = math.clamp(StrafeInput, -1f, 1f);
            float steer = math.clamp(SteerInput, -1f, 1f);

            float drive, strafeDemand, yawDemand;
            if (BrakeInput)
            {
                // Both gains ease off as the rate drops, so the brake settles instead of chattering at
                // rest. BrakeForce stays under the drive authority and the yaw brake is capped at the
                // stick's own authority, so braking can never out-demand ordinary driving and starve
                // attitude control.
                drive = -math.clamp(BrakeGain * ForwardSpeed, -BrakeForce, BrakeForce);
                strafeDemand = -math.clamp(BrakeGain * LateralSpeed, -BrakeForce, BrakeForce);
                yawDemand = -math.clamp(BrakeYawGain * YawRate, -SteerTorque, SteerTorque);
            }
            else
            {
                // Idle damping is a D term on the demand, deliberately gentle: this is a hover vehicle
                // and the glide is the point, so it settles over a second or two rather than stopping dead.
                float linearDamp = -math.clamp(IdleLinearGain * ForwardSpeed, -DriveForce, DriveForce);
                float lateralDamp = -math.clamp(IdleLinearGain * LateralSpeed, -StrafeForce, StrafeForce);
                float yawDamp = -math.clamp(IdleAngularGain * YawRate, -SteerTorque, SteerTorque);

                drive = driveIn * DriveForce + (1f - math.abs(driveIn)) * linearDamp;
                strafeDemand = strafeIn * StrafeForce + (1f - math.abs(strafeIn)) * lateralDamp;

                yawDemand = math.abs(steer) > StickDeadzone
                    ? steer * SteerTorque + (1f - math.abs(steer)) * yawDamp
                    : yawDamp;
            }

            // ---- demanded hull-frame wrench ----
            // The gravity feedforward is divided by the hull's tilt cosine because thrust is bolted to
            // the hull and gravity is not; floored so a near-vertical hull cannot demand unbounded lift.
            var desired = new GimbalWrench
            {
                Lateral = strafeDemand,
                Lift = Mass * (Gravity / math.max(TiltCos, 0.35f) + uVertAccel),
                Drive = drive,
                Pitch = PitchInertia * uPitchAccel,
                Yaw = yawDemand,
                Roll = RollInertia * uRollAccel,
            };

            // ---- allocation: 12 controls onto the 6 wrench components ----
            var z = new floatN(Controls);   // view, no copy
            GimbalRig rig = GimbalAllocation.BuildRig(in Settings, MountX, MountY, MountZ, Health, groundGain,
                in z, Dt, Mass * Gravity, MountArm);
            AllocOut[0] = GimbalAllocation.Solve(in rig, in desired, ref z, 0);

            GimbalWrench got = GimbalAllocation.Wrench(in rig, in z);
            WrenchOut[0] = desired.Lateral; WrenchOut[1] = desired.Lift; WrenchOut[2] = desired.Drive;
            WrenchOut[3] = desired.Pitch; WrenchOut[4] = desired.Yaw; WrenchOut[5] = desired.Roll;
            WrenchOut[6] = got.Lateral; WrenchOut[7] = got.Lift; WrenchOut[8] = got.Drive;
            WrenchOut[9] = got.Pitch; WrenchOut[10] = got.Yaw; WrenchOut[11] = got.Roll;
        }

        /// <summary>
        /// Discrete (Euler, zero-order-hold over <paramref name="dt"/>) hover/attitude model: three
        /// decoupled double integrators — height/vertical-velocity, roll/roll-rate, pitch/pitch-rate —
        /// driven directly by ACCELERATION inputs [vertical accel, roll angular accel, pitch angular
        /// accel]: B = dt on each rate row, no mass/inertia scaling (chosen so B's entries stay near
        /// O(dt) instead of O(dt/mass) — the latter leaves the DARE badly scaled for a heavy hull).
        /// Gravity feedforward and the accel -> force/torque conversion (via mass/rollInertia/
        /// pitchInertia) are the caller's job, not this model's. Allocates A/B/Q/R fresh with
        /// <paramref name="allocator"/> (caller disposes).
        ///
        /// <paramref name="liftGain"/> scales the VERTICAL input column alone; 1 is the nominal model.
        /// The two angular columns are untouched — an augmentation asymmetry across the four nozzles is
        /// a torque the allocation resolves, not a change in the hull's angular control effectiveness.
        /// Whether a value other than 1 identifies a real change in control effectiveness, or only
        /// detunes the gain, depends on whether the caller's actuator path already compensates for it.
        /// </summary>
        public static void BuildHoverModel(float dt, float liftGain,
            float qHeight, float qHeightRate, float qTilt, float qTiltRate, float rThrust, float rTorque,
            Allocator allocator, out floatMxN A, out floatMxN B, out floatMxN Q, out floatMxN R)
        {
            A = new floatMxN(6, 6, allocator);
            B = new floatMxN(6, 3, allocator);
            Q = new floatMxN(6, 6, allocator);
            R = new floatMxN(3, 3, allocator);

            for (int i = 0; i < 6; i++) A[i, i] = 1f;
            A[0, 1] = dt; A[2, 3] = dt; A[4, 5] = dt;

            B[1, 0] = dt * liftGain; B[3, 1] = dt; B[5, 2] = dt;

            Q[0, 0] = qHeight; Q[1, 1] = qHeightRate;
            Q[2, 2] = qTilt; Q[3, 3] = qTiltRate;
            Q[4, 4] = qTilt; Q[5, 5] = qTiltRate;

            R[0, 0] = rThrust; R[1, 1] = rTorque; R[2, 2] = rTorque;
        }
    }
}
