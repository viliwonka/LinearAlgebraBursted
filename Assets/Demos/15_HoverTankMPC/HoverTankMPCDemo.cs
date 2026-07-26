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
    /// Every axis reaches the thrusters through two solves:
    ///
    /// 1. A receding-horizon MPC over 12 body states, re-solved every fixed step, producing all six
    ///    acceleration commands at once. Three things the LQR cascade it replaced could not express:
    ///    the terrain ahead is a PER-STAGE reference, so the tank climbs before the rise reaches it;
    ///    the proximity rangers are SOFT STATE ROWS on predicted displacement, so it stops short of a
    ///    wall; and the rig's authority is a HARD INPUT BOUND, so it plans against what the thrusters
    ///    can deliver instead of demanding a wrench and letting the allocation clip it.
    /// 2. A control-allocation QP that turns the resulting wrench into the 12 thruster controls, under
    ///    servo range/rate and thrust range/rate limits. See <see cref="GimbalAllocation"/>: 12
    ///    controls against 6 wrench components, so the rig is over-actuated and the solve is what
    ///    decides how the work is shared.
    ///
    /// The driver never writes a force. W/S, Q/E and A/D set a VELOCITY REFERENCE the MPC tracks, and
    /// hands-off is a zero reference — which is what settles the tank, with no separate damping term to
    /// fight the controller for those axes.
    ///
    /// Thrust is augmented near the ground by <see cref="GroundEffect"/>, per nozzle, from a downward
    /// ray at each exhaust plane — so a tilted hull is pushed harder on its low side. The same four
    /// factors scale the force applied to the rigid body AND the allocation's Jacobian, so the two
    /// agree exactly and the demanded wrench is delivered at any height. The MPC's own model does not
    /// carry them: it is LTI and built once, and the allocation already inverts the gain exactly.
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

        /// <summary>
        /// The MPC's state layout. sFwd/sLat/sVert/yaw are DISPLACEMENTS FROM NOW — zero in every x0,
        /// because the frame is re-anchored each solve — while roll and pitch are absolute attitudes
        /// the controller regulates to zero. sVert is WORLD vertical, deliberately not ride-height
        /// error: rising terrain is a disturbance in clearance coordinates, which
        /// <see cref="MPC.solve"/> has no term for, and a moving reference in world coordinates, which
        /// it does.
        ///
        /// Laid out as (displacement, rate) PAIRS in the same order as the inputs below:
        /// <see cref="BuildMpcModel"/> writes A and B from that pairing alone, so reordering these
        /// breaks the model silently.
        /// </summary>
        public const int SFwd = 0, VFwd = 1, SLat = 2, VLat = 3, SVert = 4, VVert = 5,
                         Roll = 6, RollRate = 7, Pitch = 8, PitchRate = 9, Yaw = 10, YawRate = 11;

        /// <summary>State dimension of the MPC model.</summary>
        public const int StateCount = 12;

        /// <summary>The MPC's inputs: accelerations, hull-frame for the linear three.</summary>
        public const int AFwd = 0, ALat = 1, AVert = 2, AlphaRoll = 3, AlphaPitch = 4, AlphaYaw = 5;

        /// <summary>Input dimension of the MPC model.</summary>
        public const int InputCount = 6;

        /// <summary>Number of soft state rows: +/- forward and +/- lateral displacement.</summary>
        public const int SoftRows = 4;

        [Header("Hull")]
        [Range(300f, 5000f)] public float hullMass = 1500f;
        [Range(1f, 4f)] public float hullHalfWidth = 2f;
        [Range(1f, 6f)] public float hullHalfLength = 3f;
        [Range(0.5f, 2f)] public float hullHeight = 1f;
        [Range(RideHeightMin, RideHeightMax)] public float targetRideHeight = 2f;
        [Range(2f, 20f)] public float rayLength = 8f;

        [Header("Hover MPC")]
        [Range(500f, 6000f)] public float rollInertia = 2100f;
        [Range(500f, 8000f)] public float pitchInertia = 4600f;
        [Range(500f, 12000f)] public float yawInertia = 6500f;
        [Tooltip("Prediction stages. The horizon is this times the fixed timestep, and it is how far ahead the terrain preview and the anti-collision rows can see. CONSTRUCTION-TIME: changing it during play does nothing, since the condensed horizon is built once.")]
        [Range(5, 40)] public int horizon = 25;
        [Tooltip("Weight on forward/lateral displacement and heading. Small on purpose: these are pure integrator modes, and a weight of exactly zero would leave them undetectable and the terminal Riccati solve ill-posed. Reads physically as gentle station-keeping, not as tracking.")]
        [Range(0.001f, 5f)] public float qPos = 0.05f;
        [Range(0.1f, 100f)] public float qVel = 12f;
        [Range(1f, 400f)] public float qVert = 120f;
        [Range(0.1f, 100f)] public float qVertRate = 14f;
        [Range(1f, 400f)] public float qTilt = 90f;
        [Range(0.1f, 50f)] public float qTiltRate = 8f;
        [Range(0.1f, 100f)] public float qYawRate = 10f;
        [Range(0.001f, 5f)] public float rLinear = 0.02f;   // cost on commanded linear accel (m/s^2), not force
        [Range(0.01f, 20f)] public float rAngular = 0.4f;   // cost on commanded angular accel (rad/s^2), not torque

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

        [Header("Driver demand (a reference the MPC tracks)")]
        [Tooltip("Top forward speed W/S asks for, m/s. Hands off, the reference is zero, which is what settles the tank -- there is no separate damping term.")]
        [Range(2f, 40f)] public float maxFwdSpeed = 14f;
        [Tooltip("Top sideways speed Q/E asks for, m/s. Lateral authority is bought out of the same thrust that carries the hull, so this stays under maxFwdSpeed.")]
        [Range(1f, 30f)] public float maxLatSpeed = 9f;
        [Range(0.2f, 4f)] public float maxYawRate = 1.4f;
        [Tooltip("Peak forward force, newtons. This is the MPC's own input bound, so the controller plans against it instead of demanding a wrench the rig cannot deliver.")]
        [Range(1000f, 40000f)] public float driveForce = 9000f;
        [Range(1000f, 40000f)] public float strafeForce = 7000f;
        [Range(1000f, 40000f)] public float steerTorque = 9000f;
        [Tooltip("Share of total thrust the roll/pitch input bounds assume is available for torque. The rest is carrying the hull.")]
        [Range(0.05f, 1f)] public float torqueAuthority = 0.35f;
        [Tooltip("Metres held back from whatever a proximity ranger reports, as a soft state row on predicted displacement.")]
        [Range(0.2f, 6f)] public float collisionMargin = 1.5f;
        [Tooltip("Exact-penalty price per metre of predicted intrusion past a ranger's bound. Must out-price the velocity tracking that is driving the tank at the obstacle, which is why it is far above the library default: that default assumes cost matrices at O(1), and these run to O(100).")]
        [Range(1e3f, 1e7f)] public float collisionPenalty = 1e5f;
        [Tooltip("Steer command per unit of accumulated mouse X. A locked cursor gives unbounded deltas, so this wants to stay small.")]
        [Range(0.01f, 1f)] public float lookSensitivity = 0.12f;
        [Tooltip("Metres of commanded ride height per unit of mouse Y. Push forward to climb.")]
        [Range(0.01f, 1f)] public float climbSensitivity = 0.15f;

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
        /// <summary>Built ONCE in <see cref="Start"/>: the model is LTI, so the condensed horizon and
        /// its terminal Riccati solve never need rebuilding. Only the reference, x0 and the soft-row
        /// bounds move per step.</summary>
        floatMPCState mpc;
        NativeArray<float> mpcX0, mpcRef, mpcU0, mpcSoft, previewOut;
        NativeArray<MPCInfo> mpcOut;
        NativeArray<float> hoverState;    // [height err, closing rate, roll, roll rate, pitch, pitch rate] — panel only

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

            hoverState = new NativeArray<float>(6, Allocator.Persistent);
            BuildMpc();

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

        /// <summary>
        /// Builds the receding-horizon state ONCE. Everything here is fixed for the life of the demo:
        /// the model is LTI, the soft rows select fixed states, and the input bounds are the rig's own
        /// authority. Only the RHS of those rows moves at runtime, via <see cref="MPC.setSoftBound"/>.
        ///
        /// Rebuilding instead would cost a fresh terminal Riccati solve plus the whole condensing every
        /// step — measured at 7.08 ms for a horizon of this shape against a 0.45 ms warm solve, which
        /// is what settled the architecture (see DEVLOG.md).
        /// </summary>
        void BuildMpc()
        {
            float dt = Time.fixedDeltaTime;

            BuildMpcModel(dt, qPos, qVel, qVert, qVertRate, qTilt, qTiltRate, qYawRate, rLinear, rAngular,
                          Allocator.Temp, out var A, out var B, out var Q, out var R);

            var uLo = new floatN(InputCount, Allocator.Temp, true);
            var uHi = new floatN(InputCount, Allocator.Temp, true);

            // The rig's real authority, in acceleration units. Handing these to the MPC as hard input
            // bounds is a capability the cascade did not have: the controller now plans against what
            // the thrusters can actually deliver rather than demanding a wrench and letting the
            // allocation clip it.
            float totalThrust = GimbalAllocation.Thrusters * thrusters.maxThrust;
            float aFwd = driveForce / hullMass;
            float aLat = strafeForce / hullMass;
            float aUp = math.max(0.5f, totalThrust / hullMass - (-Physics.gravity.y));
            float torque = torqueAuthority * totalThrust;

            uLo[AFwd] = -aFwd; uHi[AFwd] = aFwd;
            uLo[ALat] = -aLat; uHi[ALat] = aLat;
            // Thrust points broadly up, so the tank cannot pull itself down harder than free fall.
            uLo[AVert] = -(-Physics.gravity.y); uHi[AVert] = aUp;
            uLo[AlphaRoll] = -torque * hullHalfWidth / rollInertia;
            uHi[AlphaRoll] = torque * hullHalfWidth / rollInertia;
            uLo[AlphaPitch] = -torque * hullHalfLength / pitchInertia;
            uHi[AlphaPitch] = torque * hullHalfLength / pitchInertia;
            uLo[AlphaYaw] = -steerTorque / yawInertia; uHi[AlphaYaw] = steerTorque / yawInertia;

            // Anti-collision. C selects predicted displacement; only its bound moves at runtime.
            var C = new floatMxN(SoftRows, StateCount, Allocator.Temp);
            C[0, SFwd] = 1f; C[1, SFwd] = -1f; C[2, SLat] = 1f; C[3, SLat] = -1f;
            var d = new floatN(SoftRows, Allocator.Temp, true);
            for (int i = 0; i < SoftRows; i++) d[i] = rayLength;   // nothing seen yet

            // The penalty is passed EXPLICITLY. A metre of intrusion has to out-price the velocity
            // tracking pushing the tank at the obstacle, which is worth roughly
            // qVel * horizon / (horizon*dt)^2 per metre here — thousands, not the O(1)-scaled cost
            // matrices the library's own default assumes. Left at the default the tank accelerates
            // into a wall at full throttle and merely reports the violation.
            mpc = new floatMPCState(StateCount, InputCount, horizon, Allocator.Persistent,
                                    in A, in B, in Q, in R, in uLo, in uHi,
                                    in C, in d, collisionPenalty, 1f);

            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose();
            uLo.Dispose(); uHi.Dispose(); C.Dispose(); d.Dispose();

            mpcX0 = new NativeArray<float>(StateCount, Allocator.Persistent);
            mpcRef = new NativeArray<float>(horizon * StateCount, Allocator.Persistent);
            mpcU0 = new NativeArray<float>(InputCount, Allocator.Persistent);
            mpcSoft = new NativeArray<float>(SoftRows, Allocator.Persistent);
            mpcOut = new NativeArray<MPCInfo>(1, Allocator.Persistent);
            previewOut = new NativeArray<float>(4, Allocator.Persistent);
        }

        /// <summary>
        /// Six decoupled double integrators — along-track, lateral, vertical, roll, pitch, yaw — driven
        /// by ACCELERATION inputs: B = dt on each rate row, no mass/inertia scaling (chosen so B's
        /// entries stay near O(dt) instead of O(dt/mass), which would leave the terminal Riccati solve
        /// badly scaled for a heavy hull). Gravity feedforward and the accel -> force/torque conversion
        /// are the caller's job, not this model's. Time-invariant: nothing here depends on the
        /// operating point. Allocates A/B/Q/R fresh with <paramref name="allocator"/> (caller disposes).
        ///
        /// <paramref name="qPos"/> weights the three integrator modes (forward, lateral, heading).
        /// It must be nonzero: at exactly zero those modes are undetectable from Q and the terminal
        /// DARE is ill-posed.
        /// </summary>
        public static void BuildMpcModel(float dt, float qPos, float qVel, float qVert, float qVertRate,
            float qTilt, float qTiltRate, float qYawRate, float rLinear, float rAngular,
            Allocator allocator, out floatMxN A, out floatMxN B, out floatMxN Q, out floatMxN R)
        {
            // Cleared, NOT uninitialized: everything below writes only the nonzero entries, and the
            // four-argument overload's flag means "leave it uninitialized".
            A = new floatMxN(StateCount, StateCount, allocator);
            B = new floatMxN(StateCount, InputCount, allocator);
            Q = new floatMxN(StateCount, StateCount, allocator);
            R = new floatMxN(InputCount, InputCount, allocator);

            for (int i = 0; i < StateCount; i++) A[i, i] = 1f;
            for (int p = 0; p < InputCount; p++)
            {
                A[2 * p, 2 * p + 1] = dt;      // position row integrates its own rate
                B[2 * p + 1, p] = dt;          // rate row is driven by its own acceleration
            }

            Q[SFwd, SFwd] = qPos; Q[VFwd, VFwd] = qVel;
            Q[SLat, SLat] = qPos; Q[VLat, VLat] = qVel;
            Q[SVert, SVert] = qVert; Q[VVert, VVert] = qVertRate;
            Q[Roll, Roll] = qTilt; Q[RollRate, RollRate] = qTiltRate;
            Q[Pitch, Pitch] = qTilt; Q[PitchRate, PitchRate] = qTiltRate;
            Q[Yaw, Yaw] = qPos; Q[YawRate, YawRate] = qYawRate;

            R[AFwd, AFwd] = rLinear; R[ALat, ALat] = rLinear; R[AVert, AVert] = rLinear;
            R[AlphaRoll, AlphaRoll] = rAngular;
            R[AlphaPitch, AlphaPitch] = rAngular;
            R[AlphaYaw, AlphaYaw] = rAngular;
        }

        void OnDestroy()
        {
            mpc.Dispose();
            if (mpcX0.IsCreated) mpcX0.Dispose();
            if (mpcRef.IsCreated) mpcRef.Dispose();
            if (mpcU0.IsCreated) mpcU0.Dispose();
            if (mpcSoft.IsCreated) mpcSoft.Dispose();
            if (mpcOut.IsCreated) mpcOut.Dispose();
            if (previewOut.IsCreated) previewOut.Dispose();
            if (hoverState.IsCreated) hoverState.Dispose();

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
                // for the next fixed step; mouse Y moves the ride-height SETPOINT the MPC is already
                // tracking, so climbing costs no extra control channel. A locked cursor makes
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
                Mpc = mpc,
                MpcX0 = mpcX0, MpcRef = mpcRef, MpcU0 = mpcU0, MpcSoft = mpcSoft,
                MpcOut = mpcOut, PreviewOut = previewOut, Horizon = horizon,
                Mass = hullMass, RollInertia = rollInertia, PitchInertia = pitchInertia,
                YawInertia = yawInertia, Gravity = -Physics.gravity.y,
                Dt = dt,

                Rpy = estimate.Rpy,
                GroundNormal = estimate.GroundNormal,
                VelWorld = estimate.Velocity,
                ForwardSpeed = estimate.ForwardSpeed,
                LateralSpeed = estimate.LateralSpeed,
                YawRate = estimate.YawRate,
                RollRate = estimate.RollRate,
                PitchRate = estimate.PitchRate,
                Clearance = estimate.Clearance,
                TiltCos = estimate.TiltCos,
                GroundValid = estimate.GroundValid,
                TargetRideHeight = targetRideHeight,

                ProxSensed = proxSensed, CollisionMargin = collisionMargin,

                Controls = controls, AllocOut = allocOut, WrenchOut = wrenchOut,
                Settings = thrusters, Health = thrusterHealth,
                MountX = mountX, MountY = mountY, MountZ = mountZ, MountArm = mountArm,
                DriveInput = Input.GetAxis("Vertical"),
                StrafeInput = lastStrafe,
                SteerInput = lastSteer,
                BrakeInput = lastBrake,
                MaxFwdSpeed = maxFwdSpeed, MaxLatSpeed = maxLatSpeed, MaxYawRate = maxYawRate,

                NozzleHeights = nozzleHeights,
                NozzleRadius = groundEffect ? nozzleRadius : 0f,
                GroundOut = groundOut,
            };

            sw.Restart();
            IJobExtensions.RunByRef(ref job);
            sw.Stop();
            ctrlMs = (float)sw.Elapsed.TotalMilliseconds;

            // The warm-start plan and working set advanced inside the job; without this copy every
            // step would solve cold.
            mpc = job.Mpc;

            LogOnceIfDiverged(mpcOut[0].status != MPCStatus.Fallback, ref hoverDivergedLogged, "hover MPC");
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
            GUILayout.Label($"Hover tank over terrain — {estMs:F3} ms sense+EKF, {ctrlMs:F3} ms control (15-state EKF + {StateCount}-state MPC over {horizon} stages + 12-control allocation QP)");
            GUILayout.Label($"Mouse X turn   Mouse Y climb   W/S drive   Q/E strafe   A/D yaw   SPACE brake   ESC {(mouseCaptured ? "release cursor" : "RESUME DRIVING")}");

            DrawEstimatorPanel();

            MPCInfo m = mpcOut[0];
            GUILayout.Label($"MPC: {m.status}  pivots={m.iterations}  activeSetChanges={m.activeSetChanges}  slack={m.maxSlackViolation:F3} m  horizon {horizon * Time.fixedDeltaTime:F2} s");
            GUILayout.Label($"preview: ground {previewOut[1]:F1} deg along track, rising {previewOut[0]:+0.00;-0.00;0.00} m by the horizon end   —   the tank climbs before the clearance error appears");
            GUILayout.Label($"anti-collision: tightest {ProximityRig.Names[(int)previewOut[2]]} at {previewOut[3]:F1} m of room (margin {collisionMargin:F1} m)");
            GUILayout.Label($"commanded accel  fwd {mpcU0[AFwd],6:F2}  lat {mpcU0[ALat],6:F2}  vert {mpcU0[AVert],6:F2} m/s^2   roll {mpcU0[AlphaRoll],6:F2}  pitch {mpcU0[AlphaPitch],6:F2}  yaw {mpcU0[AlphaYaw],6:F2} rad/s^2");

            QPInfo alloc = allocOut[0];
            GUILayout.Label($"alloc QP: {alloc.status}  pivots={alloc.iterations}  obj={alloc.objective:E2}");
            GUILayout.Label($"force  N   lateral {wrenchOut[6]:F0}/{wrenchOut[0]:F0}   lift {wrenchOut[7]:F0}/{wrenchOut[1]:F0}   drive {wrenchOut[8]:F0}/{wrenchOut[2]:F0}   (achieved/demanded)");
            GUILayout.Label($"torque Nm  pitch {wrenchOut[9]:F0}/{wrenchOut[3]:F0}   yaw {wrenchOut[10]:F0}/{wrenchOut[4]:F0}   roll {wrenchOut[11]:F0}/{wrenchOut[5]:F0}");
            GUILayout.Label($"yaw axis: {YawOwner()}   speed {estimate.ForwardSpeed,5:F1} m/s   strafe {estimate.LateralSpeed,5:F1} m/s   yaw rate {estimate.YawRate * Mathf.Rad2Deg,5:F0} deg/s   (all estimated)");
            GUILayout.Label($"ride height cmd {targetRideHeight:F2} m   sensed {estimate.Clearance:F2} m   lidar {GroundLabel()}   mouse {(mouseCaptured ? "CAPTURED" : "released")}");

            GUILayout.BeginHorizontal();
            groundEffect = GUILayout.Toggle(groundEffect, "ground effect", GUILayout.Width(110));
            GUILayout.Label(groundEffect
                ? $"mean x{MeanGroundGain():F3} (allocation only)   FL {groundOut[0]:F2} FR {groundOut[1]:F2} BL {groundOut[2]:F2} BR {groundOut[3]:F2}   clamp x{GroundEffect.MaxFactor:F2} under {GroundEffect.ClampHeight(nozzleRadius):F2} m"
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
            lookSensitivity = LabeledSlider($"mouse sens {lookSensitivity:F2}", lookSensitivity, 0.01f, 1f);
            climbSensitivity = LabeledSlider($"climb sens {climbSensitivity:F2}m", climbSensitivity, 0.01f, 1f);
            maxFwdSpeed = LabeledSlider($"top speed {maxFwdSpeed:F0} m/s", maxFwdSpeed, 2f, 40f);
            maxLatSpeed = LabeledSlider($"top strafe {maxLatSpeed:F0} m/s", maxLatSpeed, 1f, 30f);
            maxYawRate = LabeledSlider($"top yaw rate {maxYawRate * Mathf.Rad2Deg:F0} deg/s", maxYawRate, 0.2f, 4f);
            collisionMargin = LabeledSlider($"collision margin {collisionMargin:F1}m", collisionMargin, 0.2f, 6f);
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

        // What the yaw-rate REFERENCE is this step, and whether the allocation actually delivered the
        // torque the MPC asked for. Yaw authority is bought with forward thrust the hover loop also
        // wants, so a hard mouse flick can exceed what the rig can produce — say so rather than letting
        // the hull drift off heading unexplained.
        string YawOwner()
        {
            string owner;
            if (lastBrake) owner = "BRAKE (space): rate ref 0";
            else if (Mathf.Abs(lastSteer) > 0.02f) owner = "DRIVER (mouse + A/D)";
            else if (!mouseCaptured) owner = "cursor released";
            else owner = "hands off: rate ref 0";

            float torqueScale = hullMass * -Physics.gravity.y * mountArm;
            bool shortfall = Mathf.Abs(wrenchOut[10] - wrenchOut[4]) > 0.05f * torqueScale;
            return shortfall ? owner + "  [YAW SATURATED]" : owner;
        }

        /// <summary>Mean of the four nozzle augmentations. Panel only: the augmentation reaches the
        /// allocation's Jacobian per nozzle and never enters the MPC's model.</summary>
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
    /// it did not produce. One receding-horizon <see cref="MPC.solve"/> over 12 body states produces
    /// all six acceleration commands at once; those become the demanded hull-frame
    /// <see cref="GimbalWrench"/>, which <see cref="GimbalAllocation.Solve"/> then shares across 4
    /// pitch angles, 4 yaw angles and 4 throttles.
    ///
    /// The MPC's own model is LTI and is built ONCE (see <see cref="HoverTankMPCDemo.BuildMpcModel"/>);
    /// only the reference, the initial state and the soft-row bounds are rebuilt per step. Terrain
    /// ahead enters as a per-stage <see cref="SVert"/> reference and anti-collision as a soft row whose
    /// bound tracks the proximity rangers through <see cref="MPC.setSoftBound"/> — neither re-condenses.
    ///
    /// The <see cref="GroundEffect"/> augmentation enters the allocation's Jacobian, where it is exact.
    /// It deliberately does NOT enter this model: doing so would make B time-varying and cost a full
    /// re-condense every step, and the allocation already inverts the gain exactly, so a unit of
    /// commanded acceleration buys one unit at any height.
    ///
    /// Caller must RunByRef and copy Mpc back — it carries the warm-start plan and working set.
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct HoverTankMPCStepJob : IJob
    {
        /// <summary>Smallest vertical component the fitted ground normal is divided by. The slope is a
        /// ratio against it, so a near-vertical face would otherwise demand an unbounded climb.</summary>
        public const float MinGroundNormalY = 0.3f;

        /// <summary>Largest along-track ground gradient the preview will act on, rise over run — about
        /// 56 degrees. Anything steeper is a wall, not a hill the tank can follow.</summary>
        public const float MaxSlope = 1.5f;

        // ---- the MPC ----
        /// <summary>Carries the warm-start plan and working set; the caller must copy it back.</summary>
        public floatMPCState Mpc;
        /// <summary>Length <see cref="HoverTankMPCDemo.StateCount"/>, rebuilt here every step.</summary>
        public NativeArray<float> MpcX0;
        /// <summary>Length Horizon * StateCount — the per-stage reference, rebuilt here every step.</summary>
        public NativeArray<float> MpcRef;
        /// <summary>Length <see cref="HoverTankMPCDemo.InputCount"/> — the applied first input.</summary>
        public NativeArray<float> MpcU0;
        /// <summary>Length 4, the soft rows' bound in the order +fwd, -fwd, +lat, -lat.</summary>
        public NativeArray<float> MpcSoft;
        public NativeArray<MPCInfo> MpcOut;
        /// <summary>[0] predicted terrain rise at the horizon end (m), [1] along-track slope (deg),
        /// [2] tightest soft row, [3] that row's remaining clearance (m).</summary>
        public NativeArray<float> PreviewOut;
        public int Horizon;

        public float Mass, RollInertia, PitchInertia, YawInertia, Gravity;
        public float Dt;

        // ---- the estimate this step, and nothing else ----
        /// <summary>Estimated [roll, pitch, yaw], rad.</summary>
        public float3 Rpy;
        /// <summary>Fitted ground normal in WORLD axes.</summary>
        public float3 GroundNormal;
        /// <summary>Estimated world velocity, m/s — only the vertical component is read.</summary>
        public float3 VelWorld;
        public float ForwardSpeed, LateralSpeed, YawRate, RollRate, PitchRate;
        /// <summary>Ride height, the cosine of hull tilt from world up (for the gravity feedforward),
        /// and the commanded ride height the preview is written against.</summary>
        public float Clearance, TiltCos, TargetRideHeight;
        public bool GroundValid;

        // ---- anti-collision ----
        /// <summary>Ranger readings in <see cref="ProximityRig"/> order: fwd, back, left, right.</summary>
        [ReadOnly] public NativeArray<float> ProxSensed;
        /// <summary>Metres held back from whatever a ranger reports.</summary>
        public float CollisionMargin;

        // ---- driver ----
        public float DriveInput, StrafeInput, SteerInput;
        public bool BrakeInput;
        public float MaxFwdSpeed, MaxLatSpeed, MaxYawRate;

        // thruster allocation
        public NativeArray<float> Controls;
        public NativeArray<QPInfo> AllocOut;
        public NativeArray<float> WrenchOut;
        public GimbalSettings Settings;
        public float4 Health, MountX, MountY, MountZ;
        public float MountArm;

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

            // ---- x0: four states are ZERO by construction ----
            // sFwd/sLat/sVert/yaw are displacements FROM NOW, and the frame is re-anchored every solve.
            // That is what lets a range measured now be written straight against a predicted state.
            int n = HoverTankMPCDemo.StateCount;
            for (int i = 0; i < n; i++) MpcX0[i] = 0f;
            MpcX0[HoverTankMPCDemo.VFwd] = ForwardSpeed;
            MpcX0[HoverTankMPCDemo.VLat] = LateralSpeed;
            MpcX0[HoverTankMPCDemo.VVert] = VelWorld.y;
            MpcX0[HoverTankMPCDemo.Roll] = Rpy.x;
            MpcX0[HoverTankMPCDemo.RollRate] = RollRate;
            MpcX0[HoverTankMPCDemo.Pitch] = Rpy.y;
            MpcX0[HoverTankMPCDemo.PitchRate] = PitchRate;
            MpcX0[HoverTankMPCDemo.YawRate] = YawRate;

            // ---- terrain preview: the anticipation an LQR structurally cannot have ----
            // Ride height at stage k is clearance + sVert_k - rise_k; setting that equal to the target
            // gives sVert_k = rise_k - e. So the CURRENT ride-height error enters as a reference offset
            // rather than as a state, and rising ground ahead is a moving reference rather than a
            // disturbance -- which matters, because MPC.solve has no disturbance term to put it in.
            //
            // The horizontal displacement the rise is read at comes from the CURRENT speed, a
            // constant-velocity extrapolation: the tank's own acceleration over the horizon is what the
            // solve is deciding, so using the reference speed here would assume its answer.
            float3x3 basis = Attitude.Matrix(Rpy);
            float3 fwdWorld = basis.c2, rightWorld = basis.c0;

            float slopeF = 0f, slopeR = 0f;
            if (GroundValid)
            {
                // Floor the normal's vertical component away from zero before dividing, then clamp the
                // slope itself: a near-vertical face in the fan must not demand an unbounded climb.
                float ny = GroundNormal.y;
                ny = ny >= 0f ? math.max(ny, MinGroundNormalY) : math.min(ny, -MinGroundNormalY);
                slopeF = math.clamp(-math.dot(GroundNormal, fwdWorld) / ny, -MaxSlope, MaxSlope);
                slopeR = math.clamp(-math.dot(GroundNormal, rightWorld) / ny, -MaxSlope, MaxSlope);
            }

            float rideError = Clearance - TargetRideHeight;
            float climbRate = slopeF * ForwardSpeed + slopeR * LateralSpeed;

            // Hands off is a zero velocity reference, and SPACE is the same thing on every axis at
            // once. The old brake / idle-damping ladder is gone: it existed because nothing regulated
            // these axes, and a second damping term would now fight the MPC for them.
            float vFwdRef = BrakeInput ? 0f : math.clamp(DriveInput, -1f, 1f) * MaxFwdSpeed;
            float vLatRef = BrakeInput ? 0f : math.clamp(StrafeInput, -1f, 1f) * MaxLatSpeed;
            float yawRateRef = BrakeInput ? 0f : math.clamp(SteerInput, -1f, 1f) * MaxYawRate;

            for (int k = 0; k < Horizon; k++)
            {
                int b = k * n;
                for (int i = 0; i < n; i++) MpcRef[b + i] = 0f;
                MpcRef[b + HoverTankMPCDemo.VFwd] = vFwdRef;
                MpcRef[b + HoverTankMPCDemo.VLat] = vLatRef;
                MpcRef[b + HoverTankMPCDemo.YawRate] = yawRateRef;
                MpcRef[b + HoverTankMPCDemo.SVert] = climbRate * ((k + 1) * Dt) - rideError;
                MpcRef[b + HoverTankMPCDemo.VVert] = climbRate;
            }

            // ---- anti-collision: the wall moves, the horizon does not get rebuilt ----
            // Shared across stages on purpose: the range is measured NOW and the state is displacement
            // from NOW, so one bound is the correct statement at every stage. A ranger that found
            // nothing reports its full range, which simply never binds.
            MpcSoft[0] = math.max(0f, ProxSensed[0] - CollisionMargin);   // +sFwd vs the forward ranger
            MpcSoft[1] = math.max(0f, ProxSensed[1] - CollisionMargin);   // -sFwd vs the rear ranger
            MpcSoft[2] = math.max(0f, ProxSensed[3] - CollisionMargin);   // +sLat is to the RIGHT: ranger 3
            MpcSoft[3] = math.max(0f, ProxSensed[2] - CollisionMargin);   // -sLat is to the LEFT: ranger 2
            var softBound = new floatN(MpcSoft);
            MPC.setSoftBound(ref Mpc, in softBound);

            var x0 = new floatN(MpcX0);
            var reference = new floatN(MpcRef);
            var u0 = new floatN(MpcU0);
            MpcOut[0] = MPC.solve(ref Mpc, in x0, in reference, ref u0);

            PreviewOut[0] = climbRate * (Horizon * Dt);
            PreviewOut[1] = math.degrees(math.atan(slopeF));
            int tightest = 0;
            for (int i = 1; i < 4; i++) if (MpcSoft[i] < MpcSoft[tightest]) tightest = i;
            PreviewOut[2] = tightest;
            PreviewOut[3] = MpcSoft[tightest];

            // ---- demanded hull-frame wrench ----
            // The gravity feedforward is divided by the hull's tilt cosine because thrust is bolted to
            // the hull and gravity is not; floored so a near-vertical hull cannot demand unbounded lift.
            var desired = new GimbalWrench
            {
                Lateral = Mass * MpcU0[HoverTankMPCDemo.ALat],
                Lift = Mass * (Gravity / math.max(TiltCos, 0.35f) + MpcU0[HoverTankMPCDemo.AVert]),
                Drive = Mass * MpcU0[HoverTankMPCDemo.AFwd],
                Pitch = PitchInertia * MpcU0[HoverTankMPCDemo.AlphaPitch],
                Yaw = YawInertia * MpcU0[HoverTankMPCDemo.AlphaYaw],
                Roll = RollInertia * MpcU0[HoverTankMPCDemo.AlphaRoll],
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

    }
}
