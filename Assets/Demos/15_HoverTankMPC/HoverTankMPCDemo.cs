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
    /// Every axis reaches the thrusters through one control-allocation solve:
    ///
    /// 1. A 6-state discrete LQR (height error, vertical velocity, roll, roll rate, pitch, pitch rate)
    ///    sensed from 4 corner-down raycasts, producing vertical/roll/pitch acceleration commands.
    /// 2. A control-allocation QP that turns those commands plus the driver's forward/strafe/yaw
    ///    demand — and the braking and idle-damping terms, which are wrench demands like everything
    ///    else rather than forces written onto the rigid body — into the 12 thruster controls, under
    ///    servo range/rate and thrust range/rate limits. See <see cref="GimbalAllocation"/>: 12
    ///    controls against 6 wrench components, so the rig is over-actuated and the solve is what
    ///    decides how the work is shared.
    ///
    /// The corner raycasts are the only ground sense, and the attitude estimate is built from the
    /// DIFFERENCES between them, so it cannot separate hull tilt from terrain slope — see
    /// <see cref="HoverTankMPCStepJob.Execute"/> for what that costs over sloping ground.
    ///
    /// The four THRUST MOUNTS sit on the side flanks at hull mid-height (x = ±halfWidth, y = 0,
    /// z = ±halfLength), which is a different set of points from the four SENSE CORNERS on the bottom
    /// face that the raycasts fire from. A mount at y = 0 turns forward thrust into pure yaw, with no
    /// pitch moment for the hover loop to fight.
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

        static readonly string[] MountNames = { "FL", "FR", "BL", "BR" };
        static readonly Rect PanelRect = new Rect(10, 10, 560, 590);

        // self-assembled scene objects (Start)
        GameObject groundGO, hullGO, hullVisualGO;
        readonly Transform[] thrusterPivots = new Transform[4];
        readonly ParticleSystem[] plumes = new ParticleSystem[4];
        Material plumeMaterial;
        Texture2D plumeTexture;
        Camera chaseCam;
        Rigidbody rb;
        Vector3[] cornerLocal;     // FL, FR, BL, BR sense points on the bottom face
        Vector3[] mountLocal;      // FL, FR, BL, BR thrust mounts on the side flanks
        float cornerDX, cornerDZ;  // horizontal sense-corner offsets, shared with the attitude estimate
        float4 mountX, mountY, mountZ;
        float mountArm;            // lever arm the torque residual scale is measured against
        float4 thrusterHealth = new float4(1f);
        Vector3 spawnPosition;
        float lookX;               // mouse X accumulated since the last fixed step
        bool mouseCaptured = true; // driving mode; ESC releases the cursor to the panel
        bool4 cornerReturn;        // whether each corner ray found ground this step
        // last step's inputs and measured rates, cached so the readout can name the axis owner
        float lastSteer, lastStrafe, lastForwardSpeed, lastLateralSpeed, lastYawRate;
        bool lastBrake;

        // hover loop buffers (persistent — never allocated inside the job)
        floatMxN hoverK;
        floatLQRState hoverLqr;
        NativeArray<float> cornerHeights;
        NativeArray<float> prevCornerHeights;
        NativeArray<float> hoverState;    // [height err, height rate, roll, roll rate, pitch, pitch rate]
        NativeArray<float> hoverOut;      // [0] iters [1] converged [2] residual [3] rank-deficient

        // allocation buffers
        NativeArray<float> controls;      // 4 pitch angles, 4 yaw angles (rad), then 4 throttles
        NativeArray<QPInfo> allocOut;
        NativeArray<float> wrenchOut;     // 6 demanded then 6 achieved (N, N*m)

        bool hoverDivergedLogged, allocFailedLogged;
        float frameMs;

        void Start()
        {
            BuildScene();

            hoverK = new floatMxN(3, 6, Allocator.Persistent);
            hoverLqr = new floatLQRState(6, Allocator.Persistent);
            cornerHeights = new NativeArray<float>(4, Allocator.Persistent);
            prevCornerHeights = new NativeArray<float>(4, Allocator.Persistent);
            hoverState = new NativeArray<float>(6, Allocator.Persistent);
            hoverOut = new NativeArray<float>(4, Allocator.Persistent);

            controls = new NativeArray<float>(GimbalAllocation.ControlCount, Allocator.Persistent);
            allocOut = new NativeArray<QPInfo>(1, Allocator.Persistent);
            wrenchOut = new NativeArray<float>(2 * GimbalAllocation.WrenchRows, Allocator.Persistent);

            // Seed both height buffers at the setpoint: a corner that has never had a return still has
            // to hand the estimate something, and the setpoint is the one value that commands nothing.
            for (int i = 0; i < 4; i++)
            {
                cornerHeights[i] = targetRideHeight;
                prevCornerHeights[i] = targetRideHeight;
            }
            ResetControls();
        }

        void OnDestroy()
        {
            if (hoverK.IsCreated) hoverK.Dispose();
            hoverLqr.Dispose();
            if (cornerHeights.IsCreated) cornerHeights.Dispose();
            if (prevCornerHeights.IsCreated) prevCornerHeights.Dispose();
            if (hoverState.IsCreated) hoverState.Dispose();
            if (hoverOut.IsCreated) hoverOut.Dispose();

            if (controls.IsCreated) controls.Dispose();
            if (allocOut.IsCreated) allocOut.Dispose();
            if (wrenchOut.IsCreated) wrenchOut.Dispose();

            if (plumeMaterial != null) Destroy(plumeMaterial);
            if (plumeTexture != null) Destroy(plumeTexture);
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

            // Sense points stay on the BOTTOM FACE, inset so their down-rays clear the hull: they feed
            // the ride-height and attitude estimate and have nothing to do with where thrust is applied.
            cornerDX = hullHalfWidth * 0.9f;
            cornerDZ = hullHalfLength * 0.9f;
            float cornerY = -hullHeight * 0.5f;
            cornerLocal = new[]
            {
                new Vector3(-cornerDX, cornerY, +cornerDZ),   // FL
                new Vector3(+cornerDX, cornerY, +cornerDZ),   // FR
                new Vector3(-cornerDX, cornerY, -cornerDZ),   // BL
                new Vector3(+cornerDX, cornerY, -cornerDZ),   // BR
            };

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
        Vector3 CornerWorld(int i) => hullGO.transform.position + hullGO.transform.rotation * cornerLocal[i];
        Vector3 MountWorld(int i) => hullGO.transform.position + hullGO.transform.rotation * mountLocal[i];

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

            // ---- sense: 4 corner-down raycasts ----
            // A ray that finds nothing is a NO-RETURN, not a long reading. Over flat ground a miss was
            // unreachable; over terrain it is ordinary — past a drop-off, over the wall, or off the
            // edge of the field. Reporting rayLength for a miss would put several metres between that
            // corner and its neighbours, which the step job's differencing estimate reads as a violent
            // phantom tilt, so an unreturned corner HOLDS its last range the way a real range finder
            // holds its last good reading. Rates then come out as exactly zero for that corner rather
            // than as a step.
            for (int i = 0; i < 4; i++)
            {
                Vector3 world = CornerWorld(i) + Vector3.down * 0.02f;
                cornerReturn[i] = Physics.Raycast(world, Vector3.down, out RaycastHit hit, rayLength);
                if (cornerReturn[i]) cornerHeights[i] = hit.distance;
            }

            // Mouse X and A/D are two INPUT DEVICES on one axis, so they sum and clamp.
            float mouseSteer = lookX * lookSensitivity;
            lookX = 0f;
            lastSteer = Mathf.Clamp(Input.GetAxis("Horizontal") + mouseSteer, -1f, 1f);
            lastStrafe = (Input.GetKey(KeyCode.E) ? 1f : 0f) - (Input.GetKey(KeyCode.Q) ? 1f : 0f);
            lastBrake = Input.GetKey(KeyCode.Space);
            lastForwardSpeed = Vector3.Dot(rb.linearVelocity, hullGO.transform.forward);
            lastLateralSpeed = Vector3.Dot(rb.linearVelocity, hullGO.transform.right);
            lastYawRate = Vector3.Dot(rb.angularVelocity, hullGO.transform.up);

            var job = new HoverTankMPCStepJob
            {
                CornerHeights = cornerHeights, PrevCornerHeights = prevCornerHeights,
                HoverState = hoverState,
                HoverK = hoverK, HoverLqrState = hoverLqr, HoverOut = hoverOut,
                Mass = hullMass, RollInertia = rollInertia, PitchInertia = pitchInertia,
                Gravity = -Physics.gravity.y,
                QHeight = qHeight, QHeightRate = qHeightRate, QTilt = qTilt, QTiltRate = qTiltRate,
                RThrust = rThrust, RTorque = rTorque,
                TargetRideHeight = targetRideHeight, CornerDX = cornerDX, CornerDZ = cornerDZ, Dt = dt,

                Controls = controls, AllocOut = allocOut, WrenchOut = wrenchOut,
                Settings = thrusters, Health = thrusterHealth,
                MountX = mountX, MountY = mountY, MountZ = mountZ, MountArm = mountArm,
                DriveInput = Input.GetAxis("Vertical"), DriveForce = driveForce,
                StrafeInput = lastStrafe, StrafeForce = strafeForce,
                SteerInput = lastSteer, SteerTorque = steerTorque,
                BrakeInput = lastBrake,
                BrakeForce = brakeForce, BrakeGain = brakeGain, BrakeYawGain = brakeYawGain,
                IdleLinearGain = idleLinearGain, IdleAngularGain = idleAngularGain,
                ForwardSpeed = lastForwardSpeed,
                LateralSpeed = lastLateralSpeed,
                YawRate = lastYawRate,
                TiltCos = Vector3.Dot(hullGO.transform.up, Vector3.up),
            };

            var sw = Stopwatch.StartNew();
            IJobExtensions.RunByRef(ref job);
            sw.Stop();
            frameMs = (float)sw.Elapsed.TotalMilliseconds;

            hoverLqr = job.HoverLqrState;

            LogOnceIfDiverged(hoverOut[1] == 1f, ref hoverDivergedLogged, "hover LQR");
            LogOnceIfDiverged(allocOut[0].status == QPStatus.Optimal, ref allocFailedLogged, "allocation QP");

            // ---- apply thrust: one force per thruster, at its mount, along its gimbal direction ----
            // AddForceAtPosition reproduces both the force and its moment about the center of mass,
            // which is the wrench the allocation solved for.
            for (int i = 0; i < 4; i++)
            {
                float pitch = controls[i], yaw = controls[4 + i], throttle = controls[8 + i];
                float3 dir = GimbalAllocation.ForceDirection(pitch, yaw);
                float magnitude = throttle * thrusters.maxThrust * thrusterHealth[i];
                Vector3 worldDir = hullGO.transform.TransformDirection(new Vector3(dir.x, dir.y, dir.z));
                rb.AddForceAtPosition(worldDir * magnitude, MountWorld(i), ForceMode.Force);

                thrusterPivots[i].localRotation = GimbalRotation(pitch, yaw);
                UpdatePlume(i, throttle, thrusterHealth[i] > 0f);
            }
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

            for (int i = 0; i < 4; i++)
            {
                Vector3 world = CornerWorld(i);
                // A held corner is drawn dim: what the estimate is using is not what was measured.
                Gizmos.color = cornerReturn[i] ? Color.cyan : new Color(0.3f, 0.3f, 0.35f);
                Gizmos.DrawLine(world, world + Vector3.down * cornerHeights[i]);
                Gizmos.DrawSphere(world + Vector3.down * cornerHeights[i], 0.08f);
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
            GUILayout.Label($"Hover tank over terrain — {frameMs:F3} ms/frame (3x6 hover LQR + 12-control allocation QP)");
            GUILayout.Label($"Mouse X turn   Mouse Y climb   W/S drive   Q/E strafe   A/D yaw   SPACE brake   ESC {(mouseCaptured ? "release cursor" : "RESUME DRIVING")}");
            GUILayout.Label($"hover: converged={hoverOut[1] == 1f}  iters={hoverOut[0]:F0}  residual={hoverOut[2]:E1}   state: h={hoverState[0]:F2} roll={hoverState[2] * Mathf.Rad2Deg:F1} pitch={hoverState[4] * Mathf.Rad2Deg:F1}");

            QPInfo alloc = allocOut[0];
            GUILayout.Label($"alloc QP: {alloc.status}  pivots={alloc.iterations}  obj={alloc.objective:E2}");
            GUILayout.Label($"force  N   lateral {wrenchOut[6]:F0}/{wrenchOut[0]:F0}   lift {wrenchOut[7]:F0}/{wrenchOut[1]:F0}   drive {wrenchOut[8]:F0}/{wrenchOut[2]:F0}   (achieved/demanded)");
            GUILayout.Label($"torque Nm  pitch {wrenchOut[9]:F0}/{wrenchOut[3]:F0}   yaw {wrenchOut[10]:F0}/{wrenchOut[4]:F0}   roll {wrenchOut[11]:F0}/{wrenchOut[5]:F0}");
            GUILayout.Label($"yaw axis: {YawOwner()}   speed {lastForwardSpeed,5:F1} m/s   strafe {lastLateralSpeed,5:F1} m/s   yaw rate {lastYawRate * Mathf.Rad2Deg,5:F0} deg/s");
            GUILayout.Label($"ride height cmd {targetRideHeight:F2} m   ground {GroundLabel()}   mouse {(mouseCaptured ? "CAPTURED" : "released")}");

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
                for (int i = 0; i < 4; i++)
                {
                    cornerHeights[i] = targetRideHeight;
                    prevCornerHeights[i] = targetRideHeight;
                }
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

        // Corners whose ray came back this step. Anything less than 4 means part of the attitude
        // estimate is running on held ranges.
        string GroundLabel()
        {
            int hits = (cornerReturn[0] ? 1 : 0) + (cornerReturn[1] ? 1 : 0)
                     + (cornerReturn[2] ? 1 : 0) + (cornerReturn[3] ? 1 : 0);
            return hits == 4 ? "4/4 rays" : $"{hits}/4 rays  [HOLDING]";
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
    /// Per-fixed-step control law. Rebuilds the hover state from corner ride heights and warm-solves
    /// the 6-state hover LQR (3 acceleration commands: vertical, roll, pitch); resolves the driver's
    /// forward/strafe/yaw/brake inputs into the rest of the demanded hull-frame
    /// <see cref="GimbalWrench"/>; then allocates that onto 4 pitch angles, 4 yaw angles and 4
    /// throttles with <see cref="GimbalAllocation.Solve"/>. The LQR re-runs every step (warm
    /// <see cref="floatLQRState"/>, cheap once converged) to showcase the warm-start path.
    ///
    /// Caller must RunByRef and copy HoverLqrState back.
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct HoverTankMPCStepJob : IJob
    {
        /// <summary>Stick deflection below which the driver is considered hands-off the yaw axis.</summary>
        public const float StickDeadzone = 0.02f;

        // hover / attitude
        [ReadOnly] public NativeArray<float> CornerHeights;   // FL, FR, BL, BR
        public NativeArray<float> PrevCornerHeights;
        public NativeArray<float> HoverState;
        public floatMxN HoverK;
        public floatLQRState HoverLqrState;
        public NativeArray<float> HoverOut;
        public float Mass, RollInertia, PitchInertia, Gravity;
        public float QHeight, QHeightRate, QTilt, QTiltRate, RThrust, RTorque;
        public float TargetRideHeight, CornerDX, CornerDZ, Dt;

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

        public void Execute()
        {
            // ---- reconstruct the 6-state hover/attitude estimate from corner heights ----
            // roll = rotation about the forward axis, pitch = rotation about the right axis;
            // both derived purely from differenced corner ride heights (and their finite-
            // difference rates), matching the torque sign convention the allocation uses.
            //
            // KNOWN LIMIT of a ride-height-only estimate: a corner height difference is produced just
            // as readily by a SLOPING GROUND as by a tilted hull, and nothing here can tell the two
            // apart. Over terrain the loop therefore levels the hull to the LOCAL GROUND PLANE rather
            // than to gravity: the tank visibly banks into a hillside, holds that bank while it
            // traverses, and rolls back out on the far side. Its ride height stays right; its attitude
            // is wrong by the terrain gradient. Terrain gradients are shaped gentle enough
            // (TerrainField) that this stays a lean and not a divergence, which is why the steep face
            // and the wall sit well away from the spawn apron.
            float hFL = CornerHeights[0], hFR = CornerHeights[1], hBL = CornerHeights[2], hBR = CornerHeights[3];
            float heightErr = 0.25f * (hFL + hFR + hBL + hBR) - TargetRideHeight;

            float vFL = (hFL - PrevCornerHeights[0]) / Dt;
            float vFR = (hFR - PrevCornerHeights[1]) / Dt;
            float vBL = (hBL - PrevCornerHeights[2]) / Dt;
            float vBR = (hBR - PrevCornerHeights[3]) / Dt;
            float heightRate = 0.25f * (vFL + vFR + vBL + vBR);

            float hRight = 0.5f * (hFR + hBR), hLeft = 0.5f * (hFL + hBL);
            float roll = (hRight - hLeft) / (2f * CornerDX);
            float vRight = 0.5f * (vFR + vBR), vLeft = 0.5f * (vFL + vBL);
            float rollRate = (vRight - vLeft) / (2f * CornerDX);

            float hBack = 0.5f * (hBL + hBR), hFront = 0.5f * (hFL + hFR);
            float pitch = (hBack - hFront) / (2f * CornerDZ);
            float vBack = 0.5f * (vBL + vBR), vFront = 0.5f * (vFL + vFR);
            float pitchRate = (vBack - vFront) / (2f * CornerDZ);

            HoverState[0] = heightErr; HoverState[1] = heightRate;
            HoverState[2] = roll; HoverState[3] = rollRate;
            HoverState[4] = pitch; HoverState[5] = pitchRate;

            PrevCornerHeights[0] = hFL; PrevCornerHeights[1] = hFR;
            PrevCornerHeights[2] = hBL; PrevCornerHeights[3] = hBR;

            // ---- hover LQR: warm re-solve every step ----
            BuildHoverModel(Dt, QHeight, QHeightRate, QTilt, QTiltRate, RThrust, RTorque,
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
            GimbalRig rig = GimbalAllocation.BuildRig(in Settings, MountX, MountY, MountZ, Health,
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
        /// </summary>
        public static void BuildHoverModel(float dt,
            float qHeight, float qHeightRate, float qTilt, float qTiltRate, float rThrust, float rTorque,
            Allocator allocator, out floatMxN A, out floatMxN B, out floatMxN Q, out floatMxN R)
        {
            A = new floatMxN(6, 6, allocator);
            B = new floatMxN(6, 3, allocator);
            Q = new floatMxN(6, 6, allocator);
            R = new floatMxN(3, 3, allocator);

            for (int i = 0; i < 6; i++) A[i, i] = 1f;
            A[0, 1] = dt; A[2, 3] = dt; A[4, 5] = dt;

            B[1, 0] = dt; B[3, 1] = dt; B[5, 2] = dt;

            Q[0, 0] = qHeight; Q[1, 1] = qHeightRate;
            Q[2, 2] = qTilt; Q[3, 3] = qTiltRate;
            Q[4, 4] = qTilt; Q[5, 5] = qTiltRate;

            R[0, 0] = rThrust; R[1, 1] = rTorque; R[2, 2] = rTorque;
        }
    }
}
