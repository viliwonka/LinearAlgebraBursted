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
    /// Hover tank (BF2142-style) on FOUR SERVO THRUSTERS, each with its own servo angle and throttle.
    /// Three layers run per fixed step:
    ///
    /// 1. A 6-state discrete LQR (height error, vertical velocity, roll, roll rate, pitch, pitch rate)
    ///    sensed from 4 corner-down raycasts, producing vertical/roll/pitch acceleration commands.
    /// 2. A control-allocation QP that turns those commands — plus the driver's forward and yaw demand
    ///    — into the 8 thruster controls, under servo range/rate and thrust range/rate limits. See
    ///    <see cref="ThrusterAllocation"/>: 8 controls against 5 reachable wrench components, so the
    ///    rig is over-actuated and the solve is what decides how the work is shared.
    /// 3. Two 2-state double-integrator servo LQRs (turret yaw, barrel pitch) tracking a moving target.
    ///
    /// WASD is a wrench demand routed through the allocation, not a force bolted onto the hull, so the
    /// tank drives by tilting its thrusters and the QP cancels the pitch-up torque that creates.
    /// Self-assembles ground/hull/turret/barrel/target primitives in <see cref="Start"/> (sceneless,
    /// like the other demos).
    /// </summary>
    public class HoverTankDemo : MonoBehaviour
    {
        [Header("Hull")]
        [Range(300f, 5000f)] public float hullMass = 1500f;
        [Range(1f, 4f)] public float hullHalfWidth = 2f;
        [Range(1f, 6f)] public float hullHalfLength = 3f;
        [Range(0.5f, 2f)] public float hullHeight = 1f;
        [Range(0.5f, 6f)] public float targetRideHeight = 2f;
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

        [Header("Servo thrusters (allocated by QP)")]
        public ThrusterSettings thrusters = ThrusterSettings.Default;

        [Header("Turret servo (yaw)")]
        public Transform target;
        [Range(1f, 200f)] public float qYawAngle = 60f;
        [Range(0.1f, 50f)] public float qYawRate = 8f;
        [Range(0.01f, 10f)] public float rYawTorque = 0.3f;
        [Range(1f, 30f)] public float maxYawAccel = 8f;

        [Header("Barrel servo (pitch)")]
        [Range(1f, 200f)] public float qPitchAngle = 60f;
        [Range(0.1f, 50f)] public float qPitchRate = 8f;
        [Range(0.01f, 10f)] public float rPitchTorque = 0.3f;
        [Range(1f, 30f)] public float maxPitchAccel = 8f;
        [Range(-30f, 10f)] public float barrelMinPitchDeg = -5f;
        [Range(10f, 85f)] public float barrelMaxPitchDeg = 60f;

        [Header("Driver demand (routed through the allocation)")]
        [Range(1000f, 40000f)] public float driveForce = 9000f;
        [Range(1000f, 40000f)] public float steerTorque = 9000f;

        [Header("Auto target orbit (used when target is unassigned)")]
        public bool autoOrbitTarget = true;
        [Range(2f, 15f)] public float orbitRadius = 6f;
        [Range(0.5f, 6f)] public float orbitHeight = 3f;
        [Range(0.05f, 2f)] public float orbitSpeed = 0.5f;

        static readonly string[] MountNames = { "FL", "FR", "BL", "BR" };

        // self-assembled scene objects (Start)
        GameObject groundGO, hullGO, turretGO, barrelGO, autoTargetGO;
        Rigidbody rb;
        Vector3[] cornerLocal;     // FL, FR, BL, BR — METRIC offsets from the hull center of mass
        float cornerDX, cornerDZ;  // horizontal corner offsets shared by sensing + allocation
        float4 mountX, mountY, mountZ;
        float mountArm;            // lever arm the torque residual scale is measured against
        float4 thrusterHealth = new float4(1f);
        Vector3 spawnPosition;
        Vector3 orbitCenter;
        float orbitAngle;

        // hover loop buffers (persistent — never allocated inside the job)
        floatMxN hoverK;
        floatLQRState hoverLqr;
        NativeArray<float> cornerHeights;
        NativeArray<float> prevCornerHeights;
        NativeArray<float> hoverState;    // [height err, height rate, roll, roll rate, pitch, pitch rate]
        NativeArray<float> hoverOut;      // [0] iters [1] converged [2] residual [3] rank-deficient

        // allocation buffers
        NativeArray<float> controls;      // 4 servo angles (rad) then 4 throttles (fraction)
        NativeArray<QPInfo> allocOut;
        NativeArray<float> wrenchOut;     // 5 demanded then 5 achieved (N, N*m)

        // turret / barrel servo buffers
        floatMxN turretK, barrelK;
        floatLQRState turretLqr, barrelLqr;
        NativeArray<float> turretState;   // [0] yaw   (rad) [1] yaw rate
        NativeArray<float> barrelState;   // [0] pitch (rad, +up) [1] pitch rate
        NativeArray<float> turretOut;     // [0] converged
        NativeArray<float> barrelOut;     // [0] converged

        bool hoverDivergedLogged, allocFailedLogged, turretDivergedLogged, barrelDivergedLogged;
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

            controls = new NativeArray<float>(ThrusterAllocation.ControlCount, Allocator.Persistent);
            allocOut = new NativeArray<QPInfo>(1, Allocator.Persistent);
            wrenchOut = new NativeArray<float>(10, Allocator.Persistent);

            turretK = new floatMxN(1, 2, Allocator.Persistent);
            barrelK = new floatMxN(1, 2, Allocator.Persistent);
            turretLqr = new floatLQRState(2, Allocator.Persistent);
            barrelLqr = new floatLQRState(2, Allocator.Persistent);
            turretState = new NativeArray<float>(2, Allocator.Persistent);
            barrelState = new NativeArray<float>(2, Allocator.Persistent);
            turretOut = new NativeArray<float>(1, Allocator.Persistent);
            barrelOut = new NativeArray<float>(1, Allocator.Persistent);

            for (int i = 0; i < 4; i++) prevCornerHeights[i] = targetRideHeight;
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

            if (turretK.IsCreated) turretK.Dispose();
            if (barrelK.IsCreated) barrelK.Dispose();
            turretLqr.Dispose();
            barrelLqr.Dispose();
            if (turretState.IsCreated) turretState.Dispose();
            if (barrelState.IsCreated) barrelState.Dispose();
            if (turretOut.IsCreated) turretOut.Dispose();
            if (barrelOut.IsCreated) barrelOut.Dispose();
        }

        void BuildScene()
        {
            groundGO = GameObject.CreatePrimitive(PrimitiveType.Plane);
            groundGO.name = "HoverTank_Ground";
            groundGO.transform.position = Vector3.zero;
            groundGO.transform.localScale = new Vector3(8f, 1f, 8f);   // 80x80 units

            spawnPosition = new Vector3(0f, targetRideHeight + hullHeight * 0.5f, 0f);

            hullGO = GameObject.CreatePrimitive(PrimitiveType.Cube);
            hullGO.name = "HoverTank_Hull";
            hullGO.transform.localScale = new Vector3(2f * hullHalfWidth, hullHeight, 2f * hullHalfLength);
            hullGO.transform.position = spawnPosition;
            hullGO.GetComponent<Renderer>().material.color = new Color(0.25f, 0.55f, 0.3f);

            rb = hullGO.AddComponent<Rigidbody>();
            rb.mass = hullMass;
            rb.linearDamping = 0.05f;
            rb.angularDamping = 0.2f;
            rb.interpolation = RigidbodyInterpolation.Interpolate;

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

            mountX = new float4(cornerLocal[0].x, cornerLocal[1].x, cornerLocal[2].x, cornerLocal[3].x);
            mountY = new float4(cornerLocal[0].y, cornerLocal[1].y, cornerLocal[2].y, cornerLocal[3].y);
            mountZ = new float4(cornerLocal[0].z, cornerLocal[1].z, cornerLocal[2].z, cornerLocal[3].z);
            mountArm = math.max(cornerDX, cornerDZ);

            turretGO = GameObject.CreatePrimitive(PrimitiveType.Cube);
            turretGO.name = "HoverTank_Turret";
            turretGO.transform.SetParent(hullGO.transform, worldPositionStays: false);
            turretGO.transform.localScale = new Vector3(1.2f, 0.6f, 1.2f);
            turretGO.transform.localPosition = new Vector3(0f, hullHeight * 0.5f + 0.3f, 0f);
            turretGO.GetComponent<Renderer>().material.color = new Color(0.2f, 0.2f, 0.2f);
            Destroy(turretGO.GetComponent<Collider>());

            barrelGO = GameObject.CreatePrimitive(PrimitiveType.Cube);
            barrelGO.name = "HoverTank_Barrel";
            barrelGO.transform.SetParent(turretGO.transform, worldPositionStays: false);
            barrelGO.transform.localScale = new Vector3(0.3f, 0.3f, 2.5f);
            barrelGO.transform.localPosition = new Vector3(0f, 0f, 1.25f);
            barrelGO.GetComponent<Renderer>().material.color = new Color(0.1f, 0.1f, 0.1f);
            Destroy(barrelGO.GetComponent<Collider>());

            orbitCenter = new Vector3(spawnPosition.x, 0f, spawnPosition.z);

            if (target == null)
            {
                autoTargetGO = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                autoTargetGO.name = "HoverTank_Target";
                autoTargetGO.transform.localScale = Vector3.one * 0.5f;
                autoTargetGO.GetComponent<Renderer>().material.color = Color.red;
                Destroy(autoTargetGO.GetComponent<Collider>());
                autoTargetGO.transform.position = orbitCenter + new Vector3(orbitRadius, orbitHeight, 0f);
            }
        }

        // cornerLocal is metric; the hull cube carries a render scale, so TransformPoint would
        // multiply the offsets by it and put the mounts (and the lever arms the allocation is
        // solved against) outside the hull.
        Vector3 MountWorld(int i) => hullGO.transform.position + hullGO.transform.rotation * cornerLocal[i];

        void ResetControls()
        {
            float maxThrust = Mathf.Max(thrusters.maxThrust, 1f);
            float trim = Mathf.Clamp(
                hullMass * -Physics.gravity.y / (4f * maxThrust),
                Mathf.Clamp01(thrusters.minThrust / maxThrust), 1f);

            for (int i = 0; i < 4; i++)
            {
                controls[i] = 0f;
                controls[4 + i] = trim;
            }
        }

        void FixedUpdate()
        {
            float dt = Time.fixedDeltaTime;
            rb.mass = hullMass;   // keep the real body in sync with the slider used by the linearization

            // autoTargetGO is only created when target started null (BuildScene); a target
            // assigned in the inspector and destroyed later at runtime leaves autoTargetGO null too.
            if (target == null && autoOrbitTarget && autoTargetGO != null)
            {
                orbitAngle += orbitSpeed * dt;
                autoTargetGO.transform.position = orbitCenter + new Vector3(
                    orbitRadius * math.cos(orbitAngle), orbitHeight, orbitRadius * math.sin(orbitAngle));
            }

            // ---- sense: 4 corner-down raycasts ----
            for (int i = 0; i < 4; i++)
            {
                Vector3 world = MountWorld(i) + Vector3.down * 0.02f;
                cornerHeights[i] = Physics.Raycast(world, Vector3.down, out RaycastHit hit, rayLength)
                    ? hit.distance : rayLength;
            }

            // ---- turret/barrel target direction, in hull-local space ----
            // (falls back to holding the current aim if neither an assigned target nor the
            // self-created one is available, e.g. a user-assigned target destroyed at runtime)
            Transform aimTarget = target != null ? target : (autoTargetGO != null ? autoTargetGO.transform : null);
            float desiredYaw = turretState[0];
            float desiredPitch = barrelState[0];
            if (aimTarget != null)
            {
                Vector3 localDir = hullGO.transform.InverseTransformDirection(aimTarget.position - hullGO.transform.position);
                desiredYaw = math.atan2(localDir.x, localDir.z);
                float horiz = math.sqrt(localDir.x * localDir.x + localDir.z * localDir.z);
                desiredPitch = math.atan2(localDir.y, math.max(horiz, 1e-4f));
            }

            var job = new HoverTankStepJob
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
                DriveDemand = Input.GetAxis("Vertical") * driveForce,
                YawDemand = Input.GetAxis("Horizontal") * steerTorque,
                TiltCos = Vector3.Dot(hullGO.transform.up, Vector3.up),

                TurretState = turretState, TurretK = turretK, TurretLqrState = turretLqr, TurretOut = turretOut,
                QYawAngle = qYawAngle, QYawRate = qYawRate, RYawTorque = rYawTorque, MaxYawAccel = maxYawAccel,
                DesiredYaw = desiredYaw,

                BarrelState = barrelState, BarrelK = barrelK, BarrelLqrState = barrelLqr, BarrelOut = barrelOut,
                QPitchAngle = qPitchAngle, QPitchRate = qPitchRate, RPitchTorque = rPitchTorque, MaxPitchAccel = maxPitchAccel,
                DesiredPitch = desiredPitch,
                BarrelMinPitch = barrelMinPitchDeg * Mathf.Deg2Rad, BarrelMaxPitch = barrelMaxPitchDeg * Mathf.Deg2Rad,
            };

            var sw = Stopwatch.StartNew();
            IJobExtensions.RunByRef(ref job);
            sw.Stop();
            frameMs = (float)sw.Elapsed.TotalMilliseconds;

            hoverLqr = job.HoverLqrState;
            turretLqr = job.TurretLqrState;
            barrelLqr = job.BarrelLqrState;

            LogOnceIfDiverged(hoverOut[1] == 1f, ref hoverDivergedLogged, "hover LQR");
            LogOnceIfDiverged(allocOut[0].status == QPStatus.Optimal, ref allocFailedLogged, "allocation QP");
            LogOnceIfDiverged(turretOut[0] == 1f, ref turretDivergedLogged, "turret yaw LQR");
            LogOnceIfDiverged(barrelOut[0] == 1f, ref barrelDivergedLogged, "barrel pitch LQR");

            // ---- apply thrust: one force per thruster, at its mount, along its servo direction ----
            // AddForceAtPosition reproduces both the force and its moment about the center of mass,
            // which is the wrench the allocation solved for.
            for (int i = 0; i < 4; i++)
            {
                float3 dir = ThrusterAllocation.ForceDirection(controls[i]);
                float magnitude = controls[4 + i] * thrusters.maxThrust * thrusterHealth[i];
                Vector3 worldDir = hullGO.transform.TransformDirection(new Vector3(dir.x, dir.y, dir.z));
                rb.AddForceAtPosition(worldDir * magnitude, MountWorld(i), ForceMode.Force);
            }

            // ---- apply turret/barrel kinematic pose (servo states, not physics) ----
            turretGO.transform.localRotation = Quaternion.Euler(0f, turretState[0] * Mathf.Rad2Deg, 0f);
            barrelGO.transform.localRotation = Quaternion.Euler(-barrelState[0] * Mathf.Rad2Deg, 0f, 0f);
        }

        static void LogOnceIfDiverged(bool converged, ref bool alreadyLogged, string label)
        {
            if (converged) { alreadyLogged = false; return; }
            if (alreadyLogged) return;
            UnityEngine.Debug.LogWarning($"HoverTankDemo: {label} did not converge, holding the last solution");
            alreadyLogged = true;
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying || hullGO == null) return;

            Gizmos.color = Color.cyan;
            for (int i = 0; i < 4; i++)
            {
                Vector3 world = MountWorld(i);
                Gizmos.DrawLine(world, world + Vector3.down * cornerHeights[i]);
                Gizmos.DrawSphere(world + Vector3.down * cornerHeights[i], 0.08f);
            }

            float angLo = Mathf.Min(thrusters.servoMinDeg, thrusters.servoMaxDeg) * Mathf.Deg2Rad;
            float angHi = Mathf.Max(thrusters.servoMinDeg, thrusters.servoMaxDeg) * Mathf.Deg2Rad;

            for (int i = 0; i < 4; i++)
            {
                Vector3 mount = MountWorld(i);
                bool live = thrusterHealth[i] > 0f;

                // servo travel arc, swept on the exhaust side
                Gizmos.color = new Color(0.4f, 0.4f, 0.45f);
                Vector3 prev = ExhaustPoint(mount, angLo, 0.7f);
                for (int k = 1; k <= 10; k++)
                {
                    Vector3 next = ExhaustPoint(mount, Mathf.Lerp(angLo, angHi, k / 10f), 0.7f);
                    Gizmos.DrawLine(prev, next);
                    prev = next;
                }

                // commanded exhaust, length proportional to throttle
                float throttle = controls[4 + i];
                Gizmos.color = live ? Color.Lerp(Color.green, Color.red, throttle) : Color.gray;
                Gizmos.DrawLine(mount, ExhaustPoint(mount, controls[i], 0.5f + 3f * throttle));
                Gizmos.DrawSphere(mount, 0.09f);
            }

            Gizmos.color = Color.yellow;
            Transform aimTarget = target != null ? target : (autoTargetGO != null ? autoTargetGO.transform : null);
            if (aimTarget != null)
                Gizmos.DrawLine(barrelGO.transform.position, aimTarget.position);
        }

        // The nozzle points opposite the force it produces: down at servo angle 0.
        Vector3 ExhaustPoint(Vector3 mount, float angle, float length)
        {
            float3 dir = ThrusterAllocation.ForceDirection(angle);
            return mount + hullGO.transform.TransformDirection(new Vector3(-dir.x, -dir.y, -dir.z)) * length;
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 520, 580), GUI.skin.box);
            GUILayout.Label($"Hover tank — {frameMs:F3} ms/frame (3x6 hover LQR + 8-control allocation QP + 2x servo LQR)");
            GUILayout.Label($"hover: converged={hoverOut[1] == 1f}  iters={hoverOut[0]:F0}  residual={hoverOut[2]:E1}");
            GUILayout.Label($"state: h={hoverState[0]:F2} v={hoverState[1]:F2} roll={hoverState[2] * Mathf.Rad2Deg:F1} pitch={hoverState[4] * Mathf.Rad2Deg:F1}");

            QPInfo alloc = allocOut[0];
            GUILayout.Label($"alloc QP: {alloc.status}  pivots={alloc.iterations}  obj={alloc.objective:E2}");
            GUILayout.Label($"force  N   lift {wrenchOut[5]:F0}/{wrenchOut[0]:F0}   drive {wrenchOut[6]:F0}/{wrenchOut[1]:F0}   (achieved/demanded)");
            GUILayout.Label($"torque Nm  pitch {wrenchOut[7]:F0}/{wrenchOut[2]:F0}   yaw {wrenchOut[8]:F0}/{wrenchOut[3]:F0}   roll {wrenchOut[9]:F0}/{wrenchOut[4]:F0}");

            for (int i = 0; i < 4; i++)
            {
                float throttle = controls[4 + i];
                GUILayout.Label($"{MountNames[i]}  servo {controls[i] * Mathf.Rad2Deg,6:F1} deg   thrust {throttle * thrusters.maxThrust,7:F0} N ({throttle * 100f:F0}%)"
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

            GUILayout.Label($"turret: yaw={turretState[0] * Mathf.Rad2Deg:F1}deg converged={turretOut[0] == 1f}   barrel: pitch={barrelState[0] * Mathf.Rad2Deg:F1}deg converged={barrelOut[0] == 1f}");

            targetRideHeight = LabeledSlider($"ride height {targetRideHeight:F2}", targetRideHeight, 0.5f, 6f);
            qHeight = LabeledSlider($"Q height {qHeight:F0}", qHeight, 1f, 200f);
            qTilt = LabeledSlider($"Q tilt {qTilt:F0}", qTilt, 1f, 300f);
            rThrust = LabeledSlider($"R accel {rThrust:F3}", rThrust, 0.001f, 5f);
            thrusters.servoMaxDeg = LabeledSlider($"servo range +{thrusters.servoMaxDeg:F0}deg", thrusters.servoMaxDeg, 0f, 85f);
            thrusters.servoRateDeg = LabeledSlider($"servo rate {thrusters.servoRateDeg:F0}deg/s", thrusters.servoRateDeg, 15f, 720f);
            thrusters.thrustRate = LabeledSlider($"thrust rate {thrusters.thrustRate:F0}N/s", thrusters.thrustRate, 2000f, 400000f);

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
                turretState[0] = 0f; turretState[1] = 0f;
                barrelState[0] = 0f; barrelState[1] = 0f;
                thrusterHealth = new float4(1f);
                ResetControls();
            }
            autoOrbitTarget = GUILayout.Toggle(autoOrbitTarget, "orbit target");
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
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
    /// the 6-state hover LQR (3 acceleration commands: vertical, roll, pitch); turns those plus the
    /// driver's forward/yaw demand into a hull-frame <see cref="HoverWrench"/>; allocates that wrench
    /// onto 4 servo angles and 4 throttles with <see cref="ThrusterAllocation.Solve"/>; then
    /// warm-solves and Euler-integrates two independent 2-state double-integrator servo loops (turret
    /// yaw, barrel pitch). All three LQR solves re-run every step (warm <see cref="floatLQRState"/>,
    /// cheap once converged) to showcase the warm-start path, matching CartPole/Drone.
    ///
    /// Caller must RunByRef and copy the three LqrState fields back.
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct HoverTankStepJob : IJob
    {
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

        // thruster allocation
        public NativeArray<float> Controls;
        public NativeArray<QPInfo> AllocOut;
        public NativeArray<float> WrenchOut;
        public ThrusterSettings Settings;
        public float4 Health, MountX, MountY, MountZ;
        public float MountArm;
        public float DriveDemand, YawDemand;
        /// <summary>cos of the hull's tilt from world up, for the gravity feedforward.</summary>
        public float TiltCos;

        // turret yaw servo
        public NativeArray<float> TurretState;
        public floatMxN TurretK;
        public floatLQRState TurretLqrState;
        public NativeArray<float> TurretOut;
        public float QYawAngle, QYawRate, RYawTorque, MaxYawAccel, DesiredYaw;

        // barrel pitch servo
        public NativeArray<float> BarrelState;
        public floatMxN BarrelK;
        public floatLQRState BarrelLqrState;
        public NativeArray<float> BarrelOut;
        public float QPitchAngle, QPitchRate, RPitchTorque, MaxPitchAccel, DesiredPitch;
        public float BarrelMinPitch, BarrelMaxPitch;

        public void Execute()
        {
            // ---- reconstruct the 6-state hover/attitude estimate from corner heights ----
            // roll = rotation about the forward axis, pitch = rotation about the right axis;
            // both derived purely from differenced corner ride heights (and their finite-
            // difference rates), matching the torque sign convention the allocation uses.
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

            // ---- demanded hull-frame wrench ----
            // The gravity feedforward is divided by the hull's tilt cosine because thrust is bolted to
            // the hull and gravity is not; floored so a near-vertical hull cannot demand unbounded lift.
            var desired = new HoverWrench
            {
                Lift = Mass * (Gravity / math.max(TiltCos, 0.35f) + uVertAccel),
                Drive = DriveDemand,
                Pitch = PitchInertia * uPitchAccel,
                Yaw = YawDemand,
                Roll = RollInertia * uRollAccel,
            };

            // ---- allocation: 8 controls onto the 5 reachable wrench components ----
            var z = new floatN(Controls);   // view, no copy
            ThrusterRig rig = ThrusterAllocation.BuildRig(in Settings, MountX, MountY, MountZ, Health,
                in z, Dt, Mass * Gravity, MountArm);
            AllocOut[0] = ThrusterAllocation.Solve(in rig, in desired, ref z, 0);

            HoverWrench got = ThrusterAllocation.Wrench(in rig, in z);
            WrenchOut[0] = desired.Lift; WrenchOut[1] = desired.Drive; WrenchOut[2] = desired.Pitch;
            WrenchOut[3] = desired.Yaw; WrenchOut[4] = desired.Roll;
            WrenchOut[5] = got.Lift; WrenchOut[6] = got.Drive; WrenchOut[7] = got.Pitch;
            WrenchOut[8] = got.Yaw; WrenchOut[9] = got.Roll;

            // ---- turret yaw servo: 2-state double integrator tracking DesiredYaw ----
            BuildServoModel(Dt, QYawAngle, QYawRate, RYawTorque, Allocator.Temp, out var Ay, out var By, out var Qy, out var Ry);
            RiccatiInfo infoYaw = LQR.lqr(in Ay, in By, in Qy, in Ry, ref TurretK, ref TurretLqrState);
            TurretOut[0] = infoYaw ? 1f : 0f;
            Ay.Dispose(); By.Dispose(); Qy.Dispose(); Ry.Dispose();

            float yawErr = WrapAngle(TurretState[0] - DesiredYaw);
            float uYaw = -(TurretK[0, 0] * yawErr + TurretK[0, 1] * TurretState[1]);
            uYaw = math.clamp(uYaw, -MaxYawAccel, MaxYawAccel);
            TurretState[1] += Dt * uYaw;
            TurretState[0] = WrapAngle(TurretState[0] + Dt * TurretState[1]);

            // ---- barrel pitch servo: 2-state double integrator tracking DesiredPitch, hard-clamped ----
            BuildServoModel(Dt, QPitchAngle, QPitchRate, RPitchTorque, Allocator.Temp, out var Ap, out var Bp, out var Qp, out var Rp);
            RiccatiInfo infoPitch = LQR.lqr(in Ap, in Bp, in Qp, in Rp, ref BarrelK, ref BarrelLqrState);
            BarrelOut[0] = infoPitch ? 1f : 0f;
            Ap.Dispose(); Bp.Dispose(); Qp.Dispose(); Rp.Dispose();

            float pitchErr = BarrelState[0] - DesiredPitch;   // no wrap: range-limited, never spins
            float uPitch = -(BarrelK[0, 0] * pitchErr + BarrelK[0, 1] * BarrelState[1]);
            uPitch = math.clamp(uPitch, -MaxPitchAccel, MaxPitchAccel);
            BarrelState[1] += Dt * uPitch;
            BarrelState[0] += Dt * BarrelState[1];
            if (BarrelState[0] < BarrelMinPitch) { BarrelState[0] = BarrelMinPitch; BarrelState[1] = math.max(0f, BarrelState[1]); }
            if (BarrelState[0] > BarrelMaxPitch) { BarrelState[0] = BarrelMaxPitch; BarrelState[1] = math.min(0f, BarrelState[1]); }
        }

        static float WrapAngle(float a) => math.atan2(math.sin(a), math.cos(a));

        /// <summary>
        /// Discrete (Euler, zero-order-hold over <paramref name="dt"/>) hover/attitude model: three
        /// decoupled double integrators — height/vertical-velocity, roll/roll-rate, pitch/pitch-rate —
        /// driven directly by ACCELERATION inputs [vertical accel, roll angular accel, pitch angular
        /// accel]: B = dt on each rate row, no mass/inertia scaling (same kinematic convention as
        /// <see cref="BuildServoModel"/>, chosen so B's entries stay near O(dt) instead of O(dt/mass)
        /// — the latter leaves the DARE badly scaled for a heavy hull). Gravity feedforward and the
        /// accel -> force/torque conversion (via mass/rollInertia/pitchInertia) are the caller's job,
        /// not this model's. Allocates A/B/Q/R fresh with <paramref name="allocator"/> (caller
        /// disposes).
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

        /// <summary>
        /// Discrete (Euler, zero-order-hold) 2-state double integrator [angle, rate] driven directly
        /// by an angular-acceleration input (kinematic servo — no inertia term). Allocates A/B/Q/R
        /// fresh with <paramref name="allocator"/> (caller disposes).
        /// </summary>
        public static void BuildServoModel(float dt, float qAngle, float qRate, float rTorque,
            Allocator allocator, out floatMxN A, out floatMxN B, out floatMxN Q, out floatMxN R)
        {
            A = new floatMxN(2, 2, allocator);
            B = new floatMxN(2, 1, allocator);
            Q = new floatMxN(2, 2, allocator);
            R = new floatMxN(1, 1, allocator);

            A[0, 0] = 1f; A[0, 1] = dt; A[1, 1] = 1f;
            B[1, 0] = dt;

            Q[0, 0] = qAngle; Q[1, 1] = qRate;
            R[0, 0] = rTorque;
        }
    }
}
