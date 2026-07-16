using System.Diagnostics;
using LinearAlgebra;
using LinearAlgebra.Control;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// Planar quadrotor stabilized and steered by discrete LQR. State
    /// [x, z, theta, vx, vz, w], inputs [total thrust, torque] mapped to two
    /// clamped rotor forces. One Burst job per frame: warm Riccati re-solve
    /// (LQR.lqr + floatLQRState) around hover for the current sliders, then
    /// RK4 integration of the full nonlinear dynamics tracking a moving target
    /// with u = u_hover - K·(x - x_ref). Wind gusts on demand.
    /// </summary>
    public class DroneLQRDemo : MonoBehaviour
    {
        [Range(0.2f, 3f)] public float droneMass = 0.8f;
        [Range(0.05f, 1f)] public float inertia = 0.15f;
        [Range(0.1f, 1f)] public float armLength = 0.25f;
        [Range(0.1f, 100f)] public float qPosition = 20f;
        [Range(0.1f, 100f)] public float qAngle = 10f;
        [Range(0.01f, 10f)] public float rControl = 0.5f;
        [Range(2f, 30f)] public float maxRotorForce = 12f;
        public bool orbitTarget = true;

        const float SimDt = 1f / 240f;
        const int Substeps = 4;

        floatMxN K;                    // 2×6
        floatLQRState lqrState;
        NativeArray<float> state;      // [x, z, th, vx, vz, w]
        NativeArray<float> target;     // [x, z]
        NativeArray<float> outStats;   // [0] f1, [1] f2, [2] iters, [3] converged
        NativeArray<float> wind;       // [0] horizontal force
        float frameMs;
        readonly Stopwatch sw = new Stopwatch();

        void OnEnable()
        {
            K = new floatMxN(2, 6, Allocator.Persistent);
            lqrState = new floatLQRState(6, Allocator.Persistent);
            state = new NativeArray<float>(6, Allocator.Persistent);
            target = new NativeArray<float>(2, Allocator.Persistent);
            outStats = new NativeArray<float>(4, Allocator.Persistent);
            wind = new NativeArray<float>(1, Allocator.Persistent);
            state[1] = 1f;
        }

        void OnDisable()
        {
            K.Dispose(); lqrState.Dispose();
            state.Dispose(); target.Dispose(); outStats.Dispose(); wind.Dispose();
        }

        void Update()
        {
            if (orbitTarget)
            {
                target[0] = 2.2f * math.sin(0.4f * Time.time);
                target[1] = 1.5f + 0.8f * math.sin(0.9f * Time.time);
            }
            wind[0] *= 0.95f;   // gust decay

            var job = new DroneStepJob
            {
                K = K, LqrState = lqrState,
                State = state, Target = target, Out = outStats, Wind = wind,
                Mass = droneMass, Inertia = inertia, Arm = armLength,
                QPos = qPosition, QAngle = qAngle, RCost = rControl,
                MaxRotorForce = maxRotorForce,
                Dt = SimDt, Steps = Substeps,
            };

            sw.Restart();
            IJobExtensions.RunByRef(ref job);
            sw.Stop();
            frameMs = (float)sw.Elapsed.TotalMilliseconds;

            lqrState = job.LqrState;
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying || !state.IsCreated) return;

            float x = state[0], z = state[1], th = state[2];
            var pos = new Vector3(x, z, 0f);
            var right = new Vector3(math.cos(th), math.sin(th), 0f) * armLength;
            var up = new Vector3(-math.sin(th), math.cos(th), 0f);

            Gizmos.color = Color.white;
            Gizmos.DrawLine(pos - right, pos + right);
            Gizmos.DrawSphere(pos, 0.06f);
            Gizmos.color = Color.cyan;
            Gizmos.DrawLine(pos - right, pos - right + up * (outStats[0] / 20f));
            Gizmos.DrawLine(pos + right, pos + right + up * (outStats[1] / 20f));

            Gizmos.color = Color.green;
            Gizmos.DrawWireSphere(new Vector3(target[0], target[1], 0f), 0.12f);

            Gizmos.color = Color.gray;
            Gizmos.DrawLine(new Vector3(-4f, 0f, 0f), new Vector3(4f, 0f, 0f));
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 400, 290), GUI.skin.box);
            GUILayout.Label($"Drone LQR — {frameMs:F3} ms/frame (2×6 gain, warm Riccati)");
            GUILayout.Label($"rotors: L={outStats[0]:F1}N R={outStats[1]:F1}N   iters: {outStats[2]:F0}   converged: {outStats[3] == 1f}");
            GUILayout.Label($"pos=({state[0]:F2}, {state[1]:F2})  theta={state[2] * Mathf.Rad2Deg:F0}°   wind={wind[0]:F1}N");

            droneMass = LabeledSlider($"mass {droneMass:F2}", droneMass, 0.2f, 3f);
            inertia = LabeledSlider($"inertia {inertia:F2}", inertia, 0.05f, 1f);
            qPosition = LabeledSlider($"Q pos {qPosition:F0}", qPosition, 0.1f, 100f);
            rControl = LabeledSlider($"R {rControl:F2}", rControl, 0.01f, 10f);

            GUILayout.BeginHorizontal();
            orbitTarget = GUILayout.Toggle(orbitTarget, "orbit target");
            if (GUILayout.Button("Gust >")) wind[0] += 8f;
            if (GUILayout.Button("< Gust")) wind[0] -= 8f;
            if (GUILayout.Button("Reset"))
            {
                for (int i = 0; i < 6; i++) state[i] = 0f;
                state[1] = 1f;
            }
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        static float LabeledSlider(string label, float v, float lo, float hi)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, GUILayout.Width(110));
            v = GUILayout.HorizontalSlider(v, lo, hi, GUILayout.Width(240));
            GUILayout.EndHorizontal();
            return v;
        }
    }

    /// <summary>
    /// Warm LQR gain about hover, then RK4 nonlinear planar-quadrotor steps with
    /// per-rotor force clamping and wind. Caller must RunByRef + copy LqrState back.
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct DroneStepJob : IJob
    {
        public floatMxN K;
        public floatLQRState LqrState;
        public NativeArray<float> State;
        [ReadOnly] public NativeArray<float> Target;
        public NativeArray<float> Out;
        [ReadOnly] public NativeArray<float> Wind;
        public float Mass, Inertia, Arm, QPos, QAngle, RCost, MaxRotorForce, Dt;
        public int Steps;

        const float G = 9.81f;

        public void Execute()
        {
            float m = Mass, I = Inertia;
            float dt = Dt * Steps;

            // hover linearization: ax = -g·theta, az = du1/m, aw = u2/I  (Euler discretization)
            var A = new floatMxN(6, 6, Allocator.Temp);
            var B = new floatMxN(6, 2, Allocator.Temp);
            var Q = new floatMxN(6, 6, Allocator.Temp);
            var R = new floatMxN(2, 2, Allocator.Temp);

            for (int i = 0; i < 6; i++) A[i, i] = 1f;
            A[0, 3] = dt; A[1, 4] = dt; A[2, 5] = dt;
            A[3, 2] = -G * dt;
            B[4, 0] = dt / m;    // total-thrust delta -> vertical accel
            B[5, 1] = dt / I;    // torque -> angular accel

            Q[0, 0] = QPos; Q[1, 1] = QPos; Q[2, 2] = QAngle;
            Q[3, 3] = 1f; Q[4, 4] = 1f; Q[5, 5] = 1f;
            R[0, 0] = RCost; R[1, 1] = RCost * 4f;

            RiccatiInfo info = LQR.lqr(in A, in B, in Q, in R, ref K, ref LqrState);
            Out[2] = info.iterations;
            Out[3] = info ? 1f : 0f;

            float f1 = 0f, f2 = 0f;
            for (int s = 0; s < Steps; s++)
            {
                // u = u_hover - K (x - x_ref)
                float e0 = State[0] - Target[0];
                float e1 = State[1] - Target[1];
                float du1 = 0f, du2 = 0f;
                du1 -= K[0, 0] * e0 + K[0, 1] * e1 + K[0, 2] * State[2]
                     + K[0, 3] * State[3] + K[0, 4] * State[4] + K[0, 5] * State[5];
                du2 -= K[1, 0] * e0 + K[1, 1] * e1 + K[1, 2] * State[2]
                     + K[1, 3] * State[3] + K[1, 4] * State[4] + K[1, 5] * State[5];

                float u1 = m * G + du1;   // total thrust
                float u2 = du2;           // torque

                // map to rotor forces, clamp each to [0, max], map back
                f1 = math.clamp(0.5f * (u1 - u2 / Arm), 0f, MaxRotorForce);
                f2 = math.clamp(0.5f * (u1 + u2 / Arm), 0f, MaxRotorForce);
                u1 = f1 + f2;
                u2 = (f2 - f1) * Arm;

                var x = new float3x2(
                    new float3(State[0], State[1], State[2]),
                    new float3(State[3], State[4], State[5]));
                var k1 = Deriv(x, u1, u2);
                var k2 = Step(x, k1, 0.5f * Dt, u1, u2);
                var k3 = Step(x, k2, 0.5f * Dt, u1, u2);
                var k4 = Step(x, k3, Dt, u1, u2);

                float3 dp = Dt / 6f * (k1.c0 + 2f * k2.c0 + 2f * k3.c0 + k4.c0);
                float3 dv = Dt / 6f * (k1.c1 + 2f * k2.c1 + 2f * k3.c1 + k4.c1);
                State[0] += dp.x; State[1] += dp.y; State[2] += dp.z;
                State[3] += dv.x; State[4] += dv.y; State[5] += dv.z;

                if (State[1] < 0f) { State[1] = 0f; State[4] = math.max(0f, State[4]); }   // ground
            }
            Out[0] = f1; Out[1] = f2;
        }

        float3x2 Step(float3x2 x, float3x2 k, float h, float u1, float u2)
            => Deriv(new float3x2(x.c0 + h * k.c0, x.c1 + h * k.c1), u1, u2);

        float3x2 Deriv(float3x2 x, float u1, float u2)
        {
            float th = x.c0.z;
            float ax = (-u1 * math.sin(th) + Wind[0]) / Mass;
            float az = u1 * math.cos(th) / Mass - G;
            float aw = u2 / Inertia;
            return new float3x2(x.c1, new float3(ax, az, aw));
        }
    }
}
