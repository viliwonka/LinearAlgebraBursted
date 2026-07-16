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
    /// Inverted pendulum on a cart, stabilized by discrete LQR (LQR.lqr).
    /// Every frame one Burst job re-linearizes the dynamics around upright for the
    /// current (slider-adjustable) masses/length, warm-re-solves the Riccati
    /// equation via floatLQRState, then integrates the full nonlinear cart-pole
    /// with the resulting feedback u = -K·x (RK4 substeps, clamped force).
    /// Kick / Reset buttons disturb the pole; drag the physical parameters live
    /// and watch the gain adapt.
    /// </summary>
    public class CartPoleLQRDemo : MonoBehaviour
    {
        [Range(0.5f, 5f)] public float cartMass = 1f;
        [Range(0.05f, 2f)] public float poleMass = 0.3f;
        [Range(0.3f, 2f)] public float poleLength = 1f;
        [Range(0.1f, 100f)] public float qPosition = 10f;
        [Range(0.1f, 100f)] public float qAngle = 50f;
        [Range(0.01f, 10f)] public float rControl = 1f;
        [Range(5f, 100f)] public float maxForce = 30f;

        const float SimDt = 1f / 240f;   // RK4 substep
        const int Substeps = 4;          // 4 × 1/240 ≈ one 60 fps frame

        floatMxN K;
        floatLQRState lqrState;
        NativeArray<float> state;      // [p, v, theta, omega]
        NativeArray<float> outStats;   // [0] u, [1] lqr iters, [2] converged, [3] residual
        float frameMs;
        readonly Stopwatch sw = new Stopwatch();

        void OnEnable()
        {
            K = new floatMxN(1, 4, Allocator.Persistent);
            lqrState = new floatLQRState(4, Allocator.Persistent);
            state = new NativeArray<float>(4, Allocator.Persistent);
            outStats = new NativeArray<float>(4, Allocator.Persistent);
            state[2] = 0.25f;   // start tilted
        }

        void OnDisable()
        {
            K.Dispose();
            lqrState.Dispose();
            state.Dispose(); outStats.Dispose();
        }

        void Update()
        {
            var job = new CartPoleStepJob
            {
                K = K,
                LqrState = lqrState,
                State = state, Out = outStats,
                CartMass = cartMass, PoleMass = poleMass, PoleLength = poleLength,
                QPos = qPosition, QAngle = qAngle, RCost = rControl,
                MaxForce = maxForce,
                Dt = SimDt, Steps = Substeps,
            };

            sw.Restart();
            IJobExtensions.RunByRef(ref job);
            sw.Stop();
            frameMs = (float)sw.Elapsed.TotalMilliseconds;

            lqrState = job.LqrState;   // reclaim populated/warm scalar state
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying || !state.IsCreated) return;

            float p = state[0], th = state[2];
            var cart = new Vector3(p, 0f, 0f);
            var tip = cart + new Vector3(math.sin(th), math.cos(th), 0f) * poleLength;

            Gizmos.color = Color.gray;
            Gizmos.DrawLine(new Vector3(-6f, -0.15f, 0f), new Vector3(6f, -0.15f, 0f));
            Gizmos.color = Color.white;
            Gizmos.DrawCube(cart, new Vector3(0.5f, 0.25f, 0.3f));
            Gizmos.color = Color.yellow;
            Gizmos.DrawLine(cart, tip);
            Gizmos.DrawSphere(tip, 0.09f);

            // control force arrow, normalized so it stays full-scale at saturation regardless
            // of the maxForce slider
            Gizmos.color = Color.red;
            float u = outStats[0];
            Gizmos.DrawLine(cart, cart + new Vector3(u / maxForce, 0f, 0f));
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 400, 300), GUI.skin.box);
            GUILayout.Label($"Cart-pole LQR — {frameMs:F3} ms/frame (gain + 4 RK4 substeps)");
            GUILayout.Label($"u = {outStats[0]:F2} N   Riccati iters: {outStats[1]:F0}   converged: {outStats[2] == 1f}   residual: {outStats[3]:E1}");
            GUILayout.Label($"K = [{K[0, 0]:F2}, {K[0, 1]:F2}, {K[0, 2]:F2}, {K[0, 3]:F2}]");
            GUILayout.Label($"p = {state[0]:F2}   theta = {state[2] * Mathf.Rad2Deg:F1}°");

            cartMass = LabeledSlider($"cart M {cartMass:F2}", cartMass, 0.5f, 5f);
            poleMass = LabeledSlider($"pole m {poleMass:F2}", poleMass, 0.05f, 2f);
            poleLength = LabeledSlider($"pole l {poleLength:F2}", poleLength, 0.3f, 2f);
            qAngle = LabeledSlider($"Q angle {qAngle:F0}", qAngle, 0.1f, 100f);
            rControl = LabeledSlider($"R {rControl:F2}", rControl, 0.01f, 10f);

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Kick")) state[3] += 1.5f;
            if (GUILayout.Button("Big kick")) state[3] += 4f;
            if (GUILayout.Button("Reset"))
            {
                state[0] = 0f; state[1] = 0f; state[2] = 0.25f; state[3] = 0f;
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
    /// Re-linearize → warm LQR gain → RK4-integrate the nonlinear cart-pole with
    /// u = -K·x. LqrState carries the Riccati solution across frames (caller must
    /// RunByRef and copy the struct back).
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct CartPoleStepJob : IJob
    {
        public floatMxN K;
        public floatLQRState LqrState;
        public NativeArray<float> State;
        public NativeArray<float> Out;
        public float CartMass, PoleMass, PoleLength, QPos, QAngle, RCost, MaxForce, Dt;
        public int Steps;

        const float G = 9.81f;

        public void Execute()
        {
            float M = CartMass, m = PoleMass, l = PoleLength;

            // discrete linearization about upright (Euler, dt = Dt*Steps ~ one frame).
            // Fresh Temp matrices each call: construction zero-initializes, and there
            // is no public matrix zeroInPlace/fill to reuse persistent ones safely.
            float dt = Dt * Steps;
            var A = new floatMxN(4, 4, Allocator.Temp);
            var B = new floatMxN(4, 1, Allocator.Temp);
            var Q = new floatMxN(4, 4, Allocator.Temp);
            var R = new floatMxN(1, 1, Allocator.Temp);

            // continuous: v' = -(m g / M) theta + u/M ;  w' = ((M+m) g)/(M l) theta - u/(M l)
            A[0, 0] = 1f; A[0, 1] = dt;
            A[1, 1] = 1f; A[1, 2] = -(m * G / M) * dt;
            A[2, 2] = 1f; A[2, 3] = dt;
            A[3, 2] = ((M + m) * G / (M * l)) * dt; A[3, 3] = 1f;
            B[1, 0] = dt / M;
            B[3, 0] = -dt / (M * l);
            Q[0, 0] = QPos; Q[1, 1] = 1f; Q[2, 2] = QAngle; Q[3, 3] = 1f;
            R[0, 0] = RCost;

            LQRInfo info = LQR.lqr(in A, in B, in Q, in R, ref K, ref LqrState);
            Out[1] = info.iterations;
            Out[2] = info ? 1f : 0f;
            Out[3] = (float)info.residual;

            // u = -K·x, clamped; then RK4 on the full nonlinear dynamics
            float u = 0f;
            for (int s = 0; s < Steps; s++)
            {
                u = -(K[0, 0] * State[0] + K[0, 1] * State[1] + K[0, 2] * State[2] + K[0, 3] * State[3]);
                u = math.clamp(u, -MaxForce, MaxForce);

                float4 x = new float4(State[0], State[1], State[2], State[3]);
                float4 k1 = Deriv(x, u);
                float4 k2 = Deriv(x + 0.5f * Dt * k1, u);
                float4 k3 = Deriv(x + 0.5f * Dt * k2, u);
                float4 k4 = Deriv(x + Dt * k3, u);
                x += Dt / 6f * (k1 + 2f * k2 + 2f * k3 + k4);

                State[0] = x.x; State[1] = x.y; State[2] = x.z; State[3] = x.w;
            }
            Out[0] = u;
        }

        float4 Deriv(float4 x, float u)
        {
            float M = CartMass, m = PoleMass, l = PoleLength;
            float sin = math.sin(x.z), cos = math.cos(x.z);
            float denom = M + m * sin * sin;
            float vdot = (u + m * sin * (l * x.w * x.w - G * cos)) / denom;
            float wdot = (G * sin - vdot * cos) / l;
            return new float4(x.y, vdot, x.w, wdot);
        }
    }
}
