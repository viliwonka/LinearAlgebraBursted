using System.Diagnostics;
using LinearAlgebra;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// DOUBLE inverted pendulum on a cart under discrete LQR (6-state). Every
    /// frame one Burst job: builds the upright linearization by solving
    /// M0·W = [G | F] with the multi-RHS Cholesky (CHO.solveInPlace), warm
    /// re-solves the Riccati equation, then RK4-integrates the full nonlinear
    /// dynamics — each derivative evaluation solves the 3×3 mass matrix with
    /// CHO. Much twitchier than the single pole: small kicks only.
    /// </summary>
    public class DoubleCartPoleLQRDemo : MonoBehaviour
    {
        [Range(0.5f, 5f)] public float cartMass = 1.5f;
        [Range(0.05f, 1f)] public float mass1 = 0.3f;
        [Range(0.05f, 1f)] public float mass2 = 0.3f;
        [Range(0.3f, 1.5f)] public float length1 = 0.6f;
        [Range(0.3f, 1.5f)] public float length2 = 0.6f;
        [Range(0.1f, 100f)] public float qPosition = 8f;
        [Range(0.1f, 200f)] public float qAngle = 80f;
        [Range(0.01f, 10f)] public float rControl = 1f;
        [Range(10f, 200f)] public float maxForce = 80f;

        const float SimDt = 1f / 480f;   // stiff system, small RK4 substep
        const int Substeps = 8;          // 8 × 1/480 ≈ one 60 fps frame

        floatMxN K;                    // 1×6
        floatLQRState lqrState;
        NativeArray<float> state;      // [x, th1, th2, vx, w1, w2]
        NativeArray<float> outStats;   // [0] u, [1] iters, [2] converged, [3] choOk
        float frameMs;

        void OnEnable()
        {
            K = new floatMxN(1, 6, Allocator.Persistent);
            lqrState = new floatLQRState(6, Allocator.Persistent);
            state = new NativeArray<float>(6, Allocator.Persistent);
            outStats = new NativeArray<float>(4, Allocator.Persistent);
            state[1] = 0.06f; state[2] = -0.08f;
        }

        void OnDisable()
        {
            K.Dispose(); lqrState.Dispose();
            state.Dispose(); outStats.Dispose();
        }

        void Update()
        {
            var job = new DoubleCartPoleStepJob
            {
                K = K, LqrState = lqrState, State = state, Out = outStats,
                Mc = cartMass, M1 = mass1, M2 = mass2, L1 = length1, L2 = length2,
                QPos = qPosition, QAngle = qAngle, RCost = rControl,
                MaxForce = maxForce, Dt = SimDt, Steps = Substeps,
            };

            var sw = Stopwatch.StartNew();
            IJobExtensions.RunByRef(ref job);
            sw.Stop();
            frameMs = (float)sw.Elapsed.TotalMilliseconds;

            lqrState = job.LqrState;
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying || !state.IsCreated) return;

            float x = state[0], t1 = state[1], t2 = state[2];
            var cart = new Vector3(x, 0f, 0f);
            var joint = cart + new Vector3(math.sin(t1), math.cos(t1), 0f) * length1;
            var tip = joint + new Vector3(math.sin(t2), math.cos(t2), 0f) * length2;

            Gizmos.color = Color.gray;
            Gizmos.DrawLine(new Vector3(-6f, -0.15f, 0f), new Vector3(6f, -0.15f, 0f));
            Gizmos.color = Color.white;
            Gizmos.DrawCube(cart, new Vector3(0.5f, 0.25f, 0.3f));
            Gizmos.color = Color.yellow;
            Gizmos.DrawLine(cart, joint);
            Gizmos.DrawSphere(joint, 0.07f);
            Gizmos.color = new Color(1f, 0.5f, 0f);
            Gizmos.DrawLine(joint, tip);
            Gizmos.DrawSphere(tip, 0.07f);
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 320, 400, 240), GUI.skin.box);
            GUILayout.Label($"DOUBLE cart-pole LQR — {frameMs:F3} ms/frame (8 RK4+CHO substeps)");
            GUILayout.Label($"u = {outStats[0]:F1} N   Riccati iters: {outStats[1]:F0}   converged: {outStats[2] == 1f}   mass-matrix SPD: {outStats[3] == 1f}");
            GUILayout.Label($"x = {state[0]:F2}   th1 = {state[1] * Mathf.Rad2Deg:F1}°   th2 = {state[2] * Mathf.Rad2Deg:F1}°");

            cartMass = LabeledSlider($"cart M {cartMass:F2}", cartMass, 0.5f, 5f);
            mass2 = LabeledSlider($"m2 {mass2:F2}", mass2, 0.05f, 1f);
            length2 = LabeledSlider($"l2 {length2:F2}", length2, 0.3f, 1.5f);
            qAngle = LabeledSlider($"Q angle {qAngle:F0}", qAngle, 0.1f, 200f);

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Nudge")) state[4] += 0.4f;
            if (GUILayout.Button("Reset"))
            {
                for (int i = 0; i < 6; i++) state[i] = 0f;
                state[1] = 0.06f; state[2] = -0.08f;
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
    /// Linearize (multi-RHS CHO solve of M0·W = [G|F]) → warm LQR → RK4 with a
    /// 3×3 CHO solve of M(q)·qdd = Q(q,qd) per derivative evaluation.
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct DoubleCartPoleStepJob : IJob
    {
        public floatMxN K;
        public floatLQRState LqrState;
        public NativeArray<float> State;
        public NativeArray<float> Out;
        public float Mc, M1, M2, L1, L2, QPos, QAngle, RCost, MaxForce, Dt;
        public int Steps;

        const float G = 9.81f;

        public void Execute()
        {
            float dt = Dt * Steps;

            // upright mass matrix M0 and gravity/input columns [G_g | F]
            var M0 = new floatMxN(3, 3, Allocator.Temp);
            var W = new floatMxN(3, 4, Allocator.Temp);   // -> M0^-1 [G_g | F]
            FillMass(ref M0, 1f, 1f, 1f);                 // c1=c2=c12=1 at upright
            W[1, 1] = (M1 + M2) * G * L1;
            W[2, 2] = M2 * G * L2;
            W[0, 3] = 1f;

            DirectSolveInfo chol = CHO.solveInPlace(ref M0, ref W);
            Out[3] = chol ? 1f : 0f;

            var A = new floatMxN(6, 6, Allocator.Temp);
            var B = new floatMxN(6, 1, Allocator.Temp);
            for (int i = 0; i < 6; i++) A[i, i] = 1f;
            A[0, 3] = dt; A[1, 4] = dt; A[2, 5] = dt;
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++)
                    A[3 + r, c] += W[r, c] * dt;
                B[3 + r, 0] = W[r, 3] * dt;
            }

            var Q = new floatMxN(6, 6, Allocator.Temp);
            var R = new floatMxN(1, 1, Allocator.Temp);
            Q[0, 0] = QPos; Q[1, 1] = QAngle; Q[2, 2] = QAngle;
            Q[3, 3] = 1f; Q[4, 4] = 4f; Q[5, 5] = 4f;
            R[0, 0] = RCost;

            LQRInfo info = Control.lqr(in A, in B, in Q, in R, ref K, ref LqrState);
            Out[1] = info.iterations;
            Out[2] = info ? 1f : 0f;

            float u = 0f;
            for (int s = 0; s < Steps; s++)
            {
                u = 0f;
                for (int i = 0; i < 6; i++) u -= K[0, i] * State[i];
                u = math.clamp(u, -MaxForce, MaxForce);

                var z = new NativeArray<float>(6, Allocator.Temp);
                var k1 = new NativeArray<float>(6, Allocator.Temp);
                var k2 = new NativeArray<float>(6, Allocator.Temp);
                var k3 = new NativeArray<float>(6, Allocator.Temp);
                var k4 = new NativeArray<float>(6, Allocator.Temp);
                for (int i = 0; i < 6; i++) z[i] = State[i];

                Deriv(z, u, k1);
                Deriv(Blend(z, k1, 0.5f * Dt), u, k2);
                Deriv(Blend(z, k2, 0.5f * Dt), u, k3);
                Deriv(Blend(z, k3, Dt), u, k4);

                for (int i = 0; i < 6; i++)
                    State[i] += Dt / 6f * (k1[i] + 2f * k2[i] + 2f * k3[i] + k4[i]);
            }
            Out[0] = u;
        }

        static NativeArray<float> Blend(NativeArray<float> z, NativeArray<float> k, float h)
        {
            var o = new NativeArray<float>(6, Allocator.Temp);
            for (int i = 0; i < 6; i++) o[i] = z[i] + h * k[i];
            return o;
        }

        void FillMass(ref floatMxN M, float c1, float c2, float c12)
        {
            M[0, 0] = Mc + M1 + M2;
            M[0, 1] = M[1, 0] = (M1 + M2) * L1 * c1;
            M[0, 2] = M[2, 0] = M2 * L2 * c2;
            M[1, 1] = (M1 + M2) * L1 * L1;
            M[1, 2] = M[2, 1] = M2 * L1 * L2 * c12;
            M[2, 2] = M2 * L2 * L2;
        }

        void Deriv(NativeArray<float> z, float u, NativeArray<float> dz)
        {
            float t1 = z[1], t2 = z[2], w1 = z[4], w2 = z[5];
            float s1 = math.sin(t1);
            float s2 = math.sin(t2);
            float c1 = math.cos(t1), c2 = math.cos(t2);
            float s12 = math.sin(t1 - t2), c12 = math.cos(t1 - t2);

            var M = new floatMxN(3, 3, Allocator.Temp);
            FillMass(ref M, c1, c2, c12);

            var rhs = new floatN(3, Allocator.Temp);
            rhs[0] = u + (M1 + M2) * L1 * w1 * w1 * s1 + M2 * L2 * w2 * w2 * s2;
            rhs[1] = (M1 + M2) * G * L1 * s1 - M2 * L1 * L2 * w2 * w2 * s12;
            rhs[2] = M2 * G * L2 * s2 + M2 * L1 * L2 * w1 * w1 * s12;

            CHO.solveInPlace(ref M, ref rhs);   // rhs -> qdd

            dz[0] = z[3]; dz[1] = w1; dz[2] = w2;
            dz[3] = rhs[0]; dz[4] = rhs[1]; dz[5] = rhs[2];
        }
    }
}
