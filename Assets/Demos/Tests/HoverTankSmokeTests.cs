using LinearAlgebra;
using LinearAlgebra.Control;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebraDemos.Tests
{
    /// <summary>
    /// Headless, math-only smoke tests for HoverTankDemo's two LQR loops: build the
    /// demo's exact discrete A/B/Q/R via its own static model builders
    /// (<see cref="HoverTankStepJob.BuildHoverModel"/>/<see cref="HoverTankStepJob.BuildServoModel"/>),
    /// solve for K, then simulate the LINEAR closed loop x_{k+1} = (A - B K) x_k from a
    /// perturbed start. No Physics/raycasts/Rigidbody involved. Decay to near-zero
    /// implies the closed-loop spectral radius is below 1.
    /// </summary>
    public class HoverTankSmokeTests
    {
        [Test]
        public void HoverLoop_Stabilizes_From_Perturbed_State()
        {
            const int n = 6, m = 3;

            HoverTankStepJob.BuildHoverModel(
                1f / 60f,
                40f, 6f, 90f, 8f, 0.02f, 0.4f,
                Allocator.TempJob, out var A, out var B, out var Q, out var R);

            var K = new floatMxN(m, n, Allocator.TempJob);
            RiccatiInfo info = LQR.lqr(in A, in B, in Q, in R, ref K);
            Assert.IsTrue(info, $"hover LQR did not converge: {info.status}");

            var BK = new floatMxN(n, n, Allocator.TempJob);
            Blas.dot(in B, in K, ref BK);
            var Acl = new floatMxN(n, n, Allocator.TempJob);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    Acl[i, j] = A[i, j] - BK[i, j];

            var x = new NativeArray<float>(n, Allocator.TempJob);
            var xNext = new NativeArray<float>(n, Allocator.TempJob);
            x[0] = 0.8f; x[2] = 0.3f; x[4] = -0.25f;   // perturbed height / roll / pitch

            float norm = StateNorm(x, n);
            int steps = 0;
            const int maxSteps = 2000;
            while (norm >= 1e-3f && steps < maxSteps)
            {
                for (int i = 0; i < n; i++)
                {
                    float s = 0f;
                    for (int j = 0; j < n; j++) s += Acl[i, j] * x[j];
                    xNext[i] = s;
                }
                for (int i = 0; i < n; i++) x[i] = xNext[i];
                norm = StateNorm(x, n);
                steps++;
            }

            Assert.IsTrue(norm < 1e-3f, $"hover closed loop did not decay below 1e-3 within {maxSteps} steps (||x|| = {norm})");

            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose();
            K.Dispose(); BK.Dispose(); Acl.Dispose();
            x.Dispose(); xNext.Dispose();
        }

        [Test]
        public void TurretServo_Stabilizes_From_Perturbed_State()
        {
            const int n = 2, m = 1;

            HoverTankStepJob.BuildServoModel(
                1f / 60f, 60f, 8f, 0.3f,
                Allocator.TempJob, out var A, out var B, out var Q, out var R);

            var K = new floatMxN(m, n, Allocator.TempJob);
            RiccatiInfo info = LQR.lqr(in A, in B, in Q, in R, ref K);
            Assert.IsTrue(info, $"turret servo LQR did not converge: {info.status}");

            var BK = new floatMxN(n, n, Allocator.TempJob);
            Blas.dot(in B, in K, ref BK);
            var Acl = new floatMxN(n, n, Allocator.TempJob);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    Acl[i, j] = A[i, j] - BK[i, j];

            var x = new NativeArray<float>(n, Allocator.TempJob);
            var xNext = new NativeArray<float>(n, Allocator.TempJob);
            x[0] = 1.2f;   // 1.2 rad angle error, zero rate

            float norm = StateNorm(x, n);
            int steps = 0;
            const int maxSteps = 2000;
            while (norm >= 1e-3f && steps < maxSteps)
            {
                for (int i = 0; i < n; i++)
                {
                    float s = 0f;
                    for (int j = 0; j < n; j++) s += Acl[i, j] * x[j];
                    xNext[i] = s;
                }
                for (int i = 0; i < n; i++) x[i] = xNext[i];
                norm = StateNorm(x, n);
                steps++;
            }

            Assert.IsTrue(norm < 1e-3f, $"turret servo closed loop did not decay below 1e-3 within {maxSteps} steps (||x|| = {norm})");

            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose();
            K.Dispose(); BK.Dispose(); Acl.Dispose();
            x.Dispose(); xNext.Dispose();
        }

        static float StateNorm(NativeArray<float> x, int n)
        {
            float s = 0f;
            for (int i = 0; i < n; i++) s += x[i] * x[i];
            return math.sqrt(s);
        }
    }
}
