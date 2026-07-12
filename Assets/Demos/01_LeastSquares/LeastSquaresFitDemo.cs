using System.Diagnostics;
using LinearAlgebra;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using Random = Unity.Mathematics.Random;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// L2 least-squares surface fit, re-solved every frame inside a Burst job.
    /// Drop on an empty GameObject and enter play mode: a noisy animated point
    /// cloud is generated and a plane (3 params) or quadric (6 params) is fitted
    /// to it with QR.solveInPlace. Gizmos draw the points and the fitted surface;
    /// an on-screen panel shows coefficients, residual and solve time.
    /// </summary>
    public class LeastSquaresFitDemo : MonoBehaviour
    {
        public enum FitModel { Plane = 0, Quadric = 1 }

        [Range(16, 4096)] public int pointCount = 512;
        [Range(0f, 0.5f)] public float noiseSigma = 0.05f;
        [Range(0f, 0.5f)] public float outlierFraction = 0.1f;
        [Range(0f, 10f)] public float outlierScale = 4f;
        public FitModel model = FitModel.Quadric;
        public bool animate = true;

        NativeArray<float3> points;
        NativeArray<float> coeffs;   // 6 slots (quadric); plane uses first 3
        NativeArray<float> stats;    // [0] rms residual, [1] solve success flag
        float solveMs;
        uint frame;
        readonly Stopwatch sw = new Stopwatch();

        void OnEnable()
        {
            points = new NativeArray<float3>(pointCount, Allocator.Persistent);
            coeffs = new NativeArray<float>(6, Allocator.Persistent);
            stats = new NativeArray<float>(2, Allocator.Persistent);
        }

        void OnDisable()
        {
            if (points.IsCreated) points.Dispose();
            if (coeffs.IsCreated) coeffs.Dispose();
            if (stats.IsCreated) stats.Dispose();
        }

        void Update()
        {
            if (points.Length != pointCount)
            {
                points.Dispose();
                points = new NativeArray<float3>(pointCount, Allocator.Persistent);
            }

            var job = new GenerateAndFitJob
            {
                Points = points,
                Coeffs = coeffs,
                Stats = stats,
                Model = (int)model,
                NoiseSigma = noiseSigma,
                OutlierFraction = outlierFraction,
                OutlierScale = outlierScale,
                Time = animate ? UnityEngine.Time.time : 0f,
                Seed = 1u + frame++,
            };

            sw.Restart();
            job.Run();
            sw.Stop();
            solveMs = (float)sw.Elapsed.TotalMilliseconds;
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying || !points.IsCreated) return;

            Gizmos.color = new Color(1f, 0.6f, 0.1f, 0.9f);
            for (int i = 0; i < points.Length; i++)
                Gizmos.DrawCube((Vector3)points[i], Vector3.one * 0.02f);

            // fitted surface as a wire grid over [-1,1]^2
            Gizmos.color = Color.cyan;
            const int G = 16;
            for (int gx = 0; gx <= G; gx++)
            {
                for (int gz = 0; gz < G; gz++)
                {
                    float x0 = -1f + 2f * gx / G, z0 = -1f + 2f * gz / G;
                    float z1 = -1f + 2f * (gz + 1) / G;
                    Gizmos.DrawLine(new Vector3(x0, Eval(x0, z0), z0), new Vector3(x0, Eval(x0, z1), z1));
                    Gizmos.DrawLine(new Vector3(z0, Eval(z0, x0), x0), new Vector3(z1, Eval(z1, x0), x0));
                }
            }
        }

        float Eval(float x, float z)
        {
            if (model == FitModel.Plane)
                return coeffs[0] * x + coeffs[1] * z + coeffs[2];
            return coeffs[0] * x * x + coeffs[1] * z * z + coeffs[2] * x * z
                 + coeffs[3] * x + coeffs[4] * z + coeffs[5];
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 340, 190), GUI.skin.box);
            GUILayout.Label($"LS L2 fit — {model}, m={pointCount}, n={(model == FitModel.Plane ? 3 : 6)}");
            GUILayout.Label($"solve: {solveMs:F3} ms   rms residual: {stats[0]:F4}   ok: {stats[1] == 1f}");
            GUILayout.Label(model == FitModel.Plane
                ? $"y = {coeffs[0]:F3}·x + {coeffs[1]:F3}·z + {coeffs[2]:F3}"
                : $"y = {coeffs[0]:F2}x² + {coeffs[1]:F2}z² + {coeffs[2]:F2}xz + {coeffs[3]:F2}x + {coeffs[4]:F2}z + {coeffs[5]:F2}");
            GUILayout.Label($"noise σ = {noiseSigma:F3}   outliers = {outlierFraction:P0} ×{outlierScale:F1}");
            noiseSigma = GUILayout.HorizontalSlider(noiseSigma, 0f, 0.5f);
            outlierFraction = GUILayout.HorizontalSlider(outlierFraction, 0f, 0.5f);
            GUILayout.EndArea();
        }
    }

    /// <summary>Generates the animated noisy point cloud and fits it, all in one Burst job.</summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct GenerateAndFitJob : IJob
    {
        public NativeArray<float3> Points;
        public NativeArray<float> Coeffs;
        public NativeArray<float> Stats;
        public int Model;                 // 0 plane, 1 quadric
        public float NoiseSigma, OutlierFraction, OutlierScale, Time;
        public uint Seed;

        public void Execute()
        {
            int m = Points.Length;
            int n = Model == 0 ? 3 : 6;

            var rng = new Random(Seed * 2654435761u + 1u);
            var gauss = new floatGaussian(0f, NoiseSigma);

            // animated ground truth
            float ca = 0.5f * math.sin(0.31f * Time);
            float cb = 0.5f * math.cos(0.23f * Time);
            float cc = 0.35f * math.sin(0.17f * Time);
            float cd = 0.4f * math.sin(0.41f * Time + 1f);
            float ce = 0.4f * math.cos(0.29f * Time + 2f);
            float cf = 0.2f * math.sin(0.13f * Time);

            for (int i = 0; i < m; i++)
            {
                float px = rng.NextFloat(-1f, 1f);
                float pz = rng.NextFloat(-1f, 1f);
                float py = Model == 0
                    ? ca * px + cb * pz + cf
                    : ca * px * px + cb * pz * pz + cc * px * pz + cd * px + ce * pz + cf;
                py += gauss.Next(ref rng);
                if (rng.NextFloat() < OutlierFraction)
                    py += rng.NextFloat(-1f, 1f) * OutlierScale;
                Points[i] = new float3(px, py, pz);
            }

            // design matrix + rhs; QR.solveInPlace destroys both, so residuals are
            // recomputed from Points afterwards instead of from A/b.
            var A = new floatMxN(m, n, Allocator.Temp);
            var b = new floatN(m, Allocator.Temp);
            for (int i = 0; i < m; i++)
            {
                float px = Points[i].x, pz = Points[i].z;
                if (Model == 0)
                {
                    A[i, 0] = px; A[i, 1] = pz; A[i, 2] = 1f;
                }
                else
                {
                    A[i, 0] = px * px; A[i, 1] = pz * pz; A[i, 2] = px * pz;
                    A[i, 3] = px; A[i, 4] = pz; A[i, 5] = 1f;
                }
                b[i] = Points[i].y;
            }

            var x = new floatN(n, Allocator.Temp);
            DirectSolveInfo info = QR.solveInPlace(ref A, ref b, ref x);

            for (int j = 0; j < 6; j++) Coeffs[j] = j < n ? x[j] : 0f;

            float ss = 0f;
            for (int i = 0; i < m; i++)
            {
                float px = Points[i].x, pz = Points[i].z;
                float pred = Model == 0
                    ? x[0] * px + x[1] * pz + x[2]
                    : x[0] * px * px + x[1] * pz * pz + x[2] * px * pz + x[3] * px + x[4] * pz + x[5];
                float e = pred - Points[i].y;
                ss += e * e;
            }
            Stats[0] = math.sqrt(ss / m);
            Stats[1] = info ? 1f : 0f;
        }
    }
}
