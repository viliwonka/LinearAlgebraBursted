using System;

using BULA;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using Random = Unity.Mathematics.Random;

// Convergence-budget battery for the SVD/Eigen default iteration budget (Consts.sweepBudget).
// Builds matrices with KNOWN spectra (Haar-orthogonal factors x prescribed singular/eigen values)
// that stress the iterative QL/QR phases -- graded decay, tight clusters, random spread -- and
// asserts each solver (a) converges and (b) uses at most 1/4 of the default budget, so the
// max(75, 6n) default keeps a real safety margin at every tested size.
//
// The O(n^3) matrix construction and the solves run inside a [BurstCompile] IJob (n goes up to
// 1024 here, far too slow under managed execution); the job reports
// {Solved, sweeps, budget, converged} through an int buffer and the managed side does the
// asserts so failure messages stay readable.
public class fProxyConvergenceBudgetTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            Graded95Thin,
            Graded95Values,
            Graded95ValuesSymmetric,
            Graded95Symmetric,
            Graded99Thin,
            Graded99Symmetric,
            ClusteredThin,
            ClusteredSymmetric,
            RandomThin,
            RandomSymmetric
        }

        public TestType Type;
        public int N;

        // [0] Solved (1/0), [1] sweeps, [2] budget, [3] converged count
        public NativeArray<int> Out;

        public void Execute()
        {
            switch(Type)
            {
                case TestType.Graded95Thin:
                    Graded95Thin();
                break;
                case TestType.Graded95Values:
                    Graded95Values();
                break;
                case TestType.Graded95ValuesSymmetric:
                    Graded95ValuesSymmetric();
                break;
                case TestType.Graded95Symmetric:
                    Graded95Symmetric();
                break;
                case TestType.Graded99Thin:
                    Graded99Thin();
                break;
                case TestType.Graded99Symmetric:
                    Graded99Symmetric();
                break;
                case TestType.ClusteredThin:
                    ClusteredThin();
                break;
                case TestType.ClusteredSymmetric:
                    ClusteredSymmetric();
                break;
                case TestType.RandomThin:
                    RandomThin();
                break;
                case TestType.RandomSymmetric:
                    RandomSymmetric();
                break;
            }
        }

        void Store(in SVDInfo info, int budget)
        {
            Out[0] = info.Solved ? 1 : 0;
            Out[1] = info.sweeps;
            Out[2] = budget;
            Out[3] = info.converged;
        }

        void Store(in EigenInfo info, int budget)
        {
            Out[0] = info.Solved ? 1 : 0;
            Out[1] = info.sweeps;
            Out[2] = budget;
            Out[3] = info.converged;
        }

        static fProxyN GradedSpectrum(int n, fProxy ratio)
        {
            var s = new fProxyN(n, Allocator.Temp);
            fProxy v = (fProxy)100;
            for (int i = 0; i < n; i++) { s[i] = v; v *= ratio; }
            return s;
        }

        static fProxyN ClusteredSpectrum(int n)
        {
            var s = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                int cluster = i % 5;
                fProxy center = (fProxy)(100.0 / math.pow(10.0, cluster));
                fProxy jitter = (fProxy)(1.0 + 1e-4 * ((i / 5) % 7));
                s[i] = center * jitter;
            }
            for (int j = 0; j < n; j++)
            {
                int best = j; fProxy bv = s[j];
                for (int k = j + 1; k < n; k++) if (s[k] > bv) { best = k; bv = s[k]; }
                if (best != j) { fProxy t = s[j]; s[j] = s[best]; s[best] = t; }
            }
            return s;
        }

        static fProxyN RandomSpectrum(int n, uint seed)
        {
            var rng = new Random(seed);
            var s = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) s[i] = (fProxy)(0.01 + rng.NextDouble() * 99.99);
            for (int j = 0; j < n; j++)
            {
                int best = j; fProxy bv = s[j];
                for (int k = j + 1; k < n; k++) if (s[k] > bv) { best = k; bv = s[k]; }
                if (best != j) { fProxy t = s[j]; s[j] = s[best]; s[best] = t; }
            }
            return s;
        }

        // A = G * diag(sigma) * V^T, G/V independent Haar-orthogonal -> known singular values.
        static fProxyMxN BuildGeneral(int n, in fProxyN sigma, uint seed)
        {
            var rng = new Random(seed);
            var G = new fProxyMxN(n, n, Allocator.Temp);
            var Rm = new fProxyMxN(n, n, Allocator.Temp);
            var gauss = new fProxyGaussian((fProxy)0, (fProxy)1);
            Rand.randomInPlace(ref rng, ref G, ref gauss);
            QR.decompInPlace(ref G, ref Rm);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            Rand.orthogonalInPlace(ref rng, ref V);

            var A = new fProxyMxN(n, n, Allocator.Temp);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                        acc += (double)G[i, t] * (double)sigma[t] * (double)V[j, t];
                    A[i, j] = (fProxy)acc;
                }
            return A;
        }

        // A = Q * diag(sigma) * Q^T, Q Haar-orthogonal -> known SYMMETRIC eigenvalues.
        static fProxyMxN BuildSymmetric(int n, in fProxyN sigma, uint seed)
        {
            var rng = new Random(seed);
            var Q = new fProxyMxN(n, n, Allocator.Temp);
            Rand.orthogonalInPlace(ref rng, ref Q);

            var A = new fProxyMxN(n, n, Allocator.Temp);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                        acc += (double)Q[i, t] * (double)sigma[t] * (double)Q[j, t];
                    A[i, j] = (fProxy)acc;
                }
            return A;
        }

        public void Graded95Thin()
        {
            int n = N;
            var sigma = GradedSpectrum(n, (fProxy)0.95f);
            var A = BuildGeneral(n, in sigma, 0xC0FFEEu + (uint)n);
            var U = new fProxyMxN(n, n, Allocator.Temp);
            var S = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            int budget = Consts.sweepBudget(n);
            var info = SVD.thin(in A, ref U, ref S, ref V, budget);
            Store(in info, budget);
        }

        public void Graded95Values()
        {
            int n = N;
            var sigma = GradedSpectrum(n, (fProxy)0.95f);
            var A = BuildGeneral(n, in sigma, 0xBADC0DEu + (uint)n);
            var S = new fProxyN(n, Allocator.Temp);
            int budget = Consts.sweepBudget(n);
            var info = SVD.values(in A, ref S, budget, Consts.fProxyZeroThreshold);
            Store(in info, budget);
        }

        public void Graded95ValuesSymmetric()
        {
            int n = N;
            var sigma = GradedSpectrum(n, (fProxy)0.95f);
            var A = BuildSymmetric(n, in sigma, 0x5EED0001u + (uint)n);
            var eig = new fProxyN(n, Allocator.Temp);
            int budget = Consts.sweepBudget(n);
            var info = Eigen.valuesSymmetricInPlace(ref A, ref eig, budget, Consts.fProxyZeroThreshold);
            Store(in info, budget);
        }

        public void Graded95Symmetric()
        {
            int n = N;
            var sigma = GradedSpectrum(n, (fProxy)0.95f);
            var A = BuildSymmetric(n, in sigma, 0x5EED0002u + (uint)n);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            int budget = Consts.sweepBudget(n);
            var info = Eigen.symmetricInPlace(ref A, ref eig, ref V, budget, Consts.fProxyZeroThreshold);
            Store(in info, budget);
        }

        public void Graded99Thin()
        {
            int n = N;
            var sigma = GradedSpectrum(n, (fProxy)0.99f);
            var A = BuildGeneral(n, in sigma, 0x99000001u);
            var U = new fProxyMxN(n, n, Allocator.Temp);
            var S = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            int budget = Consts.sweepBudget(n);
            var info = SVD.thin(in A, ref U, ref S, ref V, budget);
            Store(in info, budget);
        }

        public void Graded99Symmetric()
        {
            int n = N;
            var sigma = GradedSpectrum(n, (fProxy)0.99f);
            var A = BuildSymmetric(n, in sigma, 0x99000002u);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            int budget = Consts.sweepBudget(n);
            var info = Eigen.symmetricInPlace(ref A, ref eig, ref V, budget, Consts.fProxyZeroThreshold);
            Store(in info, budget);
        }

        public void ClusteredThin()
        {
            int n = N;
            var sigma = ClusteredSpectrum(n);
            var A = BuildGeneral(n, in sigma, 0xC1000001u);
            var U = new fProxyMxN(n, n, Allocator.Temp);
            var S = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            int budget = Consts.sweepBudget(n);
            var info = SVD.thin(in A, ref U, ref S, ref V, budget);
            Store(in info, budget);
        }

        public void ClusteredSymmetric()
        {
            int n = N;
            var sigma = ClusteredSpectrum(n);
            var A = BuildSymmetric(n, in sigma, 0xC1000002u);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            int budget = Consts.sweepBudget(n);
            var info = Eigen.symmetricInPlace(ref A, ref eig, ref V, budget, Consts.fProxyZeroThreshold);
            Store(in info, budget);
        }

        public void RandomThin()
        {
            int n = N;
            var sigma = RandomSpectrum(n, 0xF00D0001u);
            var A = BuildGeneral(n, in sigma, 0xF00D0011u);
            var U = new fProxyMxN(n, n, Allocator.Temp);
            var S = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            int budget = Consts.sweepBudget(n);
            var info = SVD.thin(in A, ref U, ref S, ref V, budget);
            Store(in info, budget);
        }

        public void RandomSymmetric()
        {
            int n = N;
            var sigma = RandomSpectrum(n, 0xF00D0002u);
            var A = BuildSymmetric(n, in sigma, 0xF00D0022u);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            int budget = Consts.sweepBudget(n);
            var info = Eigen.symmetricInPlace(ref A, ref eig, ref V, budget, Consts.fProxyZeroThreshold);
            Store(in info, budget);
        }
    }

    // Runs one battery case in the Burst job and asserts on the managed side, keeping the
    // human-readable [ConvBattery] log line and failure messages.
    static void RunCase(TestJob.TestType type, int n, string label)
    {
        var res = new NativeArray<int>(4, Allocator.TempJob);
        new TestJob { Type = type, N = n, Out = res }.Run();
        int solved = res[0], sweeps = res[1], budget = res[2], converged = res[3];
        res.Dispose();

        double frac = sweeps / (double)budget;
        string line = $"[ConvBattery] {label} n={n} status={(solved != 0 ? "Converged" : "MaxIterations")} sweeps={sweeps} converged={converged} budget={budget} frac={frac:F4}";
        TestContext.WriteLine(line);
        Assert.IsTrue(solved != 0, line + " -- DID NOT CONVERGE");
        Assert.LessOrEqual(sweeps, budget / 4, line + " -- EXCEEDED 1/4 BUDGET MARGIN");
    }

    [TestCase(128u)]
    [TestCase(512u)]
    [TestCase(1024u)]
    public void Graded95_Thin(uint n32)
    {
        RunCase(TestJob.TestType.Graded95Thin, (int)n32, "Graded95/thin");
    }

    [TestCase(128u)]
    [TestCase(512u)]
    [TestCase(1024u)]
    public void Graded95_Values(uint n32)
    {
        RunCase(TestJob.TestType.Graded95Values, (int)n32, "Graded95/values");
    }

    [TestCase(128u)]
    [TestCase(512u)]
    [TestCase(1024u)]
    public void Graded95_ValuesSymmetric(uint n32)
    {
        RunCase(TestJob.TestType.Graded95ValuesSymmetric, (int)n32, "Graded95/valuesSymmetricInPlace");
    }

    [TestCase(128u)]
    [TestCase(512u)]
    [TestCase(1024u)]
    public void Graded95_Symmetric(uint n32)
    {
        RunCase(TestJob.TestType.Graded95Symmetric, (int)n32, "Graded95/symmetric");
    }

    [Test]
    public void Graded99_512_Thin()
    {
        RunCase(TestJob.TestType.Graded99Thin, 512, "Graded99/thin");
    }

    [Test]
    public void Graded99_512_Symmetric()
    {
        RunCase(TestJob.TestType.Graded99Symmetric, 512, "Graded99/symmetric");
    }

    [Test]
    public void Clustered_512_Thin()
    {
        RunCase(TestJob.TestType.ClusteredThin, 512, "Clustered/thin");
    }

    [Test]
    public void Clustered_512_Symmetric()
    {
        RunCase(TestJob.TestType.ClusteredSymmetric, 512, "Clustered/symmetric");
    }

    [Test]
    public void Random_512_Thin()
    {
        RunCase(TestJob.TestType.RandomThin, 512, "Random/thin");
    }

    [Test]
    public void Random_512_Symmetric()
    {
        RunCase(TestJob.TestType.RandomSymmetric, 512, "Random/symmetric");
    }
}
