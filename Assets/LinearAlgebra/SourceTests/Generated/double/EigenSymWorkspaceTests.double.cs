using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Workspace-overload tests for Eigen.eigenvaluesSymmetric and its workspace doubleEigenSymWorkspace
// (Arena.doubleEigenSymWorkspace(n)). eigenvaluesSymmetric DESTROYS its input matrix, so every call
// runs on a private copy.
//
// The ws overload is the real body (caller-owned eVec/vVec/pVec); the allocating overload delegates
// with Temp scratch, so for the same input the eigenvalues are bit-identical. Tests:
//   (a) EQUIVALENCE — allocating vs ws on the SAME symmetric matrix.
//   (b) REUSE       — ONE workspace reused across two different (same-size) inputs; the 2nd result
//                     equals a fresh allocating call.
//   (c) MIS-SIZED   — a workspace sized for the wrong n throws ArgumentException (managed).
public class doubleEigenSymWorkspaceTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WorkspaceJob : IJob
    {
        public enum TestType
        {
            Equiv6,
            Equiv1,
            WorkspaceReuse,
        }

        public TestType Type;

        static double Tol() => 256 * Consts.doubleSqrtEps;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.Equiv6:         Equiv(6, 7711); break;
                case TestType.Equiv1:         Equiv(1, 2231); break;   // 1x1 early-return path
                case TestType.WorkspaceReuse: WorkspaceReuse(); break;
            }
        }

        // n x n random symmetric matrix.
        static doubleMxN Symmetric(ref Arena arena, int n, uint seed)
        {
            var A = arena.doubleRandomMatrix(n, n, (double)(-5f), (double)5f, seed);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    double avg = (A[i, j] + A[j, i]) * (double)0.5f;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }
            return A;
        }

        void Equiv(int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = Symmetric(ref arena, n, seed);

            var Aa = A.Copy();
            var eigA = arena.doubleVec(n);
            bool okA = Eigen.eigenvaluesSymmetric(ref Aa, ref eigA);

            var Aw = A.Copy();
            var eigW = arena.doubleVec(n);
            var ws = arena.doubleEigenSymWorkspace(n);
            bool okW = Eigen.eigenvaluesSymmetric(ref Aw, ref eigW, ref ws);

            Assert.IsTrue(okA == okW);
            Assert.IsTrue(Analysis.IsZero(eigA - eigW, Tol()));

            arena.Dispose();
        }

        void WorkspaceReuse()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 6;

            var A1 = Symmetric(ref arena, n, 8801);
            var A2 = Symmetric(ref arena, n, 9902);

            var ws = arena.doubleEigenSymWorkspace(n);   // allocated ONCE

            // warm on A1
            var A1c = A1.Copy();
            var eig1 = arena.doubleVec(n);
            Eigen.eigenvaluesSymmetric(ref A1c, ref eig1, ref ws);

            // reuse on A2
            var A2w = A2.Copy();
            var eigW = arena.doubleVec(n);
            bool okW = Eigen.eigenvaluesSymmetric(ref A2w, ref eigW, ref ws);

            // fresh allocating reference on A2
            var A2a = A2.Copy();
            var eigA = arena.doubleVec(n);
            bool okA = Eigen.eigenvaluesSymmetric(ref A2a, ref eigA);

            Assert.IsTrue(okW == okA);
            Assert.IsTrue(Analysis.IsZero(eigW - eigA, Tol()));

            arena.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(WorkspaceJob.TestType));

    [TestCaseSource("GetEnums")]
    public void WorkspaceTests(WorkspaceJob.TestType type)
    {
        new WorkspaceJob() { Type = type }.Run();
    }

    // ---- mis-sized workspace guard (managed [Test]) ----

    [Test]
    public void EigenSym_BadWorkspace_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            int n = 5;
            var A = arena.doubleIdentityMatrix(n);   // symmetric -> passes the symmetry guard
            var eig = arena.doubleVec(n);
            var ws = arena.doubleEigenSymWorkspace(n + 1);   // wrong n
            Assert.Throws<ArgumentException>(
                () => Eigen.eigenvaluesSymmetric(ref A, ref eig, ref ws));
        }
        finally { arena.Dispose(); }
    }

    // Arena.doubleEigenSymWorkspace(n): three length-n vectors.
    [Test]
    public void EigenSymWorkspace_Factory_SizesCorrectly()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var ws = arena.doubleEigenSymWorkspace(7);
            Assert.AreEqual(7, ws.eVec.N);
            Assert.AreEqual(7, ws.vVec.N);
            Assert.AreEqual(7, ws.pVec.N);
        }
        finally { arena.Dispose(); }
    }
}
