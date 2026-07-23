using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Workspace-overload tests for Eigen.valuesSymmetricInPlace and its workspace fProxyEigenSymCache
// (fProxyEigenSymCache(n, Allocator)). eigenvaluesSymmetric DESTROYS its input matrix, so every call
// runs on a private copy.
//
// The ws overload is the real body (caller-owned eVec/vVec/pVec); the allocating overload delegates
// with Temp scratch, so for the same input the eigenvalues are bit-identical. Tests:
//   (a) EQUIVALENCE — allocating vs ws on the SAME symmetric matrix.
//   (b) REUSE       — ONE workspace reused across two different (same-size) inputs; the 2nd result
//                     equals a fresh allocating call.
//   (c) MIS-SIZED   — a workspace sized for the wrong n throws ArgumentException (managed).
public class fProxyEigenSymWorkspaceTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WorkspaceJob : IJob
    {
        public enum TestType
        {
            Equiv6,
            Equiv1,
            WorkspaceReuse,
        }

        public TestType Type;

        static fProxy Tol() => 256 * Consts.fProxySqrtEps;

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
        static fProxyMxN Symmetric(int n, uint seed)
        {
            var A = GenerateOP.fProxyRandomMat(n, n, (fProxy)(-5f), (fProxy)5f, seed);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    fProxy avg = (A[i, j] + A[j, i]) * (fProxy)0.5f;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }
            return A;
        }

        void Equiv(int n, uint seed)
        {
            var A = Symmetric(n, seed);

            var Aa = new fProxyMxN(in A, Allocator.Temp);
            var eigA = new fProxyN(n, Allocator.Temp);
            bool okA = Eigen.valuesSymmetricInPlace(ref Aa, ref eigA);

            var Aw = new fProxyMxN(in A, Allocator.Temp);
            var eigW = new fProxyN(n, Allocator.Temp);
            var ws = new fProxyEigenSymCache(n, Allocator.Temp);
            bool okW = Eigen.valuesSymmetricInPlace(ref Aw, ref eigW, ref ws);

            Assert.IsTrue(okA == okW);
            var diff = new fProxyN(in eigA, Allocator.Temp);
            fProxyComp.subInPlace(diff, eigW);
            Assert.IsTrue(Analysis.isZero(diff, Tol()));
        }

        void WorkspaceReuse()
        {
            int n = 6;

            var A1 = Symmetric(n, 8801);
            var A2 = Symmetric(n, 9902);

            var ws = new fProxyEigenSymCache(n, Allocator.Temp);   // allocated ONCE

            // warm on A1
            var A1c = new fProxyMxN(in A1, Allocator.Temp);
            var eig1 = new fProxyN(n, Allocator.Temp);
            Eigen.valuesSymmetricInPlace(ref A1c, ref eig1, ref ws);

            // reuse on A2
            var A2w = new fProxyMxN(in A2, Allocator.Temp);
            var eigW = new fProxyN(n, Allocator.Temp);
            bool okW = Eigen.valuesSymmetricInPlace(ref A2w, ref eigW, ref ws);

            // fresh allocating reference on A2
            var A2a = new fProxyMxN(in A2, Allocator.Temp);
            var eigA = new fProxyN(n, Allocator.Temp);
            bool okA = Eigen.valuesSymmetricInPlace(ref A2a, ref eigA);

            Assert.IsTrue(okW == okA);
            var diff = new fProxyN(in eigW, Allocator.Temp);
            fProxyComp.subInPlace(diff, eigA);
            Assert.IsTrue(Analysis.isZero(diff, Tol()));
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
        int n = 5;
        var A = GenerateOP.fProxyIdentityMat(n);   // symmetric -> passes the symmetry guard
        var eig = new fProxyN(n, Allocator.Temp);
        var ws = new fProxyEigenSymCache(n + 1, Allocator.Temp);   // wrong n
        Assert.Throws<ArgumentException>(
            () => Eigen.valuesSymmetricInPlace(ref A, ref eig, ref ws));
    }

    // fProxyEigenSymCache(n, Allocator): three length-n vectors.
    [Test]
    public void EigenSymWorkspace_Factory_SizesCorrectly()
    {
        var ws = new fProxyEigenSymCache(7, Allocator.Temp);
        Assert.AreEqual(7, ws.eVec.N);
        Assert.AreEqual(7, ws.vVec.N);
        Assert.AreEqual(7, ws.pVec.N);
    }
}
