using System.Collections;
using System.Collections.Generic;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;


using Unity.Jobs;

using UnityEngine;
using UnityEngine.TestTools;

public class fProxySolversTests {

    [BurstCompile]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            USolveIdentity,
            LSolveIdentity,
            QRSolve,
        }

        public TestType Type;

        public void Execute()
        {
            switch(Type)
            {
                case TestType.QRSolve:
                    QRSolve();
                break;  
            }
        }

        public void QRSolve()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var Q = arena.fProxyIdentityMat(dim);
            var R = arena.fProxyMat(dim);

            var A = Q.Copy();

            QR.qrDecomposition(ref Q, ref R);

            var b = arena.fProxyRandomVec(dim, -1f, 1f);

            var y = Blas.dot(b, Q);

            Solvers.solveUpperTriangular(ref R, ref y);

            var Ax = Blas.dot(A, y);

            Assert.IsTrue(Analysis.isZero(b - Ax, 1E-6f));

            arena.Dispose();
        }

    }

    [Test]
    public void QRSolveIdentity()
    {
        new TestJob() { Type = TestJob.TestType.QRSolve }.Run();
    }


    
}
