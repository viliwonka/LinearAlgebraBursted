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

            var Q = arena.fProxyIdentityMatrix(dim);
            var R = arena.fProxyMat(dim);

            var A = Q.Copy();

            OrthoOP.qrDecomposition(ref Q, ref R);

            var b = arena.fProxyRandomVector(dim, -1f, 1f);

            var y = fProxyOP.dot(b, Q);
            
            //var x = arena.Vector(dim);

            Solvers.SolveUpperTriangular(ref R, ref y);

            var Ax = fProxyOP.dot(A, y);

            Assert.IsTrue(Analysis.IsZero(b - Ax, 1E-6f));
            /*Print.Log(A);
            Print.Log(Q);
            Print.Log(R);
            */
            //AssertQR(in A, in Q, in R);

            arena.Dispose();
        }

    }

    [Test]
    public void QRSolveIdentity()
    {
        new TestJob() { Type = TestJob.TestType.QRSolve }.Run();
    }


    
}
