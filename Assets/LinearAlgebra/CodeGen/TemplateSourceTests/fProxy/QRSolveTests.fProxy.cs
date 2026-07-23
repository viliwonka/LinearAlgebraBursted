using System.Collections;
using System.Collections.Generic;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;


using Unity.Jobs;

using UnityEngine;
using UnityEngine.TestTools;

public class fProxyQRSolveTests {

    [BurstCompile(CompileSynchronously = true)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
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

            int dim = 8;

            var Q = GenerateOP.fProxyIdentityMat(dim);
            var R = new fProxyMxN(dim, dim, Allocator.Temp);

            var A = new fProxyMxN(in Q, Allocator.Temp);

            QR.decompInPlace(ref Q, ref R);

            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f);

            var y = Blas.dot(b, Q);

            Blas.triUpper(ref R, ref y);

            var Ax = Blas.dot(A, y);

            var resid = new fProxyN(in b, Allocator.Temp);
            fProxyComp.subInPlace(resid, Ax);
            Assert.IsTrue(Analysis.isZero(resid, 1E-6f));
        }

    }

    [Test]
    public void QRSolveIdentity()
    {
        new TestJob() { Type = TestJob.TestType.QRSolve }.Run();
    }


    
}
