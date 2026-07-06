using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Tests for the integer Analysis structural-predicate surface (int / short / long):
// isZero (vector + matrix), isIdentity, isSymmetric, isDiagonal, isUpperTriangular,
// isLowerTriangular. The integer surface is EXACT-EQUALITY only -- there is NO epsilon/tolerance
// overload (integers have no roundoff), so unlike fProxyAnalysisTests there are no *Epsilon
// variants here. All square-only predicates must return false (NOT throw) for a non-square matrix.
public class iProxyAnalysisTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct AnalysisTestJob : IJob
    {
        public enum TestType
        {
            ZeroVector,
            ZeroMatrix,
            Identity,
            IdentityNegatives,
            Symmetric,
            Diagonal,
            UpperTriangular,
            LowerTriangular,
            NonSquare,
        }

        public TestType Type;

        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            try
            {
                switch (Type)
                {
                    case TestType.ZeroVector: ZeroVector(ref arena); break;
                    case TestType.ZeroMatrix: ZeroMatrix(ref arena); break;
                    case TestType.Identity: Identity(ref arena); break;
                    case TestType.IdentityNegatives: IdentityNegatives(ref arena); break;
                    case TestType.Symmetric: Symmetric(ref arena); break;
                    case TestType.Diagonal: Diagonal(ref arena); break;
                    case TestType.UpperTriangular: UpperTriangular(ref arena); break;
                    case TestType.LowerTriangular: LowerTriangular(ref arena); break;
                    case TestType.NonSquare: NonSquare(ref arena); break;
                    default: throw new NotImplementedException();
                }
            }
            finally
            {
                arena.Dispose();
            }
        }

        // All-zero vector -> isZero true; a single nonzero entry -> false.
        void ZeroVector(ref Arena arena)
        {
            var v = arena.iProxyVec(5, (iProxy)0);
            Assert.IsTrue(Analysis.isZero(in v));

            v[3] = (iProxy)(-7);
            Assert.IsFalse(Analysis.isZero(in v));
        }

        // All-zero matrix -> isZero true; a single nonzero entry -> false (and it is NOT identity).
        void ZeroMatrix(ref Arena arena)
        {
            var A = arena.iProxyMat(3, 3, (iProxy)0);
            Assert.IsTrue(Analysis.isZero(in A));
            Assert.IsFalse(Analysis.isIdentity(in A)); // all-zero is not identity

            A[1, 2] = (iProxy)1;
            Assert.IsFalse(Analysis.isZero(in A));
            Assert.IsFalse(Analysis.isIdentity(in A));
        }

        // Genuine identity -> isIdentity, isSymmetric, isDiagonal all true; also not zero.
        void Identity(ref Arena arena)
        {
            var A = arena.iProxyIdentityMat(4);
            Assert.IsTrue(Analysis.isIdentity(in A));
            Assert.IsTrue(Analysis.isSymmetric(in A));
            Assert.IsTrue(Analysis.isDiagonal(in A));
            Assert.IsFalse(Analysis.isZero(in A));
        }

        // Negative cases against identity: ONE off-diagonal nonzero breaks isIdentity (and isDiagonal),
        // and a changed diagonal value breaks isIdentity while STILL being diagonal + symmetric.
        void IdentityNegatives(ref Arena arena)
        {
            var A = arena.iProxyIdentityMat(4);
            A[0, 1] = (iProxy)1; // exactly one off-diagonal entry
            Assert.IsFalse(Analysis.isIdentity(in A));
            Assert.IsFalse(Analysis.isDiagonal(in A));
            Assert.IsFalse(Analysis.isSymmetric(in A)); // A[0,1]=1 but A[1,0]=0

            var B = arena.iProxyIdentityMat(4);
            B[2, 2] = (iProxy)5; // diagonal value != 1
            Assert.IsFalse(Analysis.isIdentity(in B));
            Assert.IsTrue(Analysis.isDiagonal(in B));   // still diagonal
            Assert.IsTrue(Analysis.isSymmetric(in B));  // still symmetric
        }

        // Symmetric 3x3 {{1,2,3},{2,4,5},{3,5,6}} -> isSymmetric true; break one entry -> false.
        void Symmetric(ref Arena arena)
        {
            var A = arena.iProxyMat(3, 3);
            A[0, 0] = (iProxy)1; A[0, 1] = (iProxy)2; A[0, 2] = (iProxy)3;
            A[1, 0] = (iProxy)2; A[1, 1] = (iProxy)4; A[1, 2] = (iProxy)5;
            A[2, 0] = (iProxy)3; A[2, 1] = (iProxy)5; A[2, 2] = (iProxy)6;

            Assert.IsTrue(Analysis.isSymmetric(in A));
            Assert.IsFalse(Analysis.isDiagonal(in A)); // has off-diagonal nonzeros
            Assert.IsFalse(Analysis.isIdentity(in A));

            A[0, 1] = (iProxy)9; // now A[0,1]=9 != A[1,0]=2
            Assert.IsFalse(Analysis.isSymmetric(in A));
        }

        // Diagonal 3x3 (values 1,2,3) -> isDiagonal true, isSymmetric true, isIdentity false;
        // one off-diagonal nonzero -> isDiagonal false.
        void Diagonal(ref Arena arena)
        {
            var A = arena.iProxyMat(3, 3, (iProxy)0);
            A[0, 0] = (iProxy)1; A[1, 1] = (iProxy)2; A[2, 2] = (iProxy)3;

            Assert.IsTrue(Analysis.isDiagonal(in A));
            Assert.IsTrue(Analysis.isSymmetric(in A));
            Assert.IsFalse(Analysis.isIdentity(in A));

            A[0, 2] = (iProxy)7;
            Assert.IsFalse(Analysis.isDiagonal(in A));
        }

        // Upper triangular 3x3 {{1,2,3},{0,4,5},{0,0,6}} -> isUpperTriangular true (and NOT
        // lower/diagonal/identity, since it has above-diagonal nonzeros); one below-diagonal
        // nonzero entry breaks it.
        void UpperTriangular(ref Arena arena)
        {
            var A = arena.iProxyMat(3, 3, (iProxy)0);
            A[0, 0] = (iProxy)1; A[0, 1] = (iProxy)2; A[0, 2] = (iProxy)3;
            A[1, 1] = (iProxy)4; A[1, 2] = (iProxy)5;
            A[2, 2] = (iProxy)6;

            Assert.IsTrue(Analysis.isUpperTriangular(in A));
            Assert.IsFalse(Analysis.isLowerTriangular(in A));
            Assert.IsFalse(Analysis.isDiagonal(in A));

            A[1, 0] = (iProxy)7; // exactly one below-diagonal entry
            Assert.IsFalse(Analysis.isUpperTriangular(in A));
        }

        // Lower triangular 3x3 {{1,0,0},{2,3,0},{4,5,6}} -> isLowerTriangular true (and NOT
        // upper/diagonal/identity, since it has below-diagonal nonzeros); one above-diagonal
        // nonzero entry breaks it.
        void LowerTriangular(ref Arena arena)
        {
            var A = arena.iProxyMat(3, 3, (iProxy)0);
            A[0, 0] = (iProxy)1;
            A[1, 0] = (iProxy)2; A[1, 1] = (iProxy)3;
            A[2, 0] = (iProxy)4; A[2, 1] = (iProxy)5; A[2, 2] = (iProxy)6;

            Assert.IsTrue(Analysis.isLowerTriangular(in A));
            Assert.IsFalse(Analysis.isUpperTriangular(in A));
            Assert.IsFalse(Analysis.isDiagonal(in A));

            A[0, 2] = (iProxy)7; // exactly one above-diagonal entry
            Assert.IsFalse(Analysis.isLowerTriangular(in A));
        }

        // Non-square matrices: every square-only predicate must return false (not throw), even when
        // the leading square block looks identity-like/triangular-like.
        void NonSquare(ref Arena arena)
        {
            var A = arena.iProxyMat(2, 3, (iProxy)0);
            A[0, 0] = (iProxy)1; A[1, 1] = (iProxy)1; // identity-looking leading block

            Assert.IsFalse(Analysis.isIdentity(in A));
            Assert.IsFalse(Analysis.isSymmetric(in A));
            Assert.IsFalse(Analysis.isDiagonal(in A));
            Assert.IsFalse(Analysis.isUpperTriangular(in A));
            Assert.IsFalse(Analysis.isLowerTriangular(in A));

            // isZero still works dimension-agnostically on the flat data.
            var Z = arena.iProxyMat(2, 3, (iProxy)0);
            Assert.IsTrue(Analysis.isZero(in Z));
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(AnalysisTestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void AnalysisCases(AnalysisTestJob.TestType type)
    {
        new AnalysisTestJob() { Type = type }.Run();
    }
}
