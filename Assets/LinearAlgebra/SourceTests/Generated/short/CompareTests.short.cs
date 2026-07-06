using LinearAlgebra;
using NUnit.Framework;
using System;

using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class shortCompareTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct TestsJob : IJob 
    {
        public enum TestType
        {
            VecEquals,
            VecNotEquals,
            VecLess,
            VecLessOrEqual,
            VecGreater,
            VecGreaterOrEqual,

            MatEquals,
            MatNotEquals,
            MatLess,
            MatLessOrEqual,
            MatGreater,
            MatGreaterOrEqual,

            VecRandom,
            MatRandom,

            MatDiagonal,

            VecVecEquals,
            VecVecNotEquals,
            VecVecLess,
            VecVecLessOrEqual,
            VecVecGreater,
            VecVecGreaterOrEqual,
            VecVecRandom,

            MatMatEquals,
            MatMatNotEquals,
            MatMatLess,
            MatMatLessOrEqual,
            MatMatGreater,
            MatMatGreaterOrEqual,
            MatMatRandom,

            VecIsPow2,
            MatIsPow2,
        }

        public TestType Type;

        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            try
            {

                switch (Type)
                {
                    case TestType.VecEquals:
                        VecEquals(ref arena);
                        break;
                    case TestType.VecNotEquals:
                        VecNotEquals(ref arena);
                        break;
                    case TestType.VecLess:
                        VecLess(ref arena);
                        break;
                    case TestType.VecLessOrEqual:
                        VecLessOrEqual(ref arena);
                        break;
                    case TestType.VecGreater:
                        VecGreater(ref arena);
                        break;
                    case TestType.VecGreaterOrEqual:
                        VecGreaterOrEqual(ref arena);
                        break;
                    case TestType.VecRandom:
                        VecRandom(ref arena);
                        break;

                    case TestType.MatEquals:
                        MatEquals(ref arena);
                        break;
                    case TestType.MatNotEquals:
                        MatNotEquals(ref arena);
                        break;
                    case TestType.MatLess:
                        MatLess(ref arena);
                        break;
                    case TestType.MatLessOrEqual:
                        MatLessOrEqual(ref arena);
                        break;
                    case TestType.MatGreater:
                        MatGreater(ref arena);
                        break;
                    case TestType.MatGreaterOrEqual:
                        MatGreaterOrEqual(ref arena);
                        break;

                    case TestType.MatRandom:
                        MatRandom(ref arena);
                        break;
                    case TestType.MatDiagonal:
                        MatDiagonal(ref arena);
                        break;

                    case TestType.VecVecEquals:
                        VecVecEquals(ref arena);
                        break;
                    case TestType.VecVecNotEquals:
                        VecVecNotEquals(ref arena);
                        break;
                    case TestType.VecVecLess:
                        VecVecLess(ref arena);
                        break;
                    case TestType.VecVecLessOrEqual:
                        VecVecLessOrEqual(ref arena);
                        break;
                    case TestType.VecVecGreater:
                        VecVecGreater(ref arena);
                        break;
                    case TestType.VecVecGreaterOrEqual:
                        VecVecGreaterOrEqual(ref arena);
                        break;
                    case TestType.VecVecRandom:
                        VecVecRandom(ref arena);
                        break;
                    
                    case TestType.MatMatEquals:
                        MatMatEquals(ref arena);
                        break;
                    case TestType.MatMatNotEquals:
                        MatMatNotEquals(ref arena);
                        break;
                    case TestType.MatMatLess:
                        MatMatLess(ref arena);
                        break;
                    case TestType.MatMatLessOrEqual:
                        MatMatLessOrEqual(ref arena);
                        break;
                    case TestType.MatMatGreater:
                        MatMatGreater(ref arena);
                        break;
                    case TestType.MatMatGreaterOrEqual:
                        MatMatGreaterOrEqual(ref arena);
                        break;
                    case TestType.MatMatRandom:
                        MatMatRandom(ref arena);
                        break;

                    case TestType.VecIsPow2:
                        VecIsPow2(ref arena);
                        break;
                    case TestType.MatIsPow2:
                        MatIsPow2(ref arena);
                        break;

                    default:
                        throw new System.NotImplementedException();
                }
            }
            finally
            {
                arena.Dispose();
            }
        }

        public void VecEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortN v = arena.shortVec(dim);

            var boolVec = v == 0;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));

            boolVec = v == 1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, false));
        }

        public void VecNotEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortN v = arena.shortVec(dim);

            var boolVec = v != 0;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, false));

            boolVec = v != 1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));
        }

        public void VecLess(ref Arena arena)
        {
            int dim = 16;
            
            shortN v = arena.shortVec(dim);

            var boolVec = v < 0;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, false));

            boolVec = v < 1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));
        }

        public void VecLessOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortN v = arena.shortVec(dim);

            var boolVec = v <= 0;

            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));
            boolVec = v <= 1;

            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));

            boolVec = v <= -1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, false));
        }

        public void VecGreater(ref Arena arena)
        {
            int dim = 16;
            
            shortN v = arena.shortVec(dim);

            var boolVec = v > 0;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, false));

            boolVec = v > -1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));
        }

        public void VecGreaterOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortN v = arena.shortVec(dim);

            var boolVec = v >= 0;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));

            boolVec = v >= -1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));

            boolVec = v >= 1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, false));
        }

        public void MatEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m = arena.shortMat(dim, dim);

            var boolMat = m == 0;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));

            boolMat = m == 1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, false));
        }

        public void MatNotEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m = arena.shortMat(dim, dim);

            var boolMat = m != 0;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, false));

            boolMat = m != 1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));
        }

        public void MatLess(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m = arena.shortMat(dim, dim);

            var boolMat = m < 0;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, false));

            boolMat = m < 1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));
        }

        public void MatLessOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m = arena.shortMat(dim, dim);

            var boolMat = m <= 0;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));

            boolMat = m <= 1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));

            boolMat = m <= -1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, false));
        }

        public void MatGreater(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m = arena.shortMat(dim, dim);

            var boolMat = m > 0;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, false));

            boolMat = m > -1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));
        }

        public void MatGreaterOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m = arena.shortMat(dim, dim);

            var boolMat = m >= 0;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));

            boolMat = m >= -1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));

            boolMat = m >= 1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, false));
        }

        public void VecRandom(ref Arena arena)
        {
            int dim = 64;

            shortN v = arena.shortRandomVec(dim, -100, 100, 1451);
            v[0] = 0;

            var boolVec = v == 0;

            Assert.IsFalse(Analysis.IsAllSame(boolVec));

            boolVec = v != 0;
            Assert.IsFalse(Analysis.IsAllSame(boolVec));

            boolVec = v < 0;
            Assert.IsFalse(Analysis.IsAllSame(boolVec));

            boolVec = v > 0;
            Assert.IsFalse(Analysis.IsAllSame(boolVec));

            boolVec = v <= 0;
            Assert.IsFalse(Analysis.IsAllSame(boolVec));

            boolVec = v >= 0;
            Assert.IsFalse(Analysis.IsAllSame(boolVec));
        }

        public void MatRandom(ref Arena arena)
        {
            int dim = 32;

            shortMxN m = arena.shortRandomMat(dim, dim, -100, 100, 1451);
            m[0,0] = 0;

            var boolMat = m == 0;

            Assert.IsFalse(Analysis.IsAllSame(boolMat));

            boolMat = m != 0;
            Assert.IsFalse(Analysis.IsAllSame(boolMat));

            boolMat = m < 0;
            Assert.IsFalse(Analysis.IsAllSame(boolMat));

            boolMat = m > 0;
            Assert.IsFalse(Analysis.IsAllSame(boolMat));

            boolMat = m <= 0;
            Assert.IsFalse(Analysis.IsAllSame(boolMat));

            boolMat = m >= 0;
            Assert.IsFalse(Analysis.IsAllSame(boolMat));
        }

        public void MatDiagonal(ref Arena arena)
        {
            int dim = 32;

            shortMxN m0 = arena.shortDiagonalMat(dim, 1);
            
            var boolMat = m0 == 1;

            Assert.IsTrue(Analysis.isDiagonal(boolMat));
            Assert.IsFalse(Analysis.IsAllEqualTo(boolMat, true));
            Assert.IsFalse(Analysis.IsAllEqualTo(boolMat, false));
        }

        public void VecVecEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortN v0 = arena.shortLinVec(dim, 0, 100);
            shortN v1 = arena.shortLinVec(dim, 0, 100);

            var boolVec = v0 == v1;

            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));

            v0[0] = 1;

            boolVec = v0 == v1;

            Assert.IsFalse(Analysis.IsAllSame(boolVec));
        }

        public void VecVecNotEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortN v0 = arena.shortLinVec(dim, 0, 100);
            shortN v1 = arena.shortLinVec(dim, 200, 300);

            var boolVec = v0 != v1;

            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));

            v0 = v1;

            boolVec = v0 != v1;

            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, false));
        }

        public void VecVecLess(ref Arena arena)
        {

            int dim = 16;
            
            shortN v0 = arena.shortLinVec(dim, 0, 100);
            shortN v1 = arena.shortLinVec(dim, 200, 300);

            var boolVec = v0 < v1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));

            v0 = v1;

            boolVec = v0 < v1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, false));
        }

        public void VecVecLessOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortN v0 = arena.shortLinVec(dim, 0, 100);
            shortN v1 = arena.shortLinVec(dim, 200, 300);

            var boolVec = v0 <= v1;

            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));

            v0 = v1;

            boolVec = v0 <= v1;

            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));

            v0 = arena.shortLinVec(dim, 0, 100);
            v1 = arena.shortLinVec(dim, 100, 0);

            boolVec = v0 <= v1;

            Assert.IsFalse(Analysis.IsAllSame(boolVec));
        }

        public void VecVecGreater(ref Arena arena)
        {
            int dim = 16;
            
            shortN v0 = arena.shortLinVec(dim, 0, 100);
            shortN v1 = arena.shortLinVec(dim, 200, 300);

            var boolVec = v0 > v1;

            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, false));

            v0 = v1;

            boolVec = v0 > v1;

            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, false));

            v0 = arena.shortLinVec(dim, 100, 0);
            v1 = arena.shortLinVec(dim, 0, 100);

            boolVec = v0 > v1;
            Assert.IsFalse(Analysis.IsAllSame(boolVec));

            boolVec = v1 > v0;
            Assert.IsFalse(Analysis.IsAllSame(boolVec));
        }

        public void VecVecGreaterOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortN v0 = arena.shortLinVec(dim, 0, 100);
            shortN v1 = arena.shortLinVec(dim, 200, 300);

            var boolVec = v0 >= v1;

            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, false));
            v0 = v1;

            boolVec = v0 >= v1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolVec, true));

            v0 = arena.shortLinVec(dim, 1, 0);

            boolVec = v0 >= v1;
            Assert.IsTrue(Analysis.IsAllSame(boolVec));
        }

        public void VecVecRandom(ref Arena arena)
        {
            int dim = 64;

            shortN v0 = arena.shortRandomVec(dim, -100, 100, 1451);
            shortN v1 = arena.shortRandomVec(dim, -100, 100, 6421);

            v0[0] = v1[0];
            v0[1] = (short)(1-v1[1]);
            var boolVec = v0 == v1;

            Assert.IsFalse(Analysis.IsAllSame(boolVec));

            boolVec = v0 != v1;
            Assert.IsFalse(Analysis.IsAllSame(boolVec));

            boolVec = v0 < v1;
            Assert.IsFalse(Analysis.IsAllSame(boolVec));

            boolVec = v0 > v1;
            Assert.IsFalse(Analysis.IsAllSame(boolVec));

            boolVec = v0 <= v1;
            Assert.IsFalse(Analysis.IsAllSame(boolVec));

            boolVec = v0 >= v1;
            Assert.IsFalse(Analysis.IsAllSame(boolVec));
        }

        public void MatMatEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m0 = arena.shortRandomMat(dim, dim, 0, 100);
            shortMxN m1 = arena.shortRandomMat(dim, dim, 0, 100);

            var boolMat = m0 == m1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));

            m0[0,0] = 1;
            m0[1,1] = 1;
            m0[2,2] = 1;
            m0[3,3] = 1;

            boolMat = m0 == m1;
            Assert.IsFalse(Analysis.IsAllSame(boolMat));
        }

        public void MatMatNotEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m0 = arena.shortRandomMat(dim, dim, 0, 100, 2131);
            shortMxN m1 = arena.shortRandomMat(dim, dim, 0, 100, 2131);

            var boolMat = m0 != m1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, false));

            m1 = arena.shortRandomMat(dim, dim, 200, 300, 2131);

            boolMat = m0 != m1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));
        }

        public void MatMatLess(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m0 = arena.shortRandomMat(dim, dim, 000, 100, 2131);
            shortMxN m1 = arena.shortRandomMat(dim, dim, 200, 300, 2131);

            var boolMat = m0 < m1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));

            m0 = m1;

            boolMat = m0 < m1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, false));
        }

        public void MatMatLessOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m0 = arena.shortRandomMat(dim, dim, 0, 100, 2131);
            shortMxN m1 = arena.shortRandomMat(dim, dim, 200, 300, 2131);

            var boolMat = m0 <= m1;

            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));

            m0 = m1;

            boolMat = m0 <= m1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));

            m0 = arena.shortRandomMat(dim, dim, 100, 0, 2131);
            m1 = arena.shortRandomMat(dim, dim, 0, 100, 2131);

            boolMat = m0 <= m1;

            Assert.IsFalse(Analysis.IsAllSame(boolMat));
        }

        public void MatMatGreater(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m0 = arena.shortRandomMat(dim, dim, 0, 100, 2131);
            shortMxN m1 = arena.shortRandomMat(dim, dim, 200, 300, 2131);

            var boolMat = m0 > m1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, false));

            m0 = m1;

            boolMat = m0 > m1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, false));

            m0 = arena.shortRandomMat(dim, dim, 100, 0, 2131);
            m1 = arena.shortRandomMat(dim, dim, 0, 100, 2131);

            boolMat = m0 > m1;
            Assert.IsFalse(Analysis.IsAllSame(boolMat));

            boolMat = m1 > m0;
            Assert.IsFalse(Analysis.IsAllSame(boolMat));
        }

        public void MatMatGreaterOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m0 = arena.shortRandomMat(dim, dim, 0, 100, 2131);
            shortMxN m1 = arena.shortRandomMat(dim, dim, 200, 300, 2131);

            var boolMat = m0 >= m1;
            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, false));

            m0 = m1;

            boolMat = m0 >= m1;

            Assert.IsTrue(Analysis.IsAllEqualTo(boolMat, true));
            m0 = arena.shortRandomMat(dim, dim, 100, 0, 2131);

            boolMat = m0 >= m1;
            Assert.IsTrue(Analysis.IsAllSame(boolMat));
        }

        public void MatMatRandom(ref Arena arena)
        {
            int dim = 32;

            shortMxN m0 = arena.shortRandomMat(dim, dim, -100, 100, 1451);
            shortMxN m1 = arena.shortRandomMat(dim, dim, -100, 100, 6421);

            m0[0,0] = m1[0,0];
            m0[0,1] = (short)(1 - m1[0,1]);
            var boolMat = m0 == m1;

            Assert.IsFalse(Analysis.IsAllSame(boolMat));

            boolMat = m0 != m1;
            Assert.IsFalse(Analysis.IsAllSame(boolMat));

            boolMat = m0 < m1;
            Assert.IsFalse(Analysis.IsAllSame(boolMat));

            boolMat = m0 > m1;
            Assert.IsFalse(Analysis.IsAllSame(boolMat));

            boolMat = m0 <= m1;
            Assert.IsFalse(Analysis.IsAllSame(boolMat));

            boolMat = m0 >= m1;
            Assert.IsFalse(Analysis.IsAllSame(boolMat));
        }

        // ispow2 is a genuine PREDICATE (not a relational comparison against a second operand), so
        // it lives here alongside the rest of the boolN/boolMxN-producing comparator surface rather
        // than in shortCompMathTests/shortCompBitsTests - see UnsafeBoolOP.short.cs's ispow2
        // kernel and shortN.Comparators.cs/shortMxN.Comparators.cs's public ispow2() methods.
        public void VecIsPow2(ref Arena arena)
        {
            int n = 7;

            shortN v = arena.shortVec(n);
            v[0] = 0;   // not a power of two
            v[1] = 1;   // 2^0
            v[2] = 2;   // 2^1
            v[3] = 3;   // not a power of two
            v[4] = 4;   // 2^2
            v[5] = -8;  // negative - never a power of two, regardless of its magnitude's bit pattern
            // type MSB pattern (int.MinValue-analog): negative for every SIGNED type, so never a
            // power of two - the adversarial mirror of uint's 0x80000000 -> true (see
            // UIntTypeTests.IsPow2HighBitUnsignedTrue), which relies on the exact same bit pattern
            // being read as POSITIVE (2^31) once there is no sign bit to speak of.
            v[6] = unchecked((short)0x8000);

            var b = v.ispow2();
            Assert.IsFalse(b[0]);
            Assert.IsTrue(b[1]);
            Assert.IsTrue(b[2]);
            Assert.IsFalse(b[3]);
            Assert.IsTrue(b[4]);
            Assert.IsFalse(b[5]);
            Assert.IsFalse(b[6]);
        }

        public void MatIsPow2(ref Arena arena)
        {
            int dim = 4;

            shortMxN m = arena.shortMat(dim, dim);
            for (int i = 0; i < m.Length; i++)
                m[i] = (short)(i + 1); // 1..16 - several exact powers of two among them (1,2,4,8,16)

            var b = m.ispow2();

            // Hardcoded expected literals (matching VecIsPow2's style), not a recomputation of the
            // same x>0 && (x&(x-1))==0 formula the kernel itself uses - 1,2,4,8,16 are the powers of
            // two present in 1..16, everything else in that range is not.
            Assert.IsTrue(b[0]);    // 1  == 2^0
            Assert.IsTrue(b[1]);    // 2  == 2^1
            Assert.IsFalse(b[2]);   // 3
            Assert.IsTrue(b[3]);    // 4  == 2^2
            Assert.IsFalse(b[4]);   // 5
            Assert.IsFalse(b[5]);   // 6
            Assert.IsFalse(b[6]);   // 7
            Assert.IsTrue(b[7]);    // 8  == 2^3
            Assert.IsFalse(b[8]);   // 9
            Assert.IsFalse(b[9]);   // 10
            Assert.IsFalse(b[10]);  // 11
            Assert.IsFalse(b[11]);  // 12
            Assert.IsFalse(b[12]);  // 13
            Assert.IsFalse(b[13]);  // 14
            Assert.IsFalse(b[14]);  // 15
            Assert.IsTrue(b[15]);   // 16 == 2^4
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestsJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void Test(TestsJob.TestType type)
    {
        new TestsJob() { Type = type }.Run();
    }

}
