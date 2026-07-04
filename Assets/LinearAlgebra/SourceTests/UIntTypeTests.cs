using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Concrete (NOT codegen'd) tests for the new unsigned int-family types (uintN / uintMxN /
// uintComp / Blas uint overloads). These are deliberately hand-authored rather than expanded from
// an iProxy test template: the behaviours under test here - modular WRAPAROUND arithmetic, LOGICAL
// (not arithmetic) right shift, and UNSIGNED comparison ordering - all have expected values that
// are specific to an unsigned type and would be flat wrong if the same template were expanded for
// signed int/short/long. So a focused uint-only file is the right shape.
//
// Every exact wrap value below is spelled out (with the mod-2^32 arithmetic that produces it) so a
// future reader can confirm the constant is intentional, not a fudge to match the implementation.
public class UIntTypeTests
{
    const uint UMAX = uint.MaxValue; // 4294967295 == 0xFFFFFFFF

    [BurstCompile]
    public struct TestsJob : IJob
    {
        public enum TestType
        {
            // construction / indexing
            VecConstructIndex,
            MatConstructIndex,

            // arithmetic + wraparound
            AddWrapVec,
            SubWrapVec,
            MulWrapVec,
            DivUnsignedVec,
            ModUnsignedVec,
            ScalarMinusVecWrap,
            AddWrapMat,

            // bitwise + shifts
            BitwiseVec,
            ComplementVec,
            LeftShiftVec,
            RightShiftLogicalVec,
            ShiftLogicalVsArithmetic,

            // comparators -> boolN / boolMxN
            CompareScalarVec,
            CompareUnsignedOrderingVec,
            CompareComponentVec,
            CompareScalarMat,
            CompareComponentMat,

            // Blas
            DotVecVec,
            DotMatVec,
            DotVecMat,
            DotMatMat,
            OuterDot,
            Transpose,

            // uintComp in-place
            InPlaceScalar,
            InPlaceComponent,
            InPlaceWrap,
            InPlaceBitwise,
            ClampInPlace,

            // uintComp elementwise math (min/max/mad exist for uint; abs/relu deliberately do NOT)
            MinMaxInPlace,
            MadInPlace,

            // Select (LinearAlgebra.Select) - pure data movement, fully unsigned-safe
            SelectVecMask,
            SelectMatScalarCond,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.VecConstructIndex: VecConstructIndex(); break;
                case TestType.MatConstructIndex: MatConstructIndex(); break;

                case TestType.AddWrapVec: AddWrapVec(); break;
                case TestType.SubWrapVec: SubWrapVec(); break;
                case TestType.MulWrapVec: MulWrapVec(); break;
                case TestType.DivUnsignedVec: DivUnsignedVec(); break;
                case TestType.ModUnsignedVec: ModUnsignedVec(); break;
                case TestType.ScalarMinusVecWrap: ScalarMinusVecWrap(); break;
                case TestType.AddWrapMat: AddWrapMat(); break;

                case TestType.BitwiseVec: BitwiseVec(); break;
                case TestType.ComplementVec: ComplementVec(); break;
                case TestType.LeftShiftVec: LeftShiftVec(); break;
                case TestType.RightShiftLogicalVec: RightShiftLogicalVec(); break;
                case TestType.ShiftLogicalVsArithmetic: ShiftLogicalVsArithmetic(); break;

                case TestType.CompareScalarVec: CompareScalarVec(); break;
                case TestType.CompareUnsignedOrderingVec: CompareUnsignedOrderingVec(); break;
                case TestType.CompareComponentVec: CompareComponentVec(); break;
                case TestType.CompareScalarMat: CompareScalarMat(); break;
                case TestType.CompareComponentMat: CompareComponentMat(); break;

                case TestType.DotVecVec: DotVecVec(); break;
                case TestType.DotMatVec: DotMatVec(); break;
                case TestType.DotVecMat: DotVecMat(); break;
                case TestType.DotMatMat: DotMatMat(); break;
                case TestType.OuterDot: OuterDot(); break;
                case TestType.Transpose: Transpose(); break;

                case TestType.InPlaceScalar: InPlaceScalar(); break;
                case TestType.InPlaceComponent: InPlaceComponent(); break;
                case TestType.InPlaceWrap: InPlaceWrap(); break;
                case TestType.InPlaceBitwise: InPlaceBitwise(); break;
                case TestType.ClampInPlace: ClampInPlace(); break;

                case TestType.MinMaxInPlace: MinMaxInPlace(); break;
                case TestType.MadInPlace: MadInPlace(); break;

                case TestType.SelectVecMask: SelectVecMask(); break;
                case TestType.SelectMatScalarCond: SelectMatScalarCond(); break;

                default: throw new NotImplementedException();
            }
        }

        // ---- construction / indexing ----------------------------------------------------------

        void VecConstructIndex()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16;
            uintN a = arena.uintVec(n, 10u);

            Assert.AreEqual(n, a.N);

            for (int i = 0; i < n; i++)
                Assert.IsTrue(a[i] == 10u);

            // write-through indexer
            a[0] = 7u;
            a[n - 1] = UMAX;
            Assert.IsTrue(a[0] == 7u);
            Assert.IsTrue(a[n - 1] == UMAX);

            uintN z = arena.uintVec(n); // default (zero-filled)
            for (int i = 0; i < n; i++)
                Assert.IsTrue(z[i] == 0u);

            // 1x1 degenerate vector
            uintN one = arena.uintVec(1, 42u);
            Assert.AreEqual(1, one.N);
            Assert.IsTrue(one[0] == 42u);

            arena.Dispose();
        }

        void MatConstructIndex()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 3, cols = 4;
            uintMxN a = arena.uintMat(rows, cols, 5u);

            Assert.AreEqual(rows, a.M_Rows);
            Assert.AreEqual(cols, a.N_Cols);
            Assert.AreEqual(rows * cols, a.Length);

            for (int i = 0; i < a.Length; i++)
                Assert.IsTrue(a[i] == 5u);

            a[1, 2] = 99u;
            Assert.IsTrue(a[1, 2] == 99u);
            Assert.IsTrue(a[1 * cols + 2] == 99u); // row-major flat index agrees with [r,c]

            uintMxN id = arena.uintIdentityMat(cols);
            for (int i = 0; i < cols; i++)
                for (int j = 0; j < cols; j++)
                    Assert.IsTrue(id[i, j] == (i == j ? 1u : 0u));

            arena.Dispose();
        }

        // ---- arithmetic + wraparound ----------------------------------------------------------

        void AddWrapVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            // MaxValue + 1 wraps to 0.
            uintN a = arena.uintVec(n, UMAX);
            a += 1u;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(a[i] == 0u);

            // MaxValue + 5 == 4  (4294967295 + 5 = 4294967300; - 2^32 (4294967296) = 4).
            a = arena.uintVec(n, UMAX);
            a += 5u;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(a[i] == 4u);

            // component-wise wrap: MaxValue + 1 == 0
            a = arena.uintVec(n, UMAX);
            uintN ones = arena.uintVec(n, 1u);
            uintN r = a + ones;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(r[i] == 0u);

            arena.Dispose();
        }

        void SubWrapVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            // 0 - 1 underflows to MaxValue.
            uintN a = arena.uintVec(n, 0u);
            a -= 1u;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(a[i] == UMAX);

            // 3 - 10 == 4294967289  (-7 mod 2^32 == 4294967296 - 7).
            a = arena.uintVec(n, 3u);
            a -= 10u;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(a[i] == 4294967289u);

            // component-wise underflow: 0 - 1 == MaxValue
            a = arena.uintVec(n, 0u);
            uintN ones = arena.uintVec(n, 1u);
            uintN r = a - ones;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(r[i] == UMAX);

            arena.Dispose();
        }

        void MulWrapVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            // 2^16 * 2^16 = 2^32 ≡ 0 (mod 2^32).
            uintN a = arena.uintVec(n, 65536u);
            a *= 65536u;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(a[i] == 0u);

            // 2^31 * 2 = 2^32 ≡ 0.
            a = arena.uintVec(n, 2147483648u);
            a *= 2u;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(a[i] == 0u);

            // 3000000000 * 2 = 6000000000; - 2^32 = 1705032704.
            a = arena.uintVec(n, 3000000000u);
            a *= 2u;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(a[i] == 1705032704u);

            arena.Dispose();
        }

        void DivUnsignedVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            // Unsigned division: MaxValue / 2 == 2147483647 (as SIGNED, -1/2 would be 0).
            uintN a = arena.uintVec(n, UMAX);
            a /= 2u;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(a[i] == 2147483647u);

            // 10 / 3 == 3 (integer truncation)
            a = arena.uintVec(n, 10u);
            a /= 3u;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(a[i] == 3u);

            // scalar / vec form
            a = arena.uintVec(n, 3u);
            uintN r = 10u / a;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(r[i] == 3u);

            arena.Dispose();
        }

        void ModUnsignedVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            // Unsigned modulo: MaxValue % 2 == 1 (MaxValue is odd). Signed -1 % 2 would be -1.
            uintN a = arena.uintVec(n, UMAX);
            a %= 2u;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(a[i] == 1u);

            a = arena.uintVec(n, 10u);
            a %= 3u;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(a[i] == 1u);

            arena.Dispose();
        }

        void ScalarMinusVecWrap()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            // s - a with s < a underflows: 3 - 5 == 4294967294 (-2 mod 2^32).
            uintN a = arena.uintVec(n, 5u);
            uintN r = 3u - a;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(r[i] == 4294967294u);

            arena.Dispose();
        }

        void AddWrapMat()
        {
            var arena = new Arena(Allocator.Persistent);
            int rows = 8, cols = 8;

            uintMxN a = arena.uintMat(rows, cols, UMAX);
            a += 2u; // MaxValue + 2 == 1
            for (int i = 0; i < a.Length; i++)
                Assert.IsTrue(a[i] == 1u);

            arena.Dispose();
        }

        // ---- bitwise + shifts -----------------------------------------------------------------

        void BitwiseVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            uintN a = arena.uintVec(n, 0xF0F0F0F0u);

            uintN andR = a & 0x0F0F0F0Fu; // disjoint bits -> 0
            uintN orR = a | 0x0F0F0F0Fu;  // union -> all ones
            uintN xorR = a ^ 0xFFFFFFFFu; // flip -> 0x0F0F0F0F

            for (int i = 0; i < n; i++)
            {
                Assert.IsTrue(andR[i] == 0u);
                Assert.IsTrue(orR[i] == 0xFFFFFFFFu);
                Assert.IsTrue(xorR[i] == 0x0F0F0F0Fu);
            }

            // component-wise bitwise
            uintN b = arena.uintVec(n, 0x0F0F0F0Fu);
            uintN andC = a & b;
            uintN orC = a | b;
            uintN xorC = a ^ b;
            for (int i = 0; i < n; i++)
            {
                Assert.IsTrue(andC[i] == 0u);
                Assert.IsTrue(orC[i] == 0xFFFFFFFFu);
                Assert.IsTrue(xorC[i] == 0xFFFFFFFFu);
            }

            arena.Dispose();
        }

        void ComplementVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            uintN a = arena.uintVec(n, 0u);
            uintN c = ~a; // ~0 == 0xFFFFFFFF == MaxValue
            for (int i = 0; i < n; i++)
                Assert.IsTrue(c[i] == UMAX);

            a = arena.uintVec(n, UMAX);
            c = ~a; // ~MaxValue == 0
            for (int i = 0; i < n; i++)
                Assert.IsTrue(c[i] == 0u);

            arena.Dispose();
        }

        void LeftShiftVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            // 1 << 31 lands the bit in the high position: 0x80000000 == 2147483648.
            uintN a = arena.uintVec(n, 1u);
            uintN r = a << 31;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(r[i] == 2147483648u);

            // 0xFFFFFFFF << 4: low 4 bits zero-filled, top 4 bits dropped -> 0xFFFFFFF0.
            a = arena.uintVec(n, 0xFFFFFFFFu);
            r = a << 4;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(r[i] == 0xFFFFFFF0u); // 4294967280

            arena.Dispose();
        }

        void RightShiftLogicalVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            // High bit set, shifted right by 1: LOGICAL shift fills a ZERO in the top bit.
            // 0x80000000 >> 1 == 0x40000000 (== 1073741824), NOT 0xC0000000.
            uintN a = arena.uintVec(n, 0x80000000u);
            uintN r = a >> 1;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(r[i] == 0x40000000u);

            // 0xFFFFFFFF >> 28 == 0xF (== 15): zeros shifted into the top. A signed -1 >> 28 would
            // stay -1 (arithmetic).
            a = arena.uintVec(n, 0xFFFFFFFFu);
            r = a >> 28;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(r[i] == 0xFu);

            arena.Dispose();
        }

        void ShiftLogicalVsArithmetic()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 8;

            // Same 32-bit pattern (0x80000000), interpreted two ways:
            //   - as uint, >> 1 is LOGICAL:   0x80000000 >> 1 == 0x40000000
            //   - as int,  >> 1 is ARITHMETIC: (int)0x80000000 >> 1 == 0xC0000000 (sign-extended)
            uintN a = arena.uintVec(n, 0x80000000u);
            uintN logical = a >> 1;

            int signedPattern = unchecked((int)0x80000000u); // == int.MinValue
            uint arithmeticAsBits = unchecked((uint)(signedPattern >> 1)); // 0xC0000000 == 3221225472

            Assert.IsTrue(arithmeticAsBits == 0xC0000000u); // sanity: signed shift really is arithmetic

            for (int i = 0; i < n; i++)
            {
                Assert.IsTrue(logical[i] == 0x40000000u);         // uint result is logical
                Assert.IsTrue(logical[i] != arithmeticAsBits);     // and differs from arithmetic
            }

            arena.Dispose();
        }

        // ---- comparators -> boolN / boolMxN ---------------------------------------------------

        void CompareScalarVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            uintN v = arena.uintVec(n, 5u);

            Assert.IsTrue(Analysis.IsAllEqualTo(v < 10u, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(v < 5u, false));
            Assert.IsTrue(Analysis.IsAllEqualTo(v <= 5u, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(v > 4u, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(v > 5u, false));
            Assert.IsTrue(Analysis.IsAllEqualTo(v >= 5u, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(v == 5u, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(v == 6u, false));
            Assert.IsTrue(Analysis.IsAllEqualTo(v != 6u, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(v != 5u, false));

            arena.Dispose();
        }

        void CompareUnsignedOrderingVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            // The decisive unsigned-vs-signed test: MaxValue (all bits set) must compare as the
            // LARGEST value, not as -1. Under signed ordering (MaxValue >  0) would be false.
            uintN big = arena.uintVec(n, UMAX);
            Assert.IsTrue(Analysis.IsAllEqualTo(big > 0u, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(big >= 0u, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(big < 0u, false));
            Assert.IsTrue(Analysis.IsAllEqualTo(0u < big, true));

            // 0 is the smallest; (0 < MaxValue) is true.
            uintN zero = arena.uintVec(n, 0u);
            Assert.IsTrue(Analysis.IsAllEqualTo(zero < UMAX, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(zero > UMAX, false));

            arena.Dispose();
        }

        void CompareComponentVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            uintN a = arena.uintVec(n, 3u);
            uintN b = arena.uintVec(n, 7u);

            Assert.IsTrue(Analysis.IsAllEqualTo(a < b, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(a > b, false));
            Assert.IsTrue(Analysis.IsAllEqualTo(a <= b, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(b >= a, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(a == b, false));
            Assert.IsTrue(Analysis.IsAllEqualTo(a != b, true));

            a = b; // alias same buffer contents -> equal
            Assert.IsTrue(Analysis.IsAllEqualTo(a == b, true));

            arena.Dispose();
        }

        void CompareScalarMat()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 8;

            uintMxN m = arena.uintMat(dim, dim, 4u);
            boolMxN bm = m == 4u;
            Assert.IsTrue(Analysis.IsAllEqualTo(bm, true));

            Assert.IsTrue(Analysis.IsAllEqualTo(m < 5u, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(m > 4u, false));

            // identity matrix: diagonal ones, off-diagonal zeros -> (m == 1) is exactly diagonal
            uintMxN id = arena.uintIdentityMat(dim);
            boolMxN diag = id == 1u;
            Assert.IsTrue(Analysis.isDiagonal(diag));
            Assert.IsFalse(Analysis.IsAllEqualTo(diag, true));

            arena.Dispose();
        }

        void CompareComponentMat()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 8;

            uintMxN a = arena.uintMat(dim, dim, 2u);
            uintMxN b = arena.uintMat(dim, dim, 9u);

            Assert.IsTrue(Analysis.IsAllEqualTo(a < b, true));
            Assert.IsTrue(Analysis.IsAllEqualTo(a >= b, false));
            Assert.IsTrue(Analysis.IsAllEqualTo(a != b, true));

            arena.Dispose();
        }

        // ---- Blas -----------------------------------------------------------------------------

        void DotVecVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 32;

            uintN x = arena.uintVec(n, 1u);
            uintN y = arena.uintVec(n, 1u);
            uint d = Blas.dot(x, y);
            Assert.IsTrue(d == (uint)n); // sum of n ones

            // 2 . 3 over n elements == 6n
            x = arena.uintVec(n, 2u);
            y = arena.uintVec(n, 3u);
            d = Blas.dot(x, y);
            Assert.IsTrue(d == (uint)(6 * n));

            arena.Dispose();
        }

        void DotMatVec()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 12;

            // I * x == x
            uintMxN A = arena.uintIdentityMat(dim);
            uintN x = arena.uintVec(dim, 7u);
            uintN b = Blas.dot(A, x);
            Assert.AreEqual(dim, b.N);
            for (int i = 0; i < dim; i++)
                Assert.IsTrue(b[i] == 7u);

            // non-square: (out x in) * (in) -> out
            int inLen = 20, outLen = 5;
            uintMxN R = arena.uintMat(outLen, inLen, 2u);
            uintN xin = arena.uintVec(inLen, 3u);
            uintN bout = Blas.dot(R, xin);
            Assert.AreEqual(outLen, bout.N);
            for (int i = 0; i < outLen; i++)
                Assert.IsTrue(bout[i] == (uint)(2 * 3 * inLen)); // each row: sum of inLen*(2*3)

            arena.Dispose();
        }

        void DotVecMat()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 16;

            uintMxN A = arena.uintIdentityMat(dim);
            uintN x = arena.uintIndexOneVec(dim); // [1,2,...,dim]
            uintN b = Blas.dot(x, A);
            Assert.AreEqual(dim, b.N);
            for (int i = 0; i < dim; i++)
                Assert.IsTrue(b[i] == x[i]); // x * I == x

            arena.Dispose();
        }

        void DotMatMat()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 16;

            uintMxN A = arena.uintIdentityMat(dim);
            uintMxN R = arena.uintRandomMat(dim, dim, 0u, 50u);

            // I * R == R
            uintMxN C = Blas.dot(A, R);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    Assert.IsTrue(C[i, j] == R[i, j]);

            // I * I == I
            uintMxN D = Blas.dot(A, A);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    Assert.IsTrue(D[i, j] == (i == j ? 1u : 0u));

            arena.Dispose();
        }

        void OuterDot()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, k = 9;

            uintN x = arena.uintVec(m, 1u);
            uintN y = arena.uintVec(k, 1u);

            uintMxN A = Blas.outerDot(x, y);
            Assert.AreEqual(m, A.M_Rows);
            Assert.AreEqual(k, A.N_Cols);
            for (int i = 0; i < A.Length; i++)
                Assert.IsTrue(A[i] == 1u); // 1 * 1 everywhere

            // outer product values: (i+1)*(j+1)
            uintN xi = arena.uintIndexOneVec(m);
            uintN yj = arena.uintIndexOneVec(k);
            uintMxN B = Blas.outerDot(xi, yj);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < k; j++)
                    Assert.IsTrue(B[i, j] == xi[i] * yj[j]);

            arena.Dispose();
        }

        void Transpose()
        {
            var arena = new Arena(Allocator.Persistent);
            int rows = 5, cols = 8;

            uintMxN A = arena.uintMat(rows, cols, 0u);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    A[i, j] = (uint)(i * cols + j + 1);

            uintMxN T = Blas.trans(A);
            Assert.AreEqual(cols, T.M_Rows);
            Assert.AreEqual(rows, T.N_Cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    Assert.IsTrue(T[j, i] == A[i, j]);

            // double transpose is identity
            uintMxN TT = Blas.trans(T);
            Assert.AreEqual(rows, TT.M_Rows);
            Assert.AreEqual(cols, TT.N_Cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    Assert.IsTrue(TT[i, j] == A[i, j]);

            arena.Dispose();
        }

        // ---- uintComp in-place ----------------------------------------------------------------

        void InPlaceScalar()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            uintN v = arena.uintVec(n, 10u);

            uintComp.addInPlace(v, 5u);   // 15
            for (int i = 0; i < n; i++) Assert.IsTrue(v[i] == 15u);

            uintComp.subInPlace(v, 3u);   // 12
            for (int i = 0; i < n; i++) Assert.IsTrue(v[i] == 12u);

            uintComp.mulInPlace(v, 2u);   // 24
            for (int i = 0; i < n; i++) Assert.IsTrue(v[i] == 24u);

            uintComp.divInPlace(v, 4u);   // 6
            for (int i = 0; i < n; i++) Assert.IsTrue(v[i] == 6u);

            uintComp.modInPlace(v, 4u);   // 2
            for (int i = 0; i < n; i++) Assert.IsTrue(v[i] == 2u);

            arena.Dispose();
        }

        void InPlaceComponent()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            uintN a = arena.uintVec(n, 10u);
            uintN b = arena.uintVec(n, 4u);

            uintComp.addInPlace(a, b); // a += b -> 14, b unchanged
            for (int i = 0; i < n; i++)
            {
                Assert.IsTrue(a[i] == 14u);
                Assert.IsTrue(b[i] == 4u);
            }

            uintComp.subInPlace(a, b); // a -= b -> 10
            for (int i = 0; i < n; i++)
                Assert.IsTrue(a[i] == 10u);

            // NOTE: the buffer (T,T) mulInPlace overload mutates its SECOND operand
            // (kernel compMul(from, target) does target *= from) - this is the convention the
            // component-wise operator* relies on. So mulInPlace(a, b) computes b *= a, leaving a
            // unchanged. (Asymmetric with add/subInPlace, which mutate the first operand.)
            uintComp.mulInPlace(a, b); // b *= a -> 40 ; a stays 10
            for (int i = 0; i < n; i++)
            {
                Assert.IsTrue(b[i] == 40u);
                Assert.IsTrue(a[i] == 10u);
            }

            arena.Dispose();
        }

        void InPlaceWrap()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            // In-place addition wraps the same way the operator does.
            uintN v = arena.uintVec(n, UMAX);
            uintComp.addInPlace(v, 1u);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(v[i] == 0u);

            v = arena.uintVec(n, 0u);
            uintComp.subInPlace(v, 1u); // 0 - 1 -> MaxValue
            for (int i = 0; i < n; i++)
                Assert.IsTrue(v[i] == UMAX);

            arena.Dispose();
        }

        void InPlaceBitwise()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            uintN v = arena.uintVec(n, 0xFF00FF00u);
            uintComp.bitwiseComplementInPlace(v); // -> 0x00FF00FF
            for (int i = 0; i < n; i++)
                Assert.IsTrue(v[i] == 0x00FF00FFu);

            v = arena.uintVec(n, 0xFFFFFFFFu);
            uintComp.bitwiseAndInPlace(v, 0x0000FFFFu);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(v[i] == 0x0000FFFFu);

            v = arena.uintVec(n, 0x80000000u);
            uintComp.bitwiseRightShiftInPlace(v, 1); // logical -> 0x40000000
            for (int i = 0; i < n; i++)
                Assert.IsTrue(v[i] == 0x40000000u);

            v = arena.uintVec(n, 1u);
            uintComp.bitwiseLeftShiftInPlace(v, 31); // -> 0x80000000
            for (int i = 0; i < n; i++)
                Assert.IsTrue(v[i] == 0x80000000u);

            arena.Dispose();
        }

        void ClampInPlace()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            uintN v = arena.uintIndexZeroVec(n); // [0,1,...,n-1]
            uintComp.clampInPlace(v, 3u, 8u);
            for (int i = 0; i < n; i++)
            {
                uint expected = (uint)i;
                if (expected < 3u) expected = 3u;
                if (expected > 8u) expected = 8u;
                Assert.IsTrue(v[i] == expected);
            }

            arena.Dispose();
        }

        // uintComp elementwise math: min/max/mad ARE generated for uint (unsigned-clean), and their
        // buffer overloads mutate the FIRST operand (x for min/max, a for mad), leaving the other
        // operand(s) untouched. abs/reluInPlace are intentionally absent for uint (there is no
        // negative to take the magnitude of / clamp to zero) - they are skipFor'd off the kernel, so
        // a `v.absInPlace()` / `v.reluInPlace()` here would fail to COMPILE. That absence is the
        // contract; we simply never call them (a compile-level, not runtime, guarantee).
        void MinMaxInPlace()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            uintN x = arena.uintIndexZeroVec(n);          // [0,1,...,n-1]
            uintN y = arena.uintVec(n, 5u);               // constant 5
            uintN y0 = y.Copy();

            uintComp.minInPlace(x, y); // x = min(x, y); y untouched
            for (int i = 0; i < n; i++)
            {
                uint xi = (uint)i;
                Assert.IsTrue(x[i] == (xi < 5u ? xi : 5u));
                Assert.IsTrue(y[i] == y0[i]);
            }

            uintN a = arena.uintIndexZeroVec(n);          // [0,1,...,n-1]
            uintN b = arena.uintVec(n, 5u);
            uintN b0 = b.Copy();

            uintComp.maxInPlace(a, b); // a = max(a, b); b untouched
            for (int i = 0; i < n; i++)
            {
                uint ai = (uint)i;
                Assert.IsTrue(a[i] == (ai > 5u ? ai : 5u));
                Assert.IsTrue(b[i] == b0[i]);
            }

            // unsigned ordering: MaxValue is the LARGEST (not -1), so max picks it, min rejects it.
            uintN big = arena.uintVec(n, UMAX);
            uintN small = arena.uintVec(n, 1u);
            uintN small0 = small.Copy();
            uintComp.maxInPlace(big, small);
            for (int i = 0; i < n; i++) Assert.IsTrue(big[i] == UMAX);
            big = arena.uintVec(n, UMAX);
            uintComp.minInPlace(big, small);
            for (int i = 0; i < n; i++)
            {
                Assert.IsTrue(big[i] == 1u);
                Assert.IsTrue(small[i] == small0[i]);
            }

            arena.Dispose();
        }

        void MadInPlace()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;

            uintN a = arena.uintVec(n, 3u);
            uintN b = arena.uintVec(n, 4u);
            uintN c = arena.uintVec(n, 2u);
            uintN b0 = b.Copy();
            uintN c0 = c.Copy();

            uintComp.madInPlace(a, b, c); // a = a*b + c = 14 ; b, c untouched
            for (int i = 0; i < n; i++)
            {
                Assert.IsTrue(a[i] == 14u);
                Assert.IsTrue(b[i] == b0[i]);
                Assert.IsTrue(c[i] == c0[i]);
            }

            // modular wraparound flows through mad the same as bare arithmetic:
            // MaxValue * 1 + 1 == 0 (mod 2^32).
            uintN wa = arena.uintVec(n, UMAX);
            uintN wb = arena.uintVec(n, 1u);
            uintN wc = arena.uintVec(n, 1u);
            uintComp.madInPlace(wa, wb, wc);
            for (int i = 0; i < n; i++) Assert.IsTrue(wa[i] == 0u);

            arena.Dispose();
        }

        // ---- Select (LinearAlgebra.Select) -----------------------------------------------------
        // select() is pure data movement (dst[i] = c[i] ? b[i] : a[i]) - no comparison or
        // arithmetic - so it is unsigned-clean with no wrap/ordering caveats to cover; these two
        // cases just confirm the uint overloads pick the right operand, including UMAX (which
        // would misbehave under signed reasoning if select ever became comparison-based).

        void SelectVecMask()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 8;

            uintN a = arena.uintVec(n, 1u);
            uintN b = arena.uintVec(n, UMAX);
            boolN c = arena.boolVec(n); // zero-filled (all false)
            for (int i = 0; i < n; i++)
                c[i] = (i % 2) == 0; // alternate false/true

            uintN r = Select.select(a, b, c); // dest[i] = c[i] ? b[i] : a[i]
            for (int i = 0; i < n; i++)
                Assert.IsTrue(r[i] == (c[i] ? UMAX : 1u));

            arena.Dispose();
        }

        void SelectMatScalarCond()
        {
            var arena = new Arena(Allocator.Persistent);
            int rows = 3, cols = 4;

            uintMxN a = arena.uintMat(rows, cols, 2u);
            uintMxN b = arena.uintMat(rows, cols, UMAX);

            uintMxN rTrue = Select.select(a, b, true); // c=true -> b
            for (int i = 0; i < rTrue.Length; i++)
                Assert.IsTrue(rTrue[i] == UMAX);

            uintMxN rFalse = Select.select(a, b, false); // c=false -> a
            for (int i = 0; i < rFalse.Length; i++)
                Assert.IsTrue(rFalse[i] == 2u);

            arena.Dispose();
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

    // ---- Non-Burst construction sanity outside a job (mirrors ArenaLayoutTests style) --------

    [Test]
    public void UIntVec_AllocationTracking()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            uintN v = arena.uintVec(8, 1u);
            Assert.AreEqual(8, v.N);
            Assert.AreEqual(1, arena.AllocationsCount);
        }
        finally { arena.Dispose(); }
    }
}
