using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct longMxN {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator +(in longMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator -(in longMxN a)
        {
            longMxN matrix = a.TempCopy();
            
            longComp.signFlipInPlace(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator +(in longMxN lhs, long rhs)
        {
            longMxN matrix = lhs.TempCopy();
            
            longComp.addInPlace(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator +(long lhs, in longMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator -(in longMxN lhs, long rhs)
        {
            longMxN matrix = lhs.TempCopy();
            
            longComp.addInPlace(matrix, (long)(-rhs));

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator -(long lhs, in longMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            longMxN matrix = rhs.TempCopy();
            longComp.subInPlace(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator *(in longMxN a, long s)
        {
            longMxN matrix = a.TempCopy();

            longComp.mulInPlace(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator *(long lhs, in longMxN rhs) => rhs * lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator /(in longMxN a, long s)
        {
            longMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            longComp.divInPlace(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator /(long s, in longMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer div by zero).
            longMxN matrix = a.TempCopy();
            longComp.divInPlace(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator %(in longMxN a, long s)
        {
            longMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            longComp.modInPlace(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator %(long s, in longMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer mod by zero).
            longMxN matrix = a.TempCopy();
            longComp.modInPlace(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator ~(in longMxN a) {

            longMxN matrix = a.TempCopy();

            longComp.bitwiseComplementInPlace(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator &(in longMxN a, in long s) {

            longMxN matrix = a.TempCopy();
            longComp.bitwiseAndInPlace(matrix, s);
            return matrix;
        }

        public static longMxN operator &(in long s, in longMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator |(in longMxN a, in long s) {
            longMxN matrix = a.TempCopy();
            longComp.bitwiseOrInPlace(matrix, s);
            return matrix;
        }

        public static longMxN operator |(in long s, in longMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator ^(in longMxN a, in long s) {
            longMxN matrix = a.TempCopy();
            longComp.bitwiseXorInPlace(matrix, s);
            return matrix;
        }

        public static longMxN operator ^(in long s, in longMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator <<(in longMxN a, int shift) {
            longMxN matrix = a.TempCopy();
            longComp.bitwiseLeftShiftInPlace(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator >>(in longMxN a, int shift) {
            longMxN matrix = a.TempCopy();
            longComp.bitwiseRightShiftInPlace(matrix, shift);
            return matrix;
        }

        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator +(in longMxN lhs, in longMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            longMxN matrix = lhs.TempCopy();

            longComp.addInPlace(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

            return matrix;
        }

        /// <summary>Component-wise subtraction; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator -(in longMxN lhs, in longMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            longMxN matrix = lhs.TempCopy();

            longComp.subInPlace(matrix, rhs);

            return matrix;
        }

        /// <summary>Component-wise multiplication; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator *(in longMxN lhs, in longMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            longMxN matrix = lhs.TempCopy();

            longComp.mulInPlace(rhs, matrix);

            return matrix;
        }

        /// <summary>Component-wise division; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator /(in longMxN dividend, in longMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            longMxN newDividendMatrix = dividend.TempCopy();

            longComp.divInPlace(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>Component-wise modulo; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator %(in longMxN dividend, in longMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            longComp.modInPlace(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator &(in longMxN a, in longMxN b) {
            
            Assume.SameDim(in a, in b);

            longMxN matrix = a.TempCopy();
            longComp.bitwiseAndInPlace(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator |(in longMxN a, in longMxN b) {

            Assume.SameDim(in a, in b);

            longMxN matrix = a.TempCopy();
            longComp.bitwiseOrInPlace(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator ^(in longMxN a, in longMxN b) {

            Assume.SameDim(in a, in b);

            longMxN matrix = a.TempCopy();
            longComp.bitwiseXorInPlace(matrix, b);
            return matrix;
        }

        #endregion
    }
}