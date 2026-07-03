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
            
            longComp.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator +(in longMxN lhs, long rhs)
        {
            longMxN matrix = lhs.TempCopy();
            
            longComp.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator +(long lhs, in longMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator -(in longMxN lhs, long rhs)
        {
            longMxN matrix = lhs.TempCopy();
            
            longComp.addInpl(matrix, (long)(-rhs));

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator -(long lhs, in longMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            longMxN matrix = rhs.TempCopy();
            longComp.subInpl(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator *(in longMxN a, long s)
        {
            longMxN matrix = a.TempCopy();

            longComp.mulInpl(matrix, s);

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

            longComp.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator /(long s, in longMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer div by zero).
            longMxN matrix = a.TempCopy();
            longComp.divInpl(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator %(in longMxN a, long s)
        {
            longMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            longComp.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator %(long s, in longMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer mod by zero).
            longMxN matrix = a.TempCopy();
            longComp.modInpl(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator ~(in longMxN a) {

            longMxN matrix = a.TempCopy();

            longComp.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator &(in longMxN a, in long s) {

            longMxN matrix = a.TempCopy();
            longComp.bitwiseAndInpl(matrix, s);
            return matrix;
        }

        public static longMxN operator &(in long s, in longMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator |(in longMxN a, in long s) {
            longMxN matrix = a.TempCopy();
            longComp.bitwiseOrInpl(matrix, s);
            return matrix;
        }

        public static longMxN operator |(in long s, in longMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator ^(in longMxN a, in long s) {
            longMxN matrix = a.TempCopy();
            longComp.bitwiseXorInpl(matrix, s);
            return matrix;
        }

        public static longMxN operator ^(in long s, in longMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator <<(in longMxN a, int shift) {
            longMxN matrix = a.TempCopy();
            longComp.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator >>(in longMxN a, int shift) {
            longMxN matrix = a.TempCopy();
            longComp.bitwiseRightShiftInpl(matrix, shift);
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

            longComp.addInpl(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

            return matrix;
        }

        /// <summary>Component-wise subtraction; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator -(in longMxN lhs, in longMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            longMxN matrix = lhs.TempCopy();

            longComp.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>Component-wise multiplication; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator *(in longMxN lhs, in longMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            longMxN matrix = lhs.TempCopy();

            longComp.mulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>Component-wise division; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator /(in longMxN dividend, in longMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            longMxN newDividendMatrix = dividend.TempCopy();

            longComp.divInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>Component-wise modulo; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator %(in longMxN dividend, in longMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            longComp.modInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator &(in longMxN a, in longMxN b) {
            
            Assume.SameDim(in a, in b);

            longMxN matrix = a.TempCopy();
            longComp.bitwiseAndInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator |(in longMxN a, in longMxN b) {

            Assume.SameDim(in a, in b);

            longMxN matrix = a.TempCopy();
            longComp.bitwiseOrInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator ^(in longMxN a, in longMxN b) {

            Assume.SameDim(in a, in b);

            longMxN matrix = a.TempCopy();
            longComp.bitwiseXorInpl(matrix, b);
            return matrix;
        }

        #endregion
    }
}