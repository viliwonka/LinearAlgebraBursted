using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct longMxN {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator +(in longMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator -(in longMxN a)
        {
            longMxN matrix = a.TempCopy();
            
            longOP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator +(in longMxN lhs, long rhs)
        {
            longMxN matrix = lhs.TempCopy();
            
            longOP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator +(long lhs, in longMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator -(in longMxN lhs, long rhs)
        {
            longMxN matrix = lhs.TempCopy();
            
            longOP.addInpl(matrix, (long)(-rhs));

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator -(long lhs, in longMxN rhs)
        {
            return rhs - lhs;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator *(in longMxN a, long s)
        {
            longMxN matrix = a.TempCopy();

            longOP.mulInpl(matrix, s);

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

            longOP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator /(long s, in longMxN a)
        {
            longMxN matrix = a.TempCopy();
            
            if (s == 0f)
                throw new DivideByZeroException();

            longOP.divInpl(s, matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator %(in longMxN a, long s)
        {
            longMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            longOP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator %(long s, in longMxN a)
        {
            longMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            longOP.modInpl(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator ~(in longMxN a) {

            longMxN matrix = a.TempCopy();

            longOP.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator &(in longMxN a, in long s) {

            longMxN matrix = a.TempCopy();
            longOP.bitwiseAndInpl(matrix, s);
            return matrix;
        }

        public static longMxN operator &(in long s, in longMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator |(in longMxN a, in long s) {
            longMxN matrix = a.TempCopy();
            longOP.bitwiseOrInpl(matrix, s);
            return matrix;
        }

        public static longMxN operator |(in long s, in longMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator ^(in longMxN a, in long s) {
            longMxN matrix = a.TempCopy();
            longOP.bitwiseXorInpl(matrix, s);
            return matrix;
        }

        public static longMxN operator ^(in long s, in longMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator <<(in longMxN a, int shift) {
            longMxN matrix = a.TempCopy();
            longOP.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator >>(in longMxN a, int shift) {
            longMxN matrix = a.TempCopy();
            longOP.bitwiseRightShiftInpl(matrix, shift);
            return matrix;
        }

        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>
        /// Component-wise addition
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator +(in longMxN lhs, in longMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            longMxN matrix = lhs.TempCopy();

            longOP.addInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>
        /// Component-wise subtraction
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator -(in longMxN lhs, in longMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            longMxN matrix = lhs.TempCopy();

            longOP.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>
        /// Component-wise multiplication
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator *(in longMxN lhs, in longMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            longMxN matrix = lhs.TempCopy();

            longOP.compMulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>
        /// Component-wise division
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator /(in longMxN dividend, in longMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            longMxN newDividendMatrix = dividend.TempCopy();

            longOP.compDivInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>
        /// Component-wise modulo
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator %(in longMxN dividend, in longMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            longOP.compModDiv(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator &(in longMxN a, in longMxN b) {
            
            Assume.SameDim(in a, in b);

            longMxN matrix = a.TempCopy();
            longOP.bitwiseAndInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator |(in longMxN a, in longMxN b) {

            Assume.SameDim(in a, in b);

            longMxN matrix = a.TempCopy();
            longOP.bitwiseOrInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN operator ^(in longMxN a, in longMxN b) {

            Assume.SameDim(in a, in b);

            longMxN matrix = a.TempCopy();
            longOP.bitwiseXorInpl(matrix, b);
            return matrix;
        }

        #endregion
    }
}