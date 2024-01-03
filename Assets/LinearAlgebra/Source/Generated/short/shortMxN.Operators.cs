using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct shortMxN {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator +(in shortMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator -(in shortMxN a)
        {
            shortMxN matrix = a.TempCopy();
            
            shortOP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator +(in shortMxN lhs, short rhs)
        {
            shortMxN matrix = lhs.TempCopy();
            
            shortOP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator +(short lhs, in shortMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator -(in shortMxN lhs, short rhs)
        {
            shortMxN matrix = lhs.TempCopy();
            
            shortOP.addInpl(matrix, (short)(-rhs));

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator -(short lhs, in shortMxN rhs)
        {
            return rhs - lhs;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator *(in shortMxN a, short s)
        {
            shortMxN matrix = a.TempCopy();

            shortOP.mulInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator *(short lhs, in shortMxN rhs) => rhs * lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator /(in shortMxN a, short s)
        {
            shortMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            shortOP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator /(short s, in shortMxN a)
        {
            shortMxN matrix = a.TempCopy();
            
            if (s == 0f)
                throw new DivideByZeroException();

            shortOP.divInpl(s, matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator %(in shortMxN a, short s)
        {
            shortMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            shortOP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator %(short s, in shortMxN a)
        {
            shortMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            shortOP.modInpl(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator ~(in shortMxN a) {

            shortMxN matrix = a.TempCopy();

            shortOP.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator &(in shortMxN a, in short s) {

            shortMxN matrix = a.TempCopy();
            shortOP.bitwiseAndInpl(matrix, s);
            return matrix;
        }

        public static shortMxN operator &(in short s, in shortMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator |(in shortMxN a, in short s) {
            shortMxN matrix = a.TempCopy();
            shortOP.bitwiseOrInpl(matrix, s);
            return matrix;
        }

        public static shortMxN operator |(in short s, in shortMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator ^(in shortMxN a, in short s) {
            shortMxN matrix = a.TempCopy();
            shortOP.bitwiseXorInpl(matrix, s);
            return matrix;
        }

        public static shortMxN operator ^(in short s, in shortMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator <<(in shortMxN a, int shift) {
            shortMxN matrix = a.TempCopy();
            shortOP.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator >>(in shortMxN a, int shift) {
            shortMxN matrix = a.TempCopy();
            shortOP.bitwiseRightShiftInpl(matrix, shift);
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
        public static shortMxN operator +(in shortMxN lhs, in shortMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            shortMxN matrix = lhs.TempCopy();

            shortOP.addInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>
        /// Component-wise subtraction
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator -(in shortMxN lhs, in shortMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            shortMxN matrix = lhs.TempCopy();

            shortOP.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>
        /// Component-wise multiplication
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator *(in shortMxN lhs, in shortMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            shortMxN matrix = lhs.TempCopy();

            shortOP.compMulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>
        /// Component-wise division
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator /(in shortMxN dividend, in shortMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            shortMxN newDividendMatrix = dividend.TempCopy();

            shortOP.compDivInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>
        /// Component-wise modulo
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator %(in shortMxN dividend, in shortMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            shortOP.compModDiv(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator &(in shortMxN a, in shortMxN b) {
            
            Assume.SameDim(in a, in b);

            shortMxN matrix = a.TempCopy();
            shortOP.bitwiseAndInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator |(in shortMxN a, in shortMxN b) {

            Assume.SameDim(in a, in b);

            shortMxN matrix = a.TempCopy();
            shortOP.bitwiseOrInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator ^(in shortMxN a, in shortMxN b) {

            Assume.SameDim(in a, in b);

            shortMxN matrix = a.TempCopy();
            shortOP.bitwiseXorInpl(matrix, b);
            return matrix;
        }

        #endregion
    }
}