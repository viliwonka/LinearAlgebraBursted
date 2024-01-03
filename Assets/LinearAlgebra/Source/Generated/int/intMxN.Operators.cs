using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct intMxN {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator +(in intMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator -(in intMxN a)
        {
            intMxN matrix = a.TempCopy();
            
            intOP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator +(in intMxN lhs, int rhs)
        {
            intMxN matrix = lhs.TempCopy();
            
            intOP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator +(int lhs, in intMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator -(in intMxN lhs, int rhs)
        {
            intMxN matrix = lhs.TempCopy();
            
            intOP.addInpl(matrix, (int)(-rhs));

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator -(int lhs, in intMxN rhs)
        {
            return rhs - lhs;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator *(in intMxN a, int s)
        {
            intMxN matrix = a.TempCopy();

            intOP.mulInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator *(int lhs, in intMxN rhs) => rhs * lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator /(in intMxN a, int s)
        {
            intMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            intOP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator /(int s, in intMxN a)
        {
            intMxN matrix = a.TempCopy();
            
            if (s == 0f)
                throw new DivideByZeroException();

            intOP.divInpl(s, matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator %(in intMxN a, int s)
        {
            intMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            intOP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator %(int s, in intMxN a)
        {
            intMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            intOP.modInpl(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator ~(in intMxN a) {

            intMxN matrix = a.TempCopy();

            intOP.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator &(in intMxN a, in int s) {

            intMxN matrix = a.TempCopy();
            intOP.bitwiseAndInpl(matrix, s);
            return matrix;
        }

        public static intMxN operator &(in int s, in intMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator |(in intMxN a, in int s) {
            intMxN matrix = a.TempCopy();
            intOP.bitwiseOrInpl(matrix, s);
            return matrix;
        }

        public static intMxN operator |(in int s, in intMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator ^(in intMxN a, in int s) {
            intMxN matrix = a.TempCopy();
            intOP.bitwiseXorInpl(matrix, s);
            return matrix;
        }

        public static intMxN operator ^(in int s, in intMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator <<(in intMxN a, int shift) {
            intMxN matrix = a.TempCopy();
            intOP.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator >>(in intMxN a, int shift) {
            intMxN matrix = a.TempCopy();
            intOP.bitwiseRightShiftInpl(matrix, shift);
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
        public static intMxN operator +(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            intMxN matrix = lhs.TempCopy();

            intOP.addInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>
        /// Component-wise subtraction
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator -(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            intMxN matrix = lhs.TempCopy();

            intOP.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>
        /// Component-wise multiplication
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator *(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            intMxN matrix = lhs.TempCopy();

            intOP.compMulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>
        /// Component-wise division
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator /(in intMxN dividend, in intMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            intMxN newDividendMatrix = dividend.TempCopy();

            intOP.compDivInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>
        /// Component-wise modulo
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator %(in intMxN dividend, in intMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            intOP.compModDiv(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator &(in intMxN a, in intMxN b) {
            
            Assume.SameDim(in a, in b);

            intMxN matrix = a.TempCopy();
            intOP.bitwiseAndInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator |(in intMxN a, in intMxN b) {

            Assume.SameDim(in a, in b);

            intMxN matrix = a.TempCopy();
            intOP.bitwiseOrInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator ^(in intMxN a, in intMxN b) {

            Assume.SameDim(in a, in b);

            intMxN matrix = a.TempCopy();
            intOP.bitwiseXorInpl(matrix, b);
            return matrix;
        }

        #endregion
    }
}