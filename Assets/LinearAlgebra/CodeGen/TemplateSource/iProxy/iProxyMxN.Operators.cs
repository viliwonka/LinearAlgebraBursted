using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct iProxyMxN {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator +(in iProxyMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator -(in iProxyMxN a)
        {
            iProxyMxN matrix = a.TempCopy();
            
            iProxyOP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator +(in iProxyMxN lhs, iProxy rhs)
        {
            iProxyMxN matrix = lhs.TempCopy();
            
            iProxyOP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator +(iProxy lhs, in iProxyMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator -(in iProxyMxN lhs, iProxy rhs)
        {
            iProxyMxN matrix = lhs.TempCopy();
            
            iProxyOP.addInpl(matrix, (iProxy)(-rhs));

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator -(iProxy lhs, in iProxyMxN rhs)
        {
            return rhs - lhs;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator *(in iProxyMxN a, iProxy s)
        {
            iProxyMxN matrix = a.TempCopy();

            iProxyOP.mulInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator *(iProxy lhs, in iProxyMxN rhs) => rhs * lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator /(in iProxyMxN a, iProxy s)
        {
            iProxyMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            iProxyOP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator /(iProxy s, in iProxyMxN a)
        {
            iProxyMxN matrix = a.TempCopy();
            
            if (s == 0f)
                throw new DivideByZeroException();

            iProxyOP.divInpl(s, matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator %(in iProxyMxN a, iProxy s)
        {
            iProxyMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            iProxyOP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator %(iProxy s, in iProxyMxN a)
        {
            iProxyMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            iProxyOP.modInpl(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator ~(in iProxyMxN a) {

            iProxyMxN matrix = a.TempCopy();

            iProxyOP.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator &(in iProxyMxN a, in iProxy s) {

            iProxyMxN matrix = a.TempCopy();
            iProxyOP.bitwiseAndInpl(matrix, s);
            return matrix;
        }

        public static iProxyMxN operator &(in iProxy s, in iProxyMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator |(in iProxyMxN a, in iProxy s) {
            iProxyMxN matrix = a.TempCopy();
            iProxyOP.bitwiseOrInpl(matrix, s);
            return matrix;
        }

        public static iProxyMxN operator |(in iProxy s, in iProxyMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator ^(in iProxyMxN a, in iProxy s) {
            iProxyMxN matrix = a.TempCopy();
            iProxyOP.bitwiseXorInpl(matrix, s);
            return matrix;
        }

        public static iProxyMxN operator ^(in iProxy s, in iProxyMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator <<(in iProxyMxN a, int shift) {
            iProxyMxN matrix = a.TempCopy();
            iProxyOP.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator >>(in iProxyMxN a, int shift) {
            iProxyMxN matrix = a.TempCopy();
            iProxyOP.bitwiseRightShiftInpl(matrix, shift);
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
        public static iProxyMxN operator +(in iProxyMxN lhs, in iProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            iProxyMxN matrix = lhs.TempCopy();

            iProxyOP.addInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>
        /// Component-wise subtraction
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator -(in iProxyMxN lhs, in iProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            iProxyMxN matrix = lhs.TempCopy();

            iProxyOP.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>
        /// Component-wise multiplication
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator *(in iProxyMxN lhs, in iProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            iProxyMxN matrix = lhs.TempCopy();

            iProxyOP.compMulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>
        /// Component-wise division
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator /(in iProxyMxN dividend, in iProxyMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            iProxyMxN newDividendMatrix = dividend.TempCopy();

            iProxyOP.compDivInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>
        /// Component-wise modulo
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator %(in iProxyMxN dividend, in iProxyMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            iProxyOP.compModDiv(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator &(in iProxyMxN a, in iProxyMxN b) {
            
            Assume.SameDim(in a, in b);

            iProxyMxN matrix = a.TempCopy();
            iProxyOP.bitwiseAndInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator |(in iProxyMxN a, in iProxyMxN b) {

            Assume.SameDim(in a, in b);

            iProxyMxN matrix = a.TempCopy();
            iProxyOP.bitwiseOrInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator ^(in iProxyMxN a, in iProxyMxN b) {

            Assume.SameDim(in a, in b);

            iProxyMxN matrix = a.TempCopy();
            iProxyOP.bitwiseXorInpl(matrix, b);
            return matrix;
        }

        #endregion
    }
}