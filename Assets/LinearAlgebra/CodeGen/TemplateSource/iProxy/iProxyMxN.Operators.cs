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
            
            iProxy_OP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator +(in iProxyMxN lhs, iProxy rhs)
        {
            iProxyMxN matrix = lhs.TempCopy();
            
            iProxy_OP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator +(iProxy lhs, in iProxyMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator -(in iProxyMxN lhs, iProxy rhs)
        {
            iProxyMxN matrix = lhs.TempCopy();
            
            iProxy_OP.addInpl(matrix, (iProxy)(-rhs));

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator -(iProxy lhs, in iProxyMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            iProxyMxN matrix = rhs.TempCopy();
            iProxy_OP.subInpl(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator *(in iProxyMxN a, iProxy s)
        {
            iProxyMxN matrix = a.TempCopy();

            iProxy_OP.mulInpl(matrix, s);

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

            iProxy_OP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator /(iProxy s, in iProxyMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer div by zero).
            iProxyMxN matrix = a.TempCopy();
            iProxy_OP.divInpl(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator %(in iProxyMxN a, iProxy s)
        {
            iProxyMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            iProxy_OP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator %(iProxy s, in iProxyMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer mod by zero).
            iProxyMxN matrix = a.TempCopy();
            iProxy_OP.modInpl(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator ~(in iProxyMxN a) {

            iProxyMxN matrix = a.TempCopy();

            iProxy_OP.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator &(in iProxyMxN a, in iProxy s) {

            iProxyMxN matrix = a.TempCopy();
            iProxy_OP.bitwiseAndInpl(matrix, s);
            return matrix;
        }

        public static iProxyMxN operator &(in iProxy s, in iProxyMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator |(in iProxyMxN a, in iProxy s) {
            iProxyMxN matrix = a.TempCopy();
            iProxy_OP.bitwiseOrInpl(matrix, s);
            return matrix;
        }

        public static iProxyMxN operator |(in iProxy s, in iProxyMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator ^(in iProxyMxN a, in iProxy s) {
            iProxyMxN matrix = a.TempCopy();
            iProxy_OP.bitwiseXorInpl(matrix, s);
            return matrix;
        }

        public static iProxyMxN operator ^(in iProxy s, in iProxyMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator <<(in iProxyMxN a, int shift) {
            iProxyMxN matrix = a.TempCopy();
            iProxy_OP.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator >>(in iProxyMxN a, int shift) {
            iProxyMxN matrix = a.TempCopy();
            iProxy_OP.bitwiseRightShiftInpl(matrix, shift);
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

            iProxy_OP.addInpl(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

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

            iProxy_OP.subInpl(matrix, rhs);

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

            iProxy_OP.compMulInpl(rhs, matrix);

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

            iProxy_OP.compDivInpl(newDividendMatrix, divisor);
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

            iProxy_OP.compModDiv(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator &(in iProxyMxN a, in iProxyMxN b) {
            
            Assume.SameDim(in a, in b);

            iProxyMxN matrix = a.TempCopy();
            iProxy_OP.bitwiseAndInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator |(in iProxyMxN a, in iProxyMxN b) {

            Assume.SameDim(in a, in b);

            iProxyMxN matrix = a.TempCopy();
            iProxy_OP.bitwiseOrInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator ^(in iProxyMxN a, in iProxyMxN b) {

            Assume.SameDim(in a, in b);

            iProxyMxN matrix = a.TempCopy();
            iProxy_OP.bitwiseXorInpl(matrix, b);
            return matrix;
        }

        #endregion
    }
}