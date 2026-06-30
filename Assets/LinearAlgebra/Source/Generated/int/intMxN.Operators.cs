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
            
            int_OP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator +(in intMxN lhs, int rhs)
        {
            intMxN matrix = lhs.TempCopy();
            
            int_OP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator +(int lhs, in intMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator -(in intMxN lhs, int rhs)
        {
            intMxN matrix = lhs.TempCopy();
            
            int_OP.addInpl(matrix, (int)(-rhs));

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator -(int lhs, in intMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            intMxN matrix = rhs.TempCopy();
            int_OP.subInpl(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator *(in intMxN a, int s)
        {
            intMxN matrix = a.TempCopy();

            int_OP.mulInpl(matrix, s);

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

            int_OP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator /(int s, in intMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer div by zero).
            intMxN matrix = a.TempCopy();
            int_OP.divInpl(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator %(in intMxN a, int s)
        {
            intMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            int_OP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator %(int s, in intMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer mod by zero).
            intMxN matrix = a.TempCopy();
            int_OP.modInpl(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator ~(in intMxN a) {

            intMxN matrix = a.TempCopy();

            int_OP.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator &(in intMxN a, in int s) {

            intMxN matrix = a.TempCopy();
            int_OP.bitwiseAndInpl(matrix, s);
            return matrix;
        }

        public static intMxN operator &(in int s, in intMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator |(in intMxN a, in int s) {
            intMxN matrix = a.TempCopy();
            int_OP.bitwiseOrInpl(matrix, s);
            return matrix;
        }

        public static intMxN operator |(in int s, in intMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator ^(in intMxN a, in int s) {
            intMxN matrix = a.TempCopy();
            int_OP.bitwiseXorInpl(matrix, s);
            return matrix;
        }

        public static intMxN operator ^(in int s, in intMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator <<(in intMxN a, int shift) {
            intMxN matrix = a.TempCopy();
            int_OP.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator >>(in intMxN a, int shift) {
            intMxN matrix = a.TempCopy();
            int_OP.bitwiseRightShiftInpl(matrix, shift);
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

            int_OP.addInpl(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

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

            int_OP.subInpl(matrix, rhs);

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

            int_OP.compMulInpl(rhs, matrix);

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

            int_OP.compDivInpl(newDividendMatrix, divisor);
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

            int_OP.compModDiv(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator &(in intMxN a, in intMxN b) {
            
            Assume.SameDim(in a, in b);

            intMxN matrix = a.TempCopy();
            int_OP.bitwiseAndInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator |(in intMxN a, in intMxN b) {

            Assume.SameDim(in a, in b);

            intMxN matrix = a.TempCopy();
            int_OP.bitwiseOrInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator ^(in intMxN a, in intMxN b) {

            Assume.SameDim(in a, in b);

            intMxN matrix = a.TempCopy();
            int_OP.bitwiseXorInpl(matrix, b);
            return matrix;
        }

        #endregion
    }
}