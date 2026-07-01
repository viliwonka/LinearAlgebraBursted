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
            
            shortElem_OP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator +(in shortMxN lhs, short rhs)
        {
            shortMxN matrix = lhs.TempCopy();
            
            shortElem_OP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator +(short lhs, in shortMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator -(in shortMxN lhs, short rhs)
        {
            shortMxN matrix = lhs.TempCopy();
            
            shortElem_OP.addInpl(matrix, (short)(-rhs));

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator -(short lhs, in shortMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            shortMxN matrix = rhs.TempCopy();
            shortElem_OP.subInpl(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator *(in shortMxN a, short s)
        {
            shortMxN matrix = a.TempCopy();

            shortElem_OP.mulInpl(matrix, s);

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

            shortElem_OP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator /(short s, in shortMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer div by zero).
            shortMxN matrix = a.TempCopy();
            shortElem_OP.divInpl(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator %(in shortMxN a, short s)
        {
            shortMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            shortElem_OP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator %(short s, in shortMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer mod by zero).
            shortMxN matrix = a.TempCopy();
            shortElem_OP.modInpl(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator ~(in shortMxN a) {

            shortMxN matrix = a.TempCopy();

            shortElem_OP.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator &(in shortMxN a, in short s) {

            shortMxN matrix = a.TempCopy();
            shortElem_OP.bitwiseAndInpl(matrix, s);
            return matrix;
        }

        public static shortMxN operator &(in short s, in shortMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator |(in shortMxN a, in short s) {
            shortMxN matrix = a.TempCopy();
            shortElem_OP.bitwiseOrInpl(matrix, s);
            return matrix;
        }

        public static shortMxN operator |(in short s, in shortMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator ^(in shortMxN a, in short s) {
            shortMxN matrix = a.TempCopy();
            shortElem_OP.bitwiseXorInpl(matrix, s);
            return matrix;
        }

        public static shortMxN operator ^(in short s, in shortMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator <<(in shortMxN a, int shift) {
            shortMxN matrix = a.TempCopy();
            shortElem_OP.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator >>(in shortMxN a, int shift) {
            shortMxN matrix = a.TempCopy();
            shortElem_OP.bitwiseRightShiftInpl(matrix, shift);
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

            shortElem_OP.addInpl(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

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

            shortElem_OP.subInpl(matrix, rhs);

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

            shortElem_OP.mulInpl(rhs, matrix);

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

            shortElem_OP.divInpl(newDividendMatrix, divisor);
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

            shortElem_OP.modInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator &(in shortMxN a, in shortMxN b) {
            
            Assume.SameDim(in a, in b);

            shortMxN matrix = a.TempCopy();
            shortElem_OP.bitwiseAndInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator |(in shortMxN a, in shortMxN b) {

            Assume.SameDim(in a, in b);

            shortMxN matrix = a.TempCopy();
            shortElem_OP.bitwiseOrInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator ^(in shortMxN a, in shortMxN b) {

            Assume.SameDim(in a, in b);

            shortMxN matrix = a.TempCopy();
            shortElem_OP.bitwiseXorInpl(matrix, b);
            return matrix;
        }

        #endregion
    }
}