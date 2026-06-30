using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct floatMxN : IDisposable, IUnsafefloatArray {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator +(in floatMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator -(in floatMxN a)
        {
            floatMxN matrix = a.TempCopy();
            
            float_OP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator +(in floatMxN lhs, float rhs)
        {
            floatMxN matrix = lhs.TempCopy();
            
            float_OP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator +(float lhs, in floatMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator -(in floatMxN lhs, float rhs)
        {
            floatMxN matrix = lhs.TempCopy();
            
            float_OP.addInpl(matrix, -rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator -(float lhs, in floatMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            floatMxN matrix = rhs.TempCopy();
            float_OP.subInpl(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator *(in floatMxN a, float s)
        {
            floatMxN matrix = a.TempCopy();

            float_OP.mulInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator *(float lhs, in floatMxN rhs) => rhs * lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator /(in floatMxN a, float s)
        {
            floatMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            float_OP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator /(float s, in floatMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); division by zero MATRIX entries is left to IEEE (Inf/NaN).
            floatMxN matrix = a.TempCopy();
            float_OP.divInpl(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator %(in floatMxN a, float s)
        {
            floatMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            float_OP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator %(float s, in floatMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry is left to IEEE / runtime semantics.
            floatMxN matrix = a.TempCopy();
            float_OP.modInpl(s, matrix);

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
        public static floatMxN operator +(in floatMxN lhs, in floatMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            floatMxN matrix = lhs.TempCopy();

            float_OP.addInpl(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

            return matrix;
        }

        /// <summary>
        /// Component-wise subtraction
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator -(in floatMxN lhs, in floatMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            floatMxN matrix = lhs.TempCopy();

            float_OP.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>
        /// Component-wise multiplication
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator *(in floatMxN lhs, in floatMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            floatMxN matrix = lhs.TempCopy();

            float_OP.compMulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>
        /// Component-wise division
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator /(in floatMxN dividend, in floatMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            floatMxN newDividendMatrix = dividend.TempCopy();

            float_OP.compDivInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>
        /// Component-wise modulo
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator %(in floatMxN dividend, in floatMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            float_OP.compModDiv(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        #endregion
    }
}