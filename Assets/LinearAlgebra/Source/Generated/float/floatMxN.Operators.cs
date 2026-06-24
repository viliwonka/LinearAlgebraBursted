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
            
            floatOP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator +(in floatMxN lhs, float rhs)
        {
            floatMxN matrix = lhs.TempCopy();
            
            floatOP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator +(float lhs, in floatMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator -(in floatMxN lhs, float rhs)
        {
            floatMxN matrix = lhs.TempCopy();
            
            floatOP.addInpl(matrix, -rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator -(float lhs, in floatMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            floatMxN matrix = rhs.TempCopy();
            floatOP.subInpl(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator *(in floatMxN a, float s)
        {
            floatMxN matrix = a.TempCopy();

            floatOP.mulInpl(matrix, s);

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

            floatOP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator /(float s, in floatMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); division by zero MATRIX entries is left to IEEE (Inf/NaN).
            floatMxN matrix = a.TempCopy();
            floatOP.divInpl(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator %(in floatMxN a, float s)
        {
            floatMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            floatOP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator %(float s, in floatMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry is left to IEEE / runtime semantics.
            floatMxN matrix = a.TempCopy();
            floatOP.modInpl(s, matrix);

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

            floatOP.addInpl(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

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

            floatOP.subInpl(matrix, rhs);

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

            floatOP.compMulInpl(rhs, matrix);

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

            floatOP.compDivInpl(newDividendMatrix, divisor);
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

            floatOP.compModDiv(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        #endregion
    }
}