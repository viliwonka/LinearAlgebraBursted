using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct fProxyMxN : IDisposable, IUnsafefProxyArray {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator +(in fProxyMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator -(in fProxyMxN a)
        {
            fProxyMxN matrix = a.TempCopy();
            
            fProxyOP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator +(in fProxyMxN lhs, fProxy rhs)
        {
            fProxyMxN matrix = lhs.TempCopy();
            
            fProxyOP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator +(fProxy lhs, in fProxyMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator -(in fProxyMxN lhs, fProxy rhs)
        {
            fProxyMxN matrix = lhs.TempCopy();
            
            fProxyOP.addInpl(matrix, -rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator -(fProxy lhs, in fProxyMxN rhs)
        {
            return rhs - lhs;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator *(in fProxyMxN a, fProxy s)
        {
            fProxyMxN matrix = a.TempCopy();

            fProxyOP.mulInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator *(fProxy lhs, in fProxyMxN rhs) => rhs * lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator /(in fProxyMxN a, fProxy s)
        {
            fProxyMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            fProxyOP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator /(fProxy s, in fProxyMxN a)
        {
            fProxyMxN matrix = a.TempCopy();
            
            if (s == 0f)
                throw new DivideByZeroException();

            fProxyOP.divInpl(s, matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator %(in fProxyMxN a, fProxy s)
        {
            fProxyMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            fProxyOP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator %(fProxy s, in fProxyMxN a)
        {
            fProxyMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            fProxyOP.modInpl(s, matrix);

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
        public static fProxyMxN operator +(in fProxyMxN lhs, in fProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            fProxyMxN matrix = lhs.TempCopy();

            fProxyOP.addInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>
        /// Component-wise subtraction
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator -(in fProxyMxN lhs, in fProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            fProxyMxN matrix = lhs.TempCopy();

            fProxyOP.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>
        /// Component-wise multiplication
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator *(in fProxyMxN lhs, in fProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            fProxyMxN matrix = lhs.TempCopy();

            fProxyOP.compMulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>
        /// Component-wise division
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator /(in fProxyMxN dividend, in fProxyMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            fProxyMxN newDividendMatrix = dividend.TempCopy();

            fProxyOP.compDivInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>
        /// Component-wise modulo
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator %(in fProxyMxN dividend, in fProxyMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            fProxyOP.compModDiv(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        #endregion
    }
}