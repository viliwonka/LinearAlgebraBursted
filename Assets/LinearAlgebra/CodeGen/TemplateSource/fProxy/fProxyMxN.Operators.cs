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
            
            fProxyElem_OP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator +(in fProxyMxN lhs, fProxy rhs)
        {
            fProxyMxN matrix = lhs.TempCopy();
            
            fProxyElem_OP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator +(fProxy lhs, in fProxyMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator -(in fProxyMxN lhs, fProxy rhs)
        {
            fProxyMxN matrix = lhs.TempCopy();
            
            fProxyElem_OP.addInpl(matrix, -rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator -(fProxy lhs, in fProxyMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            fProxyMxN matrix = rhs.TempCopy();
            fProxyElem_OP.subInpl(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator *(in fProxyMxN a, fProxy s)
        {
            fProxyMxN matrix = a.TempCopy();

            fProxyElem_OP.mulInpl(matrix, s);

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

            fProxyElem_OP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator /(fProxy s, in fProxyMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); division by zero MATRIX entries is left to IEEE (Inf/NaN).
            fProxyMxN matrix = a.TempCopy();
            fProxyElem_OP.divInpl(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator %(in fProxyMxN a, fProxy s)
        {
            fProxyMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            fProxyElem_OP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator %(fProxy s, in fProxyMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry is left to IEEE / runtime semantics.
            fProxyMxN matrix = a.TempCopy();
            fProxyElem_OP.modInpl(s, matrix);

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

            fProxyElem_OP.addInpl(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

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

            fProxyElem_OP.subInpl(matrix, rhs);

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

            fProxyElem_OP.mulInpl(rhs, matrix);

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

            fProxyElem_OP.divInpl(newDividendMatrix, divisor);
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

            fProxyElem_OP.modInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        #endregion
    }
}