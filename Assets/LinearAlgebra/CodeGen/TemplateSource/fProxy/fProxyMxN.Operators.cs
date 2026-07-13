using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct fProxyMxN : IDisposable, IUnsafefProxyArray {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator +(in fProxyMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator -(in fProxyMxN a)
        {
            fProxyMxN matrix = a.TempCopy();
            
            fProxyComp.signFlipInPlace(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator +(in fProxyMxN lhs, fProxy rhs)
        {
            fProxyMxN matrix = lhs.TempCopy();
            
            fProxyComp.addInPlace(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator +(fProxy lhs, in fProxyMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator -(in fProxyMxN lhs, fProxy rhs)
        {
            fProxyMxN matrix = lhs.TempCopy();
            
            fProxyComp.addInPlace(matrix, -rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator -(fProxy lhs, in fProxyMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            fProxyMxN matrix = rhs.TempCopy();
            fProxyComp.subInPlace(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator *(in fProxyMxN a, fProxy s)
        {
            fProxyMxN matrix = a.TempCopy();

            fProxyComp.mulInPlace(matrix, s);

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

            fProxyComp.divInPlace(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator /(fProxy s, in fProxyMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); division by zero MATRIX entries is left to IEEE (Inf/NaN).
            fProxyMxN matrix = a.TempCopy();
            fProxyComp.divInPlace(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator %(in fProxyMxN a, fProxy s)
        {
            fProxyMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            fProxyComp.modInPlace(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator %(fProxy s, in fProxyMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry is left to IEEE / runtime semantics.
            fProxyMxN matrix = a.TempCopy();
            fProxyComp.modInPlace(s, matrix);

            return matrix;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator +(in fProxyMxN lhs, in fProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            fProxyMxN matrix = lhs.TempCopy();

            fProxyComp.addInPlace(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

            return matrix;
        }

        /// <summary>Component-wise subtraction; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator -(in fProxyMxN lhs, in fProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            fProxyMxN matrix = lhs.TempCopy();

            fProxyComp.subInPlace(matrix, rhs);

            return matrix;
        }

        /// <summary>Component-wise multiplication; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator *(in fProxyMxN lhs, in fProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            fProxyMxN matrix = lhs.TempCopy();

            fProxyComp.mulInPlace(matrix, rhs);

            return matrix;
        }

        /// <summary>Component-wise division; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator /(in fProxyMxN dividend, in fProxyMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            fProxyMxN newDividendMatrix = dividend.TempCopy();

            fProxyComp.divInPlace(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>Component-wise modulo; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN operator %(in fProxyMxN dividend, in fProxyMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            fProxyComp.modInPlace(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        #endregion
    }
}