using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct fProxyN {

        #region SCALAR OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator +(in fProxyN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator -(in fProxyN a) {

            fProxyN vec = a.TempCopy();
            fProxyOP.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator +(in fProxyN a, fProxy s) {

            fProxyN vec = a.TempCopy();
            fProxyOP.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator +(fProxy s, in fProxyN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator -(in fProxyN a, fProxy s) {
            
            fProxyN vec = a.TempCopy();
            fProxyOP.addInpl(vec, -s);
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator -(fProxy s, in fProxyN a)
        {
            fProxyN vec = a.TempCopy();
            fProxyOP.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator *(in fProxyN a, fProxy s) {
            
            fProxyN vec = a.TempCopy();

            fProxyOP.mulInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator *(fProxy s, in fProxyN a) => a * s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator /(in fProxyN a, fProxy s)
        {
            fProxyN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            fProxyOP.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator /(fProxy s, fProxyN a)
        {
            fProxyN vec = a.TempCopy();

            fProxyOP.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator %(in fProxyN a, fProxy s)
        {
            fProxyN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            fProxyOP.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator %(fProxy s, fProxyN a)
        {
            fProxyN vec = a.TempCopy();

            fProxyOP.modInpl(s, vec);

            return vec;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>
        /// Component-wise addition
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator +(in fProxyN a, in fProxyN b) {

            Assume.SameDim(in a, in b);

            fProxyN vec = a.TempCopy();

            fProxyOP.addInpl(b, vec);

            return vec;
        }

        /// <summary>
        /// Component-wise subtraction
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator -(in fProxyN a, in fProxyN b) {

            Assume.SameDim(in a, in b);

            fProxyN vec = a.TempCopy();
            fProxyOP.subInpl(vec, b);
            
            return vec;
        }

        /// <summary>
        /// Component-wise multiplication
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator *(in fProxyN a, in fProxyN b) {

            Assume.SameDim(in a, in b);

            fProxyN vec = a.TempCopy();

            fProxyOP.compMulInpl(b, vec);

            return vec;
        }

        /// <summary>
        /// Component-wise division
        /// Vectors have to be same dimensions
        /// Dividend / divisor
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator /(in fProxyN dividend, in fProxyN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            fProxyN newDividendVec = dividend.TempCopy();
            fProxyOP.compDivInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>
        /// Component-wise modulo
        /// Vectors have to be same dimensions
        /// Dividend / divisor
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator %(in fProxyN dividend, in fProxyN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            fProxyN newDividendVec = dividend.TempCopy();
            fProxyOP.compModDiv(newDividendVec, divisor);

            return newDividendVec;
        }

        #endregion

    }
}