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
            fProxyComp.signFlipInPlace(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator +(in fProxyN a, fProxy s) {

            fProxyN vec = a.TempCopy();
            fProxyComp.addInPlace(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator +(fProxy s, in fProxyN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator -(in fProxyN a, fProxy s) {
            
            fProxyN vec = a.TempCopy();
            fProxyComp.addInPlace(vec, -s);
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator -(fProxy s, in fProxyN a)
        {
            fProxyN vec = a.TempCopy();
            fProxyComp.subInPlace(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator *(in fProxyN a, fProxy s) {
            
            fProxyN vec = a.TempCopy();

            fProxyComp.mulInPlace(vec, s);

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

            fProxyComp.divInPlace(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator /(fProxy s, fProxyN a)
        {
            fProxyN vec = a.TempCopy();

            fProxyComp.divInPlace(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator %(in fProxyN a, fProxy s)
        {
            fProxyN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            fProxyComp.modInPlace(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator %(fProxy s, fProxyN a)
        {
            fProxyN vec = a.TempCopy();

            fProxyComp.modInPlace(s, vec);

            return vec;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator +(in fProxyN a, in fProxyN b) {

            Assume.SameDim(in a, in b);

            fProxyN vec = a.TempCopy();

            fProxyComp.addInPlace(vec, b);   // vec += b  (vec is the copy of a)

            return vec;
        }

        /// <summary>Component-wise subtraction; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator -(in fProxyN a, in fProxyN b) {

            Assume.SameDim(in a, in b);

            fProxyN vec = a.TempCopy();
            fProxyComp.subInPlace(vec, b);
            
            return vec;
        }

        /// <summary>Component-wise multiplication; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator *(in fProxyN a, in fProxyN b) {

            Assume.SameDim(in a, in b);

            fProxyN vec = a.TempCopy();

            fProxyComp.mulInPlace(vec, b);

            return vec;
        }

        /// <summary>Component-wise division (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator /(in fProxyN dividend, in fProxyN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            fProxyN newDividendVec = dividend.TempCopy();
            fProxyComp.divInPlace(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>Component-wise modulo (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator %(in fProxyN dividend, in fProxyN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            fProxyN newDividendVec = dividend.TempCopy();
            fProxyComp.modInPlace(newDividendVec, divisor);

            return newDividendVec;
        }

        #endregion

    }
}