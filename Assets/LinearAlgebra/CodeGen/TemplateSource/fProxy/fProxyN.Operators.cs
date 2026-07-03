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
            fProxyComp.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator +(in fProxyN a, fProxy s) {

            fProxyN vec = a.TempCopy();
            fProxyComp.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator +(fProxy s, in fProxyN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator -(in fProxyN a, fProxy s) {
            
            fProxyN vec = a.TempCopy();
            fProxyComp.addInpl(vec, -s);
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator -(fProxy s, in fProxyN a)
        {
            fProxyN vec = a.TempCopy();
            fProxyComp.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator *(in fProxyN a, fProxy s) {
            
            fProxyN vec = a.TempCopy();

            fProxyComp.mulInpl(vec, s);

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

            fProxyComp.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator /(fProxy s, fProxyN a)
        {
            fProxyN vec = a.TempCopy();

            fProxyComp.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator %(in fProxyN a, fProxy s)
        {
            fProxyN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            fProxyComp.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator %(fProxy s, fProxyN a)
        {
            fProxyN vec = a.TempCopy();

            fProxyComp.modInpl(s, vec);

            return vec;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator +(in fProxyN a, in fProxyN b) {

            Assume.SameDim(in a, in b);

            fProxyN vec = a.TempCopy();

            fProxyComp.addInpl(vec, b);   // vec += b  (vec is the copy of a)

            return vec;
        }

        /// <summary>Component-wise subtraction; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator -(in fProxyN a, in fProxyN b) {

            Assume.SameDim(in a, in b);

            fProxyN vec = a.TempCopy();
            fProxyComp.subInpl(vec, b);
            
            return vec;
        }

        /// <summary>Component-wise multiplication; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator *(in fProxyN a, in fProxyN b) {

            Assume.SameDim(in a, in b);

            fProxyN vec = a.TempCopy();

            fProxyComp.mulInpl(b, vec);

            return vec;
        }

        /// <summary>Component-wise division (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator /(in fProxyN dividend, in fProxyN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            fProxyN newDividendVec = dividend.TempCopy();
            fProxyComp.divInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>Component-wise modulo (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN operator %(in fProxyN dividend, in fProxyN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            fProxyN newDividendVec = dividend.TempCopy();
            fProxyComp.modInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        #endregion

    }
}