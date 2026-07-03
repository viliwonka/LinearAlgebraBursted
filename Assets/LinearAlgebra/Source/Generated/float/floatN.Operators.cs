using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct floatN {

        #region SCALAR OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator +(in floatN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator -(in floatN a) {

            floatN vec = a.TempCopy();
            floatComp.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator +(in floatN a, float s) {

            floatN vec = a.TempCopy();
            floatComp.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator +(float s, in floatN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator -(in floatN a, float s) {
            
            floatN vec = a.TempCopy();
            floatComp.addInpl(vec, -s);
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator -(float s, in floatN a)
        {
            floatN vec = a.TempCopy();
            floatComp.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator *(in floatN a, float s) {
            
            floatN vec = a.TempCopy();

            floatComp.mulInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator *(float s, in floatN a) => a * s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator /(in floatN a, float s)
        {
            floatN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            floatComp.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator /(float s, floatN a)
        {
            floatN vec = a.TempCopy();

            floatComp.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator %(in floatN a, float s)
        {
            floatN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            floatComp.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator %(float s, floatN a)
        {
            floatN vec = a.TempCopy();

            floatComp.modInpl(s, vec);

            return vec;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator +(in floatN a, in floatN b) {

            Assume.SameDim(in a, in b);

            floatN vec = a.TempCopy();

            floatComp.addInpl(vec, b);   // vec += b  (vec is the copy of a)

            return vec;
        }

        /// <summary>Component-wise subtraction; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator -(in floatN a, in floatN b) {

            Assume.SameDim(in a, in b);

            floatN vec = a.TempCopy();
            floatComp.subInpl(vec, b);
            
            return vec;
        }

        /// <summary>Component-wise multiplication; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator *(in floatN a, in floatN b) {

            Assume.SameDim(in a, in b);

            floatN vec = a.TempCopy();

            floatComp.mulInpl(b, vec);

            return vec;
        }

        /// <summary>Component-wise division (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator /(in floatN dividend, in floatN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            floatN newDividendVec = dividend.TempCopy();
            floatComp.divInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>Component-wise modulo (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator %(in floatN dividend, in floatN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            floatN newDividendVec = dividend.TempCopy();
            floatComp.modInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        #endregion

    }
}