using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct doubleN {

        #region SCALAR OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator +(in doubleN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator -(in doubleN a) {

            doubleN vec = a.TempCopy();
            doubleElem_OP.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator +(in doubleN a, double s) {

            doubleN vec = a.TempCopy();
            doubleElem_OP.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator +(double s, in doubleN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator -(in doubleN a, double s) {
            
            doubleN vec = a.TempCopy();
            doubleElem_OP.addInpl(vec, -s);
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator -(double s, in doubleN a)
        {
            doubleN vec = a.TempCopy();
            doubleElem_OP.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator *(in doubleN a, double s) {
            
            doubleN vec = a.TempCopy();

            doubleElem_OP.mulInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator *(double s, in doubleN a) => a * s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator /(in doubleN a, double s)
        {
            doubleN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            doubleElem_OP.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator /(double s, doubleN a)
        {
            doubleN vec = a.TempCopy();

            doubleElem_OP.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator %(in doubleN a, double s)
        {
            doubleN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            doubleElem_OP.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator %(double s, doubleN a)
        {
            doubleN vec = a.TempCopy();

            doubleElem_OP.modInpl(s, vec);

            return vec;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator +(in doubleN a, in doubleN b) {

            Assume.SameDim(in a, in b);

            doubleN vec = a.TempCopy();

            doubleElem_OP.addInpl(vec, b);   // vec += b  (vec is the copy of a)

            return vec;
        }

        /// <summary>Component-wise subtraction; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator -(in doubleN a, in doubleN b) {

            Assume.SameDim(in a, in b);

            doubleN vec = a.TempCopy();
            doubleElem_OP.subInpl(vec, b);
            
            return vec;
        }

        /// <summary>Component-wise multiplication; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator *(in doubleN a, in doubleN b) {

            Assume.SameDim(in a, in b);

            doubleN vec = a.TempCopy();

            doubleElem_OP.mulInpl(b, vec);

            return vec;
        }

        /// <summary>Component-wise division (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator /(in doubleN dividend, in doubleN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            doubleN newDividendVec = dividend.TempCopy();
            doubleElem_OP.divInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>Component-wise modulo (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN operator %(in doubleN dividend, in doubleN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            doubleN newDividendVec = dividend.TempCopy();
            doubleElem_OP.modInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        #endregion

    }
}