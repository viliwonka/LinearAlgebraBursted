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
            floatElem_OP.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator +(in floatN a, float s) {

            floatN vec = a.TempCopy();
            floatElem_OP.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator +(float s, in floatN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator -(in floatN a, float s) {
            
            floatN vec = a.TempCopy();
            floatElem_OP.addInpl(vec, -s);
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator -(float s, in floatN a)
        {
            floatN vec = a.TempCopy();
            floatElem_OP.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator *(in floatN a, float s) {
            
            floatN vec = a.TempCopy();

            floatElem_OP.mulInpl(vec, s);

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

            floatElem_OP.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator /(float s, floatN a)
        {
            floatN vec = a.TempCopy();

            floatElem_OP.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator %(in floatN a, float s)
        {
            floatN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            floatElem_OP.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator %(float s, floatN a)
        {
            floatN vec = a.TempCopy();

            floatElem_OP.modInpl(s, vec);

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
        public static floatN operator +(in floatN a, in floatN b) {

            Assume.SameDim(in a, in b);

            floatN vec = a.TempCopy();

            floatElem_OP.addInpl(vec, b);   // vec += b  (vec is the copy of a)

            return vec;
        }

        /// <summary>
        /// Component-wise subtraction
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator -(in floatN a, in floatN b) {

            Assume.SameDim(in a, in b);

            floatN vec = a.TempCopy();
            floatElem_OP.subInpl(vec, b);
            
            return vec;
        }

        /// <summary>
        /// Component-wise multiplication
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator *(in floatN a, in floatN b) {

            Assume.SameDim(in a, in b);

            floatN vec = a.TempCopy();

            floatElem_OP.mulInpl(b, vec);

            return vec;
        }

        /// <summary>
        /// Component-wise division
        /// Vectors have to be same dimensions
        /// Dividend / divisor
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator /(in floatN dividend, in floatN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            floatN newDividendVec = dividend.TempCopy();
            floatElem_OP.divInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>
        /// Component-wise modulo
        /// Vectors have to be same dimensions
        /// Dividend / divisor
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator %(in floatN dividend, in floatN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            floatN newDividendVec = dividend.TempCopy();
            floatElem_OP.modInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        #endregion

    }
}