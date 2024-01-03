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
            floatOP.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator +(in floatN a, float s) {

            floatN vec = a.TempCopy();
            floatOP.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator +(float s, in floatN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator -(in floatN a, float s) {
            
            floatN vec = a.TempCopy();
            floatOP.addInpl(vec, -s);
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator -(float s, in floatN a)
        {
            floatN vec = a.TempCopy();
            floatOP.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator *(in floatN a, float s) {
            
            floatN vec = a.TempCopy();

            floatOP.mulInpl(vec, s);

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

            floatOP.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator /(float s, floatN a)
        {
            floatN vec = a.TempCopy();

            floatOP.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator %(in floatN a, float s)
        {
            floatN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            floatOP.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN operator %(float s, floatN a)
        {
            floatN vec = a.TempCopy();

            floatOP.modInpl(s, vec);

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

            floatOP.addInpl(b, vec);

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
            floatOP.subInpl(vec, b);
            
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

            floatOP.compMulInpl(b, vec);

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
            floatOP.compDivInpl(newDividendVec, divisor);

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
            floatOP.compModDiv(newDividendVec, divisor);

            return newDividendVec;
        }

        #endregion

    }
}