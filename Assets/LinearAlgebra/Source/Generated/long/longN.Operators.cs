using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct longN {

        #region SCALAR OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator +(in longN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator -(in longN a) {

            longN vec = a.TempCopy();
            longOP.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator +(in longN a, long s) {

            longN vec = a.TempCopy();
            longOP.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator +(long s, in longN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator -(in longN a, long s) {
            
            longN vec = a.TempCopy();
            longOP.addInpl(vec, (long)(-s));
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator -(long s, in longN a)
        {
            longN vec = a.TempCopy();
            longOP.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator *(in longN a, long s) {
            
            longN vec = a.TempCopy();

            longOP.mulInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator *(long s, in longN a) => a * s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator /(in longN a, long s)
        {
            longN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            longOP.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator /(long s, longN a)
        {
            longN vec = a.TempCopy();

            longOP.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator %(in longN a, long s)
        {
            longN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            longOP.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator %(long s, longN a)
        {
            longN vec = a.TempCopy();

            longOP.modInpl(s, vec);

            return vec;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator ~(in longN a) {

            longN matrix = a.TempCopy();

            longOP.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator &(in longN a, in long s) {

            longN matrix = a.TempCopy();
            longOP.bitwiseAndInpl(matrix, s);
            return matrix;
        }
        public static longN operator &(in long s, in longN a) => a & s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator |(in longN a, in long s) {
            longN matrix = a.TempCopy();
            longOP.bitwiseOrInpl(matrix, s);
            return matrix;
        }
        public static longN operator |(in long s, in longN a) => a | s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator ^(in longN a, in long b) {
            longN matrix = a.TempCopy();
            longOP.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        public static longN operator ^(in long s, in longN a) => a ^ s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator <<(in longN a, int shift) {
            longN matrix = a.TempCopy();
            longOP.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator >>(in longN a, int shift) {
            longN matrix = a.TempCopy();
            longOP.bitwiseRightShiftInpl(matrix, shift);
            return matrix;
        }

        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>
        /// Component-wise addition
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator +(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN vec = a.TempCopy();

            longOP.addInpl(b, vec);

            return vec;
        }

        /// <summary>
        /// Component-wise subtraction
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator -(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN vec = a.TempCopy();
            longOP.subInpl(vec, b);
            
            return vec;
        }

        /// <summary>
        /// Component-wise multiplication
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator *(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN vec = a.TempCopy();

            longOP.compMulInpl(b, vec);

            return vec;
        }

        /// <summary>
        /// Component-wise division
        /// Vectors have to be same dimensions
        /// Dividend / divisor
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator /(in longN dividend, in longN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            longN newDividendVec = dividend.TempCopy();
            longOP.compDivInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>
        /// Component-wise modulo
        /// Vectors have to be same dimensions
        /// Dividend / divisor
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator %(in longN dividend, in longN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            longN newDividendVec = dividend.TempCopy();
            longOP.compModDiv(newDividendVec, divisor);

            return newDividendVec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator &(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN matrix = a.TempCopy();
            longOP.bitwiseAndInpl(matrix, b);
            return matrix;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator |(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN matrix = a.TempCopy();
            longOP.bitwiseOrInpl(matrix, b);
            return matrix;
        }
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator ^(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN matrix = a.TempCopy();
            longOP.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        
        #endregion

    }
}