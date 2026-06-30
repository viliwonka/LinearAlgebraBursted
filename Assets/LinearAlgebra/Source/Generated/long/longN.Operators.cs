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
            long_OP.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator +(in longN a, long s) {

            longN vec = a.TempCopy();
            long_OP.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator +(long s, in longN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator -(in longN a, long s) {
            
            longN vec = a.TempCopy();
            long_OP.addInpl(vec, (long)(-s));
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator -(long s, in longN a)
        {
            longN vec = a.TempCopy();
            long_OP.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator *(in longN a, long s) {
            
            longN vec = a.TempCopy();

            long_OP.mulInpl(vec, s);

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

            long_OP.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator /(long s, longN a)
        {
            longN vec = a.TempCopy();

            long_OP.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator %(in longN a, long s)
        {
            longN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            long_OP.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator %(long s, longN a)
        {
            longN vec = a.TempCopy();

            long_OP.modInpl(s, vec);

            return vec;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator ~(in longN a) {

            longN matrix = a.TempCopy();

            long_OP.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator &(in longN a, in long s) {

            longN matrix = a.TempCopy();
            long_OP.bitwiseAndInpl(matrix, s);
            return matrix;
        }
        public static longN operator &(in long s, in longN a) => a & s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator |(in longN a, in long s) {
            longN matrix = a.TempCopy();
            long_OP.bitwiseOrInpl(matrix, s);
            return matrix;
        }
        public static longN operator |(in long s, in longN a) => a | s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator ^(in longN a, in long b) {
            longN matrix = a.TempCopy();
            long_OP.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        public static longN operator ^(in long s, in longN a) => a ^ s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator <<(in longN a, int shift) {
            longN matrix = a.TempCopy();
            long_OP.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator >>(in longN a, int shift) {
            longN matrix = a.TempCopy();
            long_OP.bitwiseRightShiftInpl(matrix, shift);
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

            long_OP.addInpl(vec, b);   // vec += b  (vec is the copy of a)

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
            long_OP.subInpl(vec, b);
            
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

            long_OP.compMulInpl(b, vec);

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
            long_OP.compDivInpl(newDividendVec, divisor);

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
            long_OP.compModDiv(newDividendVec, divisor);

            return newDividendVec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator &(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN matrix = a.TempCopy();
            long_OP.bitwiseAndInpl(matrix, b);
            return matrix;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator |(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN matrix = a.TempCopy();
            long_OP.bitwiseOrInpl(matrix, b);
            return matrix;
        }
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator ^(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN matrix = a.TempCopy();
            long_OP.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        
        #endregion

    }
}