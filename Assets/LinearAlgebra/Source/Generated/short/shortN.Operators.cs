using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct shortN {

        #region SCALAR OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator +(in shortN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator -(in shortN a) {

            shortN vec = a.TempCopy();
            shortOP.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator +(in shortN a, short s) {

            shortN vec = a.TempCopy();
            shortOP.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator +(short s, in shortN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator -(in shortN a, short s) {
            
            shortN vec = a.TempCopy();
            shortOP.addInpl(vec, (short)(-s));
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator -(short s, in shortN a)
        {
            shortN vec = a.TempCopy();
            shortOP.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator *(in shortN a, short s) {
            
            shortN vec = a.TempCopy();

            shortOP.mulInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator *(short s, in shortN a) => a * s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator /(in shortN a, short s)
        {
            shortN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            shortOP.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator /(short s, shortN a)
        {
            shortN vec = a.TempCopy();

            shortOP.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator %(in shortN a, short s)
        {
            shortN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            shortOP.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator %(short s, shortN a)
        {
            shortN vec = a.TempCopy();

            shortOP.modInpl(s, vec);

            return vec;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator ~(in shortN a) {

            shortN matrix = a.TempCopy();

            shortOP.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator &(in shortN a, in short s) {

            shortN matrix = a.TempCopy();
            shortOP.bitwiseAndInpl(matrix, s);
            return matrix;
        }
        public static shortN operator &(in short s, in shortN a) => a & s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator |(in shortN a, in short s) {
            shortN matrix = a.TempCopy();
            shortOP.bitwiseOrInpl(matrix, s);
            return matrix;
        }
        public static shortN operator |(in short s, in shortN a) => a | s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator ^(in shortN a, in short b) {
            shortN matrix = a.TempCopy();
            shortOP.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        public static shortN operator ^(in short s, in shortN a) => a ^ s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator <<(in shortN a, int shift) {
            shortN matrix = a.TempCopy();
            shortOP.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator >>(in shortN a, int shift) {
            shortN matrix = a.TempCopy();
            shortOP.bitwiseRightShiftInpl(matrix, shift);
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
        public static shortN operator +(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN vec = a.TempCopy();

            shortOP.addInpl(b, vec);

            return vec;
        }

        /// <summary>
        /// Component-wise subtraction
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator -(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN vec = a.TempCopy();
            shortOP.subInpl(vec, b);
            
            return vec;
        }

        /// <summary>
        /// Component-wise multiplication
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator *(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN vec = a.TempCopy();

            shortOP.compMulInpl(b, vec);

            return vec;
        }

        /// <summary>
        /// Component-wise division
        /// Vectors have to be same dimensions
        /// Dividend / divisor
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator /(in shortN dividend, in shortN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            shortN newDividendVec = dividend.TempCopy();
            shortOP.compDivInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>
        /// Component-wise modulo
        /// Vectors have to be same dimensions
        /// Dividend / divisor
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator %(in shortN dividend, in shortN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            shortN newDividendVec = dividend.TempCopy();
            shortOP.compModDiv(newDividendVec, divisor);

            return newDividendVec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator &(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN matrix = a.TempCopy();
            shortOP.bitwiseAndInpl(matrix, b);
            return matrix;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator |(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN matrix = a.TempCopy();
            shortOP.bitwiseOrInpl(matrix, b);
            return matrix;
        }
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator ^(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN matrix = a.TempCopy();
            shortOP.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        
        #endregion

    }
}