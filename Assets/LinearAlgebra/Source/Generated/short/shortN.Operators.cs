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
            short_OP.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator +(in shortN a, short s) {

            shortN vec = a.TempCopy();
            short_OP.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator +(short s, in shortN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator -(in shortN a, short s) {
            
            shortN vec = a.TempCopy();
            short_OP.addInpl(vec, (short)(-s));
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator -(short s, in shortN a)
        {
            shortN vec = a.TempCopy();
            short_OP.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator *(in shortN a, short s) {
            
            shortN vec = a.TempCopy();

            short_OP.mulInpl(vec, s);

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

            short_OP.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator /(short s, shortN a)
        {
            shortN vec = a.TempCopy();

            short_OP.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator %(in shortN a, short s)
        {
            shortN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            short_OP.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator %(short s, shortN a)
        {
            shortN vec = a.TempCopy();

            short_OP.modInpl(s, vec);

            return vec;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator ~(in shortN a) {

            shortN matrix = a.TempCopy();

            short_OP.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator &(in shortN a, in short s) {

            shortN matrix = a.TempCopy();
            short_OP.bitwiseAndInpl(matrix, s);
            return matrix;
        }
        public static shortN operator &(in short s, in shortN a) => a & s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator |(in shortN a, in short s) {
            shortN matrix = a.TempCopy();
            short_OP.bitwiseOrInpl(matrix, s);
            return matrix;
        }
        public static shortN operator |(in short s, in shortN a) => a | s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator ^(in shortN a, in short b) {
            shortN matrix = a.TempCopy();
            short_OP.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        public static shortN operator ^(in short s, in shortN a) => a ^ s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator <<(in shortN a, int shift) {
            shortN matrix = a.TempCopy();
            short_OP.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator >>(in shortN a, int shift) {
            shortN matrix = a.TempCopy();
            short_OP.bitwiseRightShiftInpl(matrix, shift);
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

            short_OP.addInpl(vec, b);   // vec += b  (vec is the copy of a)

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
            short_OP.subInpl(vec, b);
            
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

            short_OP.compMulInpl(b, vec);

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
            short_OP.compDivInpl(newDividendVec, divisor);

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
            short_OP.compModDiv(newDividendVec, divisor);

            return newDividendVec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator &(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN matrix = a.TempCopy();
            short_OP.bitwiseAndInpl(matrix, b);
            return matrix;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator |(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN matrix = a.TempCopy();
            short_OP.bitwiseOrInpl(matrix, b);
            return matrix;
        }
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator ^(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN matrix = a.TempCopy();
            short_OP.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        
        #endregion

    }
}