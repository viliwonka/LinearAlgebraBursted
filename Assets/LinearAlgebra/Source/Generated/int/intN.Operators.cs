using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct intN {

        #region SCALAR OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator +(in intN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator -(in intN a) {

            intN vec = a.TempCopy();
            intElem_OP.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator +(in intN a, int s) {

            intN vec = a.TempCopy();
            intElem_OP.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator +(int s, in intN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator -(in intN a, int s) {
            
            intN vec = a.TempCopy();
            intElem_OP.addInpl(vec, (int)(-s));
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator -(int s, in intN a)
        {
            intN vec = a.TempCopy();
            intElem_OP.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator *(in intN a, int s) {
            
            intN vec = a.TempCopy();

            intElem_OP.mulInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator *(int s, in intN a) => a * s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator /(in intN a, int s)
        {
            intN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            intElem_OP.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator /(int s, intN a)
        {
            intN vec = a.TempCopy();

            intElem_OP.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator %(in intN a, int s)
        {
            intN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            intElem_OP.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator %(int s, intN a)
        {
            intN vec = a.TempCopy();

            intElem_OP.modInpl(s, vec);

            return vec;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator ~(in intN a) {

            intN matrix = a.TempCopy();

            intElem_OP.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator &(in intN a, in int s) {

            intN matrix = a.TempCopy();
            intElem_OP.bitwiseAndInpl(matrix, s);
            return matrix;
        }
        public static intN operator &(in int s, in intN a) => a & s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator |(in intN a, in int s) {
            intN matrix = a.TempCopy();
            intElem_OP.bitwiseOrInpl(matrix, s);
            return matrix;
        }
        public static intN operator |(in int s, in intN a) => a | s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator ^(in intN a, in int b) {
            intN matrix = a.TempCopy();
            intElem_OP.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        public static intN operator ^(in int s, in intN a) => a ^ s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator <<(in intN a, int shift) {
            intN matrix = a.TempCopy();
            intElem_OP.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator >>(in intN a, int shift) {
            intN matrix = a.TempCopy();
            intElem_OP.bitwiseRightShiftInpl(matrix, shift);
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
        public static intN operator +(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN vec = a.TempCopy();

            intElem_OP.addInpl(vec, b);   // vec += b  (vec is the copy of a)

            return vec;
        }

        /// <summary>
        /// Component-wise subtraction
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator -(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN vec = a.TempCopy();
            intElem_OP.subInpl(vec, b);
            
            return vec;
        }

        /// <summary>
        /// Component-wise multiplication
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator *(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN vec = a.TempCopy();

            intElem_OP.mulInpl(b, vec);

            return vec;
        }

        /// <summary>
        /// Component-wise division
        /// Vectors have to be same dimensions
        /// Dividend / divisor
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator /(in intN dividend, in intN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            intN newDividendVec = dividend.TempCopy();
            intElem_OP.divInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>
        /// Component-wise modulo
        /// Vectors have to be same dimensions
        /// Dividend / divisor
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator %(in intN dividend, in intN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            intN newDividendVec = dividend.TempCopy();
            intElem_OP.modInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator &(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN matrix = a.TempCopy();
            intElem_OP.bitwiseAndInpl(matrix, b);
            return matrix;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator |(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN matrix = a.TempCopy();
            intElem_OP.bitwiseOrInpl(matrix, b);
            return matrix;
        }
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator ^(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN matrix = a.TempCopy();
            intElem_OP.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        
        #endregion

    }
}