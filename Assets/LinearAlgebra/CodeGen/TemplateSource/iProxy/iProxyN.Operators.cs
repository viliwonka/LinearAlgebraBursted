using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct iProxyN {

        #region SCALAR OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator +(in iProxyN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator -(in iProxyN a) {

            iProxyN vec = a.TempCopy();
            iProxyOP.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator +(in iProxyN a, iProxy s) {

            iProxyN vec = a.TempCopy();
            iProxyOP.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator +(iProxy s, in iProxyN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator -(in iProxyN a, iProxy s) {
            
            iProxyN vec = a.TempCopy();
            iProxyOP.addInpl(vec, (iProxy)(-s));
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator -(iProxy s, in iProxyN a)
        {
            iProxyN vec = a.TempCopy();
            iProxyOP.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator *(in iProxyN a, iProxy s) {
            
            iProxyN vec = a.TempCopy();

            iProxyOP.mulInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator *(iProxy s, in iProxyN a) => a * s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator /(in iProxyN a, iProxy s)
        {
            iProxyN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            iProxyOP.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator /(iProxy s, iProxyN a)
        {
            iProxyN vec = a.TempCopy();

            iProxyOP.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator %(in iProxyN a, iProxy s)
        {
            iProxyN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            iProxyOP.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator %(iProxy s, iProxyN a)
        {
            iProxyN vec = a.TempCopy();

            iProxyOP.modInpl(s, vec);

            return vec;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator ~(in iProxyN a) {

            iProxyN matrix = a.TempCopy();

            iProxyOP.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator &(in iProxyN a, in iProxy s) {

            iProxyN matrix = a.TempCopy();
            iProxyOP.bitwiseAndInpl(matrix, s);
            return matrix;
        }
        public static iProxyN operator &(in iProxy s, in iProxyN a) => a & s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator |(in iProxyN a, in iProxy s) {
            iProxyN matrix = a.TempCopy();
            iProxyOP.bitwiseOrInpl(matrix, s);
            return matrix;
        }
        public static iProxyN operator |(in iProxy s, in iProxyN a) => a | s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator ^(in iProxyN a, in iProxy b) {
            iProxyN matrix = a.TempCopy();
            iProxyOP.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        public static iProxyN operator ^(in iProxy s, in iProxyN a) => a ^ s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator <<(in iProxyN a, int shift) {
            iProxyN matrix = a.TempCopy();
            iProxyOP.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator >>(in iProxyN a, int shift) {
            iProxyN matrix = a.TempCopy();
            iProxyOP.bitwiseRightShiftInpl(matrix, shift);
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
        public static iProxyN operator +(in iProxyN a, in iProxyN b) {

            Assume.SameDim(in a, in b);

            iProxyN vec = a.TempCopy();

            iProxyOP.addInpl(b, vec);

            return vec;
        }

        /// <summary>
        /// Component-wise subtraction
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator -(in iProxyN a, in iProxyN b) {

            Assume.SameDim(in a, in b);

            iProxyN vec = a.TempCopy();
            iProxyOP.subInpl(vec, b);
            
            return vec;
        }

        /// <summary>
        /// Component-wise multiplication
        /// Vectors have to be same dimensions
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator *(in iProxyN a, in iProxyN b) {

            Assume.SameDim(in a, in b);

            iProxyN vec = a.TempCopy();

            iProxyOP.compMulInpl(b, vec);

            return vec;
        }

        /// <summary>
        /// Component-wise division
        /// Vectors have to be same dimensions
        /// Dividend / divisor
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator /(in iProxyN dividend, in iProxyN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            iProxyN newDividendVec = dividend.TempCopy();
            iProxyOP.compDivInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>
        /// Component-wise modulo
        /// Vectors have to be same dimensions
        /// Dividend / divisor
        /// </summary>
        /// <returns>Same dimension vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator %(in iProxyN dividend, in iProxyN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            iProxyN newDividendVec = dividend.TempCopy();
            iProxyOP.compModDiv(newDividendVec, divisor);

            return newDividendVec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator &(in iProxyN a, in iProxyN b) {

            Assume.SameDim(in a, in b);

            iProxyN matrix = a.TempCopy();
            iProxyOP.bitwiseAndInpl(matrix, b);
            return matrix;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator |(in iProxyN a, in iProxyN b) {

            Assume.SameDim(in a, in b);

            iProxyN matrix = a.TempCopy();
            iProxyOP.bitwiseOrInpl(matrix, b);
            return matrix;
        }
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator ^(in iProxyN a, in iProxyN b) {

            Assume.SameDim(in a, in b);

            iProxyN matrix = a.TempCopy();
            iProxyOP.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        
        #endregion

    }
}