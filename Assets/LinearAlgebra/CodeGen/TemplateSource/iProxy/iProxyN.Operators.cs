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
            iProxyComp.signFlipInPlace(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator +(in iProxyN a, iProxy s) {

            iProxyN vec = a.TempCopy();
            iProxyComp.addInPlace(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator +(iProxy s, in iProxyN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator -(in iProxyN a, iProxy s) {
            
            iProxyN vec = a.TempCopy();
            iProxyComp.addInPlace(vec, (iProxy)(-s));
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator -(iProxy s, in iProxyN a)
        {
            iProxyN vec = a.TempCopy();
            iProxyComp.subInPlace(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator *(in iProxyN a, iProxy s) {
            
            iProxyN vec = a.TempCopy();

            iProxyComp.mulInPlace(vec, s);

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

            iProxyComp.divInPlace(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator /(iProxy s, iProxyN a)
        {
            iProxyN vec = a.TempCopy();

            iProxyComp.divInPlace(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator %(in iProxyN a, iProxy s)
        {
            iProxyN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            iProxyComp.modInPlace(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator %(iProxy s, iProxyN a)
        {
            iProxyN vec = a.TempCopy();

            iProxyComp.modInPlace(s, vec);

            return vec;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator ~(in iProxyN a) {

            iProxyN matrix = a.TempCopy();

            iProxyComp.bitwiseComplementInPlace(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator &(in iProxyN a, in iProxy s) {

            iProxyN matrix = a.TempCopy();
            iProxyComp.bitwiseAndInPlace(matrix, s);
            return matrix;
        }
        public static iProxyN operator &(in iProxy s, in iProxyN a) => a & s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator |(in iProxyN a, in iProxy s) {
            iProxyN matrix = a.TempCopy();
            iProxyComp.bitwiseOrInPlace(matrix, s);
            return matrix;
        }
        public static iProxyN operator |(in iProxy s, in iProxyN a) => a | s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator ^(in iProxyN a, in iProxy b) {
            iProxyN matrix = a.TempCopy();
            iProxyComp.bitwiseXorInPlace(matrix, b);
            return matrix;
        }
        public static iProxyN operator ^(in iProxy s, in iProxyN a) => a ^ s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator <<(in iProxyN a, int shift) {
            iProxyN matrix = a.TempCopy();
            iProxyComp.bitwiseLeftShiftInPlace(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator >>(in iProxyN a, int shift) {
            iProxyN matrix = a.TempCopy();
            iProxyComp.bitwiseRightShiftInPlace(matrix, shift);
            return matrix;
        }

        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator +(in iProxyN a, in iProxyN b) {

            Assume.SameDim(in a, in b);

            iProxyN vec = a.TempCopy();

            iProxyComp.addInPlace(vec, b);   // vec += b  (vec is the copy of a)

            return vec;
        }

        /// <summary>Component-wise subtraction; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator -(in iProxyN a, in iProxyN b) {

            Assume.SameDim(in a, in b);

            iProxyN vec = a.TempCopy();
            iProxyComp.subInPlace(vec, b);
            
            return vec;
        }

        /// <summary>Component-wise multiplication; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator *(in iProxyN a, in iProxyN b) {

            Assume.SameDim(in a, in b);

            iProxyN vec = a.TempCopy();

            iProxyComp.mulInPlace(b, vec);

            return vec;
        }

        /// <summary>Component-wise division (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator /(in iProxyN dividend, in iProxyN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            iProxyN newDividendVec = dividend.TempCopy();
            iProxyComp.divInPlace(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>Component-wise modulo (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator %(in iProxyN dividend, in iProxyN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            iProxyN newDividendVec = dividend.TempCopy();
            iProxyComp.modInPlace(newDividendVec, divisor);

            return newDividendVec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator &(in iProxyN a, in iProxyN b) {

            Assume.SameDim(in a, in b);

            iProxyN matrix = a.TempCopy();
            iProxyComp.bitwiseAndInPlace(matrix, b);
            return matrix;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator |(in iProxyN a, in iProxyN b) {

            Assume.SameDim(in a, in b);

            iProxyN matrix = a.TempCopy();
            iProxyComp.bitwiseOrInPlace(matrix, b);
            return matrix;
        }
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyN operator ^(in iProxyN a, in iProxyN b) {

            Assume.SameDim(in a, in b);

            iProxyN matrix = a.TempCopy();
            iProxyComp.bitwiseXorInPlace(matrix, b);
            return matrix;
        }
        
        #endregion

    }
}