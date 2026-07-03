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
            intComp.signFlipInPlace(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator +(in intN a, int s) {

            intN vec = a.TempCopy();
            intComp.addInPlace(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator +(int s, in intN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator -(in intN a, int s) {
            
            intN vec = a.TempCopy();
            intComp.addInPlace(vec, (int)(-s));
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator -(int s, in intN a)
        {
            intN vec = a.TempCopy();
            intComp.subInPlace(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator *(in intN a, int s) {
            
            intN vec = a.TempCopy();

            intComp.mulInPlace(vec, s);

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

            intComp.divInPlace(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator /(int s, intN a)
        {
            intN vec = a.TempCopy();

            intComp.divInPlace(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator %(in intN a, int s)
        {
            intN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            intComp.modInPlace(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator %(int s, intN a)
        {
            intN vec = a.TempCopy();

            intComp.modInPlace(s, vec);

            return vec;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator ~(in intN a) {

            intN matrix = a.TempCopy();

            intComp.bitwiseComplementInPlace(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator &(in intN a, in int s) {

            intN matrix = a.TempCopy();
            intComp.bitwiseAndInPlace(matrix, s);
            return matrix;
        }
        public static intN operator &(in int s, in intN a) => a & s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator |(in intN a, in int s) {
            intN matrix = a.TempCopy();
            intComp.bitwiseOrInPlace(matrix, s);
            return matrix;
        }
        public static intN operator |(in int s, in intN a) => a | s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator ^(in intN a, in int b) {
            intN matrix = a.TempCopy();
            intComp.bitwiseXorInPlace(matrix, b);
            return matrix;
        }
        public static intN operator ^(in int s, in intN a) => a ^ s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator <<(in intN a, int shift) {
            intN matrix = a.TempCopy();
            intComp.bitwiseLeftShiftInPlace(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator >>(in intN a, int shift) {
            intN matrix = a.TempCopy();
            intComp.bitwiseRightShiftInPlace(matrix, shift);
            return matrix;
        }

        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator +(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN vec = a.TempCopy();

            intComp.addInPlace(vec, b);   // vec += b  (vec is the copy of a)

            return vec;
        }

        /// <summary>Component-wise subtraction; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator -(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN vec = a.TempCopy();
            intComp.subInPlace(vec, b);
            
            return vec;
        }

        /// <summary>Component-wise multiplication; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator *(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN vec = a.TempCopy();

            intComp.mulInPlace(b, vec);

            return vec;
        }

        /// <summary>Component-wise division (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator /(in intN dividend, in intN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            intN newDividendVec = dividend.TempCopy();
            intComp.divInPlace(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>Component-wise modulo (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator %(in intN dividend, in intN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            intN newDividendVec = dividend.TempCopy();
            intComp.modInPlace(newDividendVec, divisor);

            return newDividendVec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator &(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN matrix = a.TempCopy();
            intComp.bitwiseAndInPlace(matrix, b);
            return matrix;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator |(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN matrix = a.TempCopy();
            intComp.bitwiseOrInPlace(matrix, b);
            return matrix;
        }
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator ^(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN matrix = a.TempCopy();
            intComp.bitwiseXorInPlace(matrix, b);
            return matrix;
        }
        
        #endregion

    }
}