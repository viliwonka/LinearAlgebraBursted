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
            intComp.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator +(in intN a, int s) {

            intN vec = a.TempCopy();
            intComp.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator +(int s, in intN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator -(in intN a, int s) {
            
            intN vec = a.TempCopy();
            intComp.addInpl(vec, (int)(-s));
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator -(int s, in intN a)
        {
            intN vec = a.TempCopy();
            intComp.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator *(in intN a, int s) {
            
            intN vec = a.TempCopy();

            intComp.mulInpl(vec, s);

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

            intComp.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator /(int s, intN a)
        {
            intN vec = a.TempCopy();

            intComp.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator %(in intN a, int s)
        {
            intN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            intComp.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator %(int s, intN a)
        {
            intN vec = a.TempCopy();

            intComp.modInpl(s, vec);

            return vec;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator ~(in intN a) {

            intN matrix = a.TempCopy();

            intComp.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator &(in intN a, in int s) {

            intN matrix = a.TempCopy();
            intComp.bitwiseAndInpl(matrix, s);
            return matrix;
        }
        public static intN operator &(in int s, in intN a) => a & s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator |(in intN a, in int s) {
            intN matrix = a.TempCopy();
            intComp.bitwiseOrInpl(matrix, s);
            return matrix;
        }
        public static intN operator |(in int s, in intN a) => a | s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator ^(in intN a, in int b) {
            intN matrix = a.TempCopy();
            intComp.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        public static intN operator ^(in int s, in intN a) => a ^ s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator <<(in intN a, int shift) {
            intN matrix = a.TempCopy();
            intComp.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator >>(in intN a, int shift) {
            intN matrix = a.TempCopy();
            intComp.bitwiseRightShiftInpl(matrix, shift);
            return matrix;
        }

        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator +(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN vec = a.TempCopy();

            intComp.addInpl(vec, b);   // vec += b  (vec is the copy of a)

            return vec;
        }

        /// <summary>Component-wise subtraction; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator -(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN vec = a.TempCopy();
            intComp.subInpl(vec, b);
            
            return vec;
        }

        /// <summary>Component-wise multiplication; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator *(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN vec = a.TempCopy();

            intComp.mulInpl(b, vec);

            return vec;
        }

        /// <summary>Component-wise division (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator /(in intN dividend, in intN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            intN newDividendVec = dividend.TempCopy();
            intComp.divInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>Component-wise modulo (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator %(in intN dividend, in intN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            intN newDividendVec = dividend.TempCopy();
            intComp.modInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator &(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN matrix = a.TempCopy();
            intComp.bitwiseAndInpl(matrix, b);
            return matrix;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator |(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN matrix = a.TempCopy();
            intComp.bitwiseOrInpl(matrix, b);
            return matrix;
        }
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN operator ^(in intN a, in intN b) {

            Assume.SameDim(in a, in b);

            intN matrix = a.TempCopy();
            intComp.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        
        #endregion

    }
}