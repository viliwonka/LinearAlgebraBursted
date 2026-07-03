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
            longComp.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator +(in longN a, long s) {

            longN vec = a.TempCopy();
            longComp.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator +(long s, in longN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator -(in longN a, long s) {
            
            longN vec = a.TempCopy();
            longComp.addInpl(vec, (long)(-s));
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator -(long s, in longN a)
        {
            longN vec = a.TempCopy();
            longComp.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator *(in longN a, long s) {
            
            longN vec = a.TempCopy();

            longComp.mulInpl(vec, s);

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

            longComp.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator /(long s, longN a)
        {
            longN vec = a.TempCopy();

            longComp.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator %(in longN a, long s)
        {
            longN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            longComp.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator %(long s, longN a)
        {
            longN vec = a.TempCopy();

            longComp.modInpl(s, vec);

            return vec;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator ~(in longN a) {

            longN matrix = a.TempCopy();

            longComp.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator &(in longN a, in long s) {

            longN matrix = a.TempCopy();
            longComp.bitwiseAndInpl(matrix, s);
            return matrix;
        }
        public static longN operator &(in long s, in longN a) => a & s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator |(in longN a, in long s) {
            longN matrix = a.TempCopy();
            longComp.bitwiseOrInpl(matrix, s);
            return matrix;
        }
        public static longN operator |(in long s, in longN a) => a | s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator ^(in longN a, in long b) {
            longN matrix = a.TempCopy();
            longComp.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        public static longN operator ^(in long s, in longN a) => a ^ s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator <<(in longN a, int shift) {
            longN matrix = a.TempCopy();
            longComp.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator >>(in longN a, int shift) {
            longN matrix = a.TempCopy();
            longComp.bitwiseRightShiftInpl(matrix, shift);
            return matrix;
        }

        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator +(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN vec = a.TempCopy();

            longComp.addInpl(vec, b);   // vec += b  (vec is the copy of a)

            return vec;
        }

        /// <summary>Component-wise subtraction; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator -(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN vec = a.TempCopy();
            longComp.subInpl(vec, b);
            
            return vec;
        }

        /// <summary>Component-wise multiplication; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator *(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN vec = a.TempCopy();

            longComp.mulInpl(b, vec);

            return vec;
        }

        /// <summary>Component-wise division (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator /(in longN dividend, in longN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            longN newDividendVec = dividend.TempCopy();
            longComp.divInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>Component-wise modulo (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator %(in longN dividend, in longN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            longN newDividendVec = dividend.TempCopy();
            longComp.modInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator &(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN matrix = a.TempCopy();
            longComp.bitwiseAndInpl(matrix, b);
            return matrix;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator |(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN matrix = a.TempCopy();
            longComp.bitwiseOrInpl(matrix, b);
            return matrix;
        }
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN operator ^(in longN a, in longN b) {

            Assume.SameDim(in a, in b);

            longN matrix = a.TempCopy();
            longComp.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        
        #endregion

    }
}