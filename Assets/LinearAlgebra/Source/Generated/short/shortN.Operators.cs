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
            shortComp.signFlipInpl(vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator +(in shortN a, short s) {

            shortN vec = a.TempCopy();
            shortComp.addInpl(vec, s);

            return vec; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator +(short s, in shortN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator -(in shortN a, short s) {
            
            shortN vec = a.TempCopy();
            shortComp.addInpl(vec, (short)(-s));
            
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator -(short s, in shortN a)
        {
            shortN vec = a.TempCopy();
            shortComp.subInpl(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator *(in shortN a, short s) {
            
            shortN vec = a.TempCopy();

            shortComp.mulInpl(vec, s);

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

            shortComp.divInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator /(short s, shortN a)
        {
            shortN vec = a.TempCopy();

            shortComp.divInpl(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator %(in shortN a, short s)
        {
            shortN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            shortComp.modInpl(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator %(short s, shortN a)
        {
            shortN vec = a.TempCopy();

            shortComp.modInpl(s, vec);

            return vec;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator ~(in shortN a) {

            shortN matrix = a.TempCopy();

            shortComp.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator &(in shortN a, in short s) {

            shortN matrix = a.TempCopy();
            shortComp.bitwiseAndInpl(matrix, s);
            return matrix;
        }
        public static shortN operator &(in short s, in shortN a) => a & s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator |(in shortN a, in short s) {
            shortN matrix = a.TempCopy();
            shortComp.bitwiseOrInpl(matrix, s);
            return matrix;
        }
        public static shortN operator |(in short s, in shortN a) => a | s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator ^(in shortN a, in short b) {
            shortN matrix = a.TempCopy();
            shortComp.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        public static shortN operator ^(in short s, in shortN a) => a ^ s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator <<(in shortN a, int shift) {
            shortN matrix = a.TempCopy();
            shortComp.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator >>(in shortN a, int shift) {
            shortN matrix = a.TempCopy();
            shortComp.bitwiseRightShiftInpl(matrix, shift);
            return matrix;
        }

        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator +(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN vec = a.TempCopy();

            shortComp.addInpl(vec, b);   // vec += b  (vec is the copy of a)

            return vec;
        }

        /// <summary>Component-wise subtraction; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator -(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN vec = a.TempCopy();
            shortComp.subInpl(vec, b);
            
            return vec;
        }

        /// <summary>Component-wise multiplication; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator *(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN vec = a.TempCopy();

            shortComp.mulInpl(b, vec);

            return vec;
        }

        /// <summary>Component-wise division (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator /(in shortN dividend, in shortN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            shortN newDividendVec = dividend.TempCopy();
            shortComp.divInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>Component-wise modulo (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator %(in shortN dividend, in shortN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            shortN newDividendVec = dividend.TempCopy();
            shortComp.modInpl(newDividendVec, divisor);

            return newDividendVec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator &(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN matrix = a.TempCopy();
            shortComp.bitwiseAndInpl(matrix, b);
            return matrix;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator |(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN matrix = a.TempCopy();
            shortComp.bitwiseOrInpl(matrix, b);
            return matrix;
        }
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator ^(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN matrix = a.TempCopy();
            shortComp.bitwiseXorInpl(matrix, b);
            return matrix;
        }
        
        #endregion

    }
}