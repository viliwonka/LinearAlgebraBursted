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
            shortComp.signFlipInPlace(vec);

            return vec;
        }
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator +(in shortN a, short s) {

            shortN vec = a.TempCopy();
            shortComp.addInPlace(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator +(short s, in shortN a) => a + s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator -(in shortN a, short s) {

            shortN vec = a.TempCopy();
            // v - s via a direct kernel, not v + (-s): the latter needs unary minus on the scalar,
            // which uint can't do (see OP.Component.short.cs), so this line is identical for
            // every generated type.
            shortComp.subInPlace(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator -(short s, in shortN a)
        {
            shortN vec = a.TempCopy();
            shortComp.subInPlace(s, vec);
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator *(in shortN a, short s) {
            
            shortN vec = a.TempCopy();

            shortComp.mulInPlace(vec, s);

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

            shortComp.divInPlace(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator /(short s, shortN a)
        {
            shortN vec = a.TempCopy();

            shortComp.divInPlace(s, vec);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator %(in shortN a, short s)
        {
            shortN vec = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            shortComp.modInPlace(vec, s);

            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator %(short s, shortN a)
        {
            shortN vec = a.TempCopy();

            shortComp.modInPlace(s, vec);

            return vec;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator ~(in shortN a) {

            shortN matrix = a.TempCopy();

            shortComp.bitwiseComplementInPlace(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator &(in shortN a, in short s) {

            shortN matrix = a.TempCopy();
            shortComp.bitwiseAndInPlace(matrix, s);
            return matrix;
        }
        public static shortN operator &(in short s, in shortN a) => a & s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator |(in shortN a, in short s) {
            shortN matrix = a.TempCopy();
            shortComp.bitwiseOrInPlace(matrix, s);
            return matrix;
        }
        public static shortN operator |(in short s, in shortN a) => a | s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator ^(in shortN a, in short b) {
            shortN matrix = a.TempCopy();
            shortComp.bitwiseXorInPlace(matrix, b);
            return matrix;
        }
        public static shortN operator ^(in short s, in shortN a) => a ^ s;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator <<(in shortN a, int shift) {
            shortN matrix = a.TempCopy();
            shortComp.bitwiseLeftShiftInPlace(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator >>(in shortN a, int shift) {
            shortN matrix = a.TempCopy();
            shortComp.bitwiseRightShiftInPlace(matrix, shift);
            return matrix;
        }

        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator +(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN vec = a.TempCopy();

            shortComp.addInPlace(vec, b);   // vec += b  (vec is the copy of a)

            return vec;
        }

        /// <summary>Component-wise subtraction; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator -(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN vec = a.TempCopy();
            shortComp.subInPlace(vec, b);
            
            return vec;
        }

        /// <summary>Component-wise multiplication; vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator *(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN vec = a.TempCopy();

            shortComp.mulInPlace(b, vec);

            return vec;
        }

        /// <summary>Component-wise division (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator /(in shortN dividend, in shortN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            shortN newDividendVec = dividend.TempCopy();
            shortComp.divInPlace(newDividendVec, divisor);

            return newDividendVec;
        }

        /// <summary>Component-wise modulo (dividend / divisor); vectors must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator %(in shortN dividend, in shortN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            shortN newDividendVec = dividend.TempCopy();
            shortComp.modInPlace(newDividendVec, divisor);

            return newDividendVec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator &(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN matrix = a.TempCopy();
            shortComp.bitwiseAndInPlace(matrix, b);
            return matrix;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator |(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN matrix = a.TempCopy();
            shortComp.bitwiseOrInPlace(matrix, b);
            return matrix;
        }
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN operator ^(in shortN a, in shortN b) {

            Assume.SameDim(in a, in b);

            shortN matrix = a.TempCopy();
            shortComp.bitwiseXorInPlace(matrix, b);
            return matrix;
        }
        
        #endregion

    }
}