using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct shortMxN {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator +(in shortMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator -(in shortMxN a)
        {
            shortMxN matrix = a.TempCopy();
            
            shortComp.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator +(in shortMxN lhs, short rhs)
        {
            shortMxN matrix = lhs.TempCopy();
            
            shortComp.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator +(short lhs, in shortMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator -(in shortMxN lhs, short rhs)
        {
            shortMxN matrix = lhs.TempCopy();
            
            shortComp.addInpl(matrix, (short)(-rhs));

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator -(short lhs, in shortMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            shortMxN matrix = rhs.TempCopy();
            shortComp.subInpl(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator *(in shortMxN a, short s)
        {
            shortMxN matrix = a.TempCopy();

            shortComp.mulInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator *(short lhs, in shortMxN rhs) => rhs * lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator /(in shortMxN a, short s)
        {
            shortMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            shortComp.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator /(short s, in shortMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer div by zero).
            shortMxN matrix = a.TempCopy();
            shortComp.divInpl(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator %(in shortMxN a, short s)
        {
            shortMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            shortComp.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator %(short s, in shortMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer mod by zero).
            shortMxN matrix = a.TempCopy();
            shortComp.modInpl(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator ~(in shortMxN a) {

            shortMxN matrix = a.TempCopy();

            shortComp.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator &(in shortMxN a, in short s) {

            shortMxN matrix = a.TempCopy();
            shortComp.bitwiseAndInpl(matrix, s);
            return matrix;
        }

        public static shortMxN operator &(in short s, in shortMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator |(in shortMxN a, in short s) {
            shortMxN matrix = a.TempCopy();
            shortComp.bitwiseOrInpl(matrix, s);
            return matrix;
        }

        public static shortMxN operator |(in short s, in shortMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator ^(in shortMxN a, in short s) {
            shortMxN matrix = a.TempCopy();
            shortComp.bitwiseXorInpl(matrix, s);
            return matrix;
        }

        public static shortMxN operator ^(in short s, in shortMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator <<(in shortMxN a, int shift) {
            shortMxN matrix = a.TempCopy();
            shortComp.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator >>(in shortMxN a, int shift) {
            shortMxN matrix = a.TempCopy();
            shortComp.bitwiseRightShiftInpl(matrix, shift);
            return matrix;
        }

        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator +(in shortMxN lhs, in shortMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            shortMxN matrix = lhs.TempCopy();

            shortComp.addInpl(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

            return matrix;
        }

        /// <summary>Component-wise subtraction; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator -(in shortMxN lhs, in shortMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            shortMxN matrix = lhs.TempCopy();

            shortComp.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>Component-wise multiplication; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator *(in shortMxN lhs, in shortMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            shortMxN matrix = lhs.TempCopy();

            shortComp.mulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>Component-wise division; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator /(in shortMxN dividend, in shortMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            shortMxN newDividendMatrix = dividend.TempCopy();

            shortComp.divInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>Component-wise modulo; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator %(in shortMxN dividend, in shortMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            shortComp.modInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator &(in shortMxN a, in shortMxN b) {
            
            Assume.SameDim(in a, in b);

            shortMxN matrix = a.TempCopy();
            shortComp.bitwiseAndInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator |(in shortMxN a, in shortMxN b) {

            Assume.SameDim(in a, in b);

            shortMxN matrix = a.TempCopy();
            shortComp.bitwiseOrInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN operator ^(in shortMxN a, in shortMxN b) {

            Assume.SameDim(in a, in b);

            shortMxN matrix = a.TempCopy();
            shortComp.bitwiseXorInpl(matrix, b);
            return matrix;
        }

        #endregion
    }
}