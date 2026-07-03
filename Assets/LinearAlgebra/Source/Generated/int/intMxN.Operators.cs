using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct intMxN {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator +(in intMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator -(in intMxN a)
        {
            intMxN matrix = a.TempCopy();
            
            intComp.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator +(in intMxN lhs, int rhs)
        {
            intMxN matrix = lhs.TempCopy();
            
            intComp.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator +(int lhs, in intMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator -(in intMxN lhs, int rhs)
        {
            intMxN matrix = lhs.TempCopy();
            
            intComp.addInpl(matrix, (int)(-rhs));

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator -(int lhs, in intMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            intMxN matrix = rhs.TempCopy();
            intComp.subInpl(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator *(in intMxN a, int s)
        {
            intMxN matrix = a.TempCopy();

            intComp.mulInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator *(int lhs, in intMxN rhs) => rhs * lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator /(in intMxN a, int s)
        {
            intMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            intComp.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator /(int s, in intMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer div by zero).
            intMxN matrix = a.TempCopy();
            intComp.divInpl(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator %(in intMxN a, int s)
        {
            intMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            intComp.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator %(int s, in intMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer mod by zero).
            intMxN matrix = a.TempCopy();
            intComp.modInpl(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator ~(in intMxN a) {

            intMxN matrix = a.TempCopy();

            intComp.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator &(in intMxN a, in int s) {

            intMxN matrix = a.TempCopy();
            intComp.bitwiseAndInpl(matrix, s);
            return matrix;
        }

        public static intMxN operator &(in int s, in intMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator |(in intMxN a, in int s) {
            intMxN matrix = a.TempCopy();
            intComp.bitwiseOrInpl(matrix, s);
            return matrix;
        }

        public static intMxN operator |(in int s, in intMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator ^(in intMxN a, in int s) {
            intMxN matrix = a.TempCopy();
            intComp.bitwiseXorInpl(matrix, s);
            return matrix;
        }

        public static intMxN operator ^(in int s, in intMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator <<(in intMxN a, int shift) {
            intMxN matrix = a.TempCopy();
            intComp.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator >>(in intMxN a, int shift) {
            intMxN matrix = a.TempCopy();
            intComp.bitwiseRightShiftInpl(matrix, shift);
            return matrix;
        }

        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator +(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            intMxN matrix = lhs.TempCopy();

            intComp.addInpl(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

            return matrix;
        }

        /// <summary>Component-wise subtraction; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator -(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            intMxN matrix = lhs.TempCopy();

            intComp.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>Component-wise multiplication; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator *(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            intMxN matrix = lhs.TempCopy();

            intComp.mulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>Component-wise division; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator /(in intMxN dividend, in intMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            intMxN newDividendMatrix = dividend.TempCopy();

            intComp.divInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>Component-wise modulo; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator %(in intMxN dividend, in intMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            intComp.modInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator &(in intMxN a, in intMxN b) {
            
            Assume.SameDim(in a, in b);

            intMxN matrix = a.TempCopy();
            intComp.bitwiseAndInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator |(in intMxN a, in intMxN b) {

            Assume.SameDim(in a, in b);

            intMxN matrix = a.TempCopy();
            intComp.bitwiseOrInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator ^(in intMxN a, in intMxN b) {

            Assume.SameDim(in a, in b);

            intMxN matrix = a.TempCopy();
            intComp.bitwiseXorInpl(matrix, b);
            return matrix;
        }

        #endregion
    }
}