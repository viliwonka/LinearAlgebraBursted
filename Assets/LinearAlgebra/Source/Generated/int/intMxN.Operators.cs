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

            intComp.signFlipInPlace(matrix);

            return matrix;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator +(in intMxN lhs, int rhs)
        {
            intMxN matrix = lhs.TempCopy();

            intComp.addInPlace(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator +(int lhs, in intMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator -(in intMxN lhs, int rhs)
        {
            intMxN matrix = lhs.TempCopy();

            // v - s via a direct kernel, not v + (-s): the latter needs unary minus on the scalar,
            // which uint can't do (see OP.Component.int.cs), so this line is identical for
            // every generated type.
            intComp.subInPlace(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator -(int lhs, in intMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            intMxN matrix = rhs.TempCopy();
            intComp.subInPlace(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator *(in intMxN a, int s)
        {
            intMxN matrix = a.TempCopy();

            intComp.mulInPlace(matrix, s);

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

            intComp.divInPlace(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator /(int s, in intMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer div by zero).
            intMxN matrix = a.TempCopy();
            intComp.divInPlace(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator %(in intMxN a, int s)
        {
            intMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            intComp.modInPlace(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator %(int s, in intMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer mod by zero).
            intMxN matrix = a.TempCopy();
            intComp.modInPlace(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator ~(in intMxN a) {

            intMxN matrix = a.TempCopy();

            intComp.bitwiseComplementInPlace(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator &(in intMxN a, in int s) {

            intMxN matrix = a.TempCopy();
            intComp.bitwiseAndInPlace(matrix, s);
            return matrix;
        }

        public static intMxN operator &(in int s, in intMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator |(in intMxN a, in int s) {
            intMxN matrix = a.TempCopy();
            intComp.bitwiseOrInPlace(matrix, s);
            return matrix;
        }

        public static intMxN operator |(in int s, in intMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator ^(in intMxN a, in int s) {
            intMxN matrix = a.TempCopy();
            intComp.bitwiseXorInPlace(matrix, s);
            return matrix;
        }

        public static intMxN operator ^(in int s, in intMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator <<(in intMxN a, int shift) {
            intMxN matrix = a.TempCopy();
            intComp.bitwiseLeftShiftInPlace(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator >>(in intMxN a, int shift) {
            intMxN matrix = a.TempCopy();
            intComp.bitwiseRightShiftInPlace(matrix, shift);
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

            intComp.addInPlace(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

            return matrix;
        }

        /// <summary>Component-wise subtraction; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator -(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            intMxN matrix = lhs.TempCopy();

            intComp.subInPlace(matrix, rhs);

            return matrix;
        }

        /// <summary>Component-wise multiplication; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator *(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            intMxN matrix = lhs.TempCopy();

            intComp.mulInPlace(rhs, matrix);

            return matrix;
        }

        /// <summary>Component-wise division; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator /(in intMxN dividend, in intMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            intMxN newDividendMatrix = dividend.TempCopy();

            intComp.divInPlace(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>Component-wise modulo; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator %(in intMxN dividend, in intMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            intComp.modInPlace(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator &(in intMxN a, in intMxN b) {
            
            Assume.SameDim(in a, in b);

            intMxN matrix = a.TempCopy();
            intComp.bitwiseAndInPlace(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator |(in intMxN a, in intMxN b) {

            Assume.SameDim(in a, in b);

            intMxN matrix = a.TempCopy();
            intComp.bitwiseOrInPlace(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN operator ^(in intMxN a, in intMxN b) {

            Assume.SameDim(in a, in b);

            intMxN matrix = a.TempCopy();
            intComp.bitwiseXorInPlace(matrix, b);
            return matrix;
        }

        #endregion
    }
}