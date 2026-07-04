using System;
using System.Runtime.CompilerServices;


namespace LinearAlgebra
{

    public partial struct uintMxN {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator +(in uintMxN a) => a;

        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator +(in uintMxN lhs, uint rhs)
        {
            uintMxN matrix = lhs.TempCopy();

            uintComp.addInPlace(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator +(uint lhs, in uintMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator -(in uintMxN lhs, uint rhs)
        {
            uintMxN matrix = lhs.TempCopy();

            // v - s via a direct kernel, not v + (-s): the latter needs unary minus on the scalar,
            // which uint can't do (see OP.Component.uint.cs), so this line is identical for
            // every generated type.
            uintComp.subInPlace(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator -(uint lhs, in uintMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            uintMxN matrix = rhs.TempCopy();
            uintComp.subInPlace(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator *(in uintMxN a, uint s)
        {
            uintMxN matrix = a.TempCopy();

            uintComp.mulInPlace(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator *(uint lhs, in uintMxN rhs) => rhs * lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator /(in uintMxN a, uint s)
        {
            uintMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            uintComp.divInPlace(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator /(uint s, in uintMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer div by zero).
            uintMxN matrix = a.TempCopy();
            uintComp.divInPlace(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator %(in uintMxN a, uint s)
        {
            uintMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            uintComp.modInPlace(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator %(uint s, in uintMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer mod by zero).
            uintMxN matrix = a.TempCopy();
            uintComp.modInPlace(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator ~(in uintMxN a) {

            uintMxN matrix = a.TempCopy();

            uintComp.bitwiseComplementInPlace(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator &(in uintMxN a, in uint s) {

            uintMxN matrix = a.TempCopy();
            uintComp.bitwiseAndInPlace(matrix, s);
            return matrix;
        }

        public static uintMxN operator &(in uint s, in uintMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator |(in uintMxN a, in uint s) {
            uintMxN matrix = a.TempCopy();
            uintComp.bitwiseOrInPlace(matrix, s);
            return matrix;
        }

        public static uintMxN operator |(in uint s, in uintMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator ^(in uintMxN a, in uint s) {
            uintMxN matrix = a.TempCopy();
            uintComp.bitwiseXorInPlace(matrix, s);
            return matrix;
        }

        public static uintMxN operator ^(in uint s, in uintMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator <<(in uintMxN a, int shift) {
            uintMxN matrix = a.TempCopy();
            uintComp.bitwiseLeftShiftInPlace(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator >>(in uintMxN a, int shift) {
            uintMxN matrix = a.TempCopy();
            uintComp.bitwiseRightShiftInPlace(matrix, shift);
            return matrix;
        }

        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator +(in uintMxN lhs, in uintMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            uintMxN matrix = lhs.TempCopy();

            uintComp.addInPlace(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

            return matrix;
        }

        /// <summary>Component-wise subtraction; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator -(in uintMxN lhs, in uintMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            uintMxN matrix = lhs.TempCopy();

            uintComp.subInPlace(matrix, rhs);

            return matrix;
        }

        /// <summary>Component-wise multiplication; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator *(in uintMxN lhs, in uintMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            uintMxN matrix = lhs.TempCopy();

            uintComp.mulInPlace(rhs, matrix);

            return matrix;
        }

        /// <summary>Component-wise division; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator /(in uintMxN dividend, in uintMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            uintMxN newDividendMatrix = dividend.TempCopy();

            uintComp.divInPlace(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>Component-wise modulo; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator %(in uintMxN dividend, in uintMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            uintComp.modInPlace(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator &(in uintMxN a, in uintMxN b) {
            
            Assume.SameDim(in a, in b);

            uintMxN matrix = a.TempCopy();
            uintComp.bitwiseAndInPlace(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator |(in uintMxN a, in uintMxN b) {

            Assume.SameDim(in a, in b);

            uintMxN matrix = a.TempCopy();
            uintComp.bitwiseOrInPlace(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uintMxN operator ^(in uintMxN a, in uintMxN b) {

            Assume.SameDim(in a, in b);

            uintMxN matrix = a.TempCopy();
            uintComp.bitwiseXorInPlace(matrix, b);
            return matrix;
        }

        #endregion
    }
}