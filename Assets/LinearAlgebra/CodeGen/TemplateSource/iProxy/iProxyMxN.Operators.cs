using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct iProxyMxN {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator +(in iProxyMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator -(in iProxyMxN a)
        {
            iProxyMxN matrix = a.TempCopy();
            
            iProxyComp.signFlipInPlace(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator +(in iProxyMxN lhs, iProxy rhs)
        {
            iProxyMxN matrix = lhs.TempCopy();
            
            iProxyComp.addInPlace(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator +(iProxy lhs, in iProxyMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator -(in iProxyMxN lhs, iProxy rhs)
        {
            iProxyMxN matrix = lhs.TempCopy();
            
            iProxyComp.addInPlace(matrix, (iProxy)(-rhs));

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator -(iProxy lhs, in iProxyMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            iProxyMxN matrix = rhs.TempCopy();
            iProxyComp.subInPlace(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator *(in iProxyMxN a, iProxy s)
        {
            iProxyMxN matrix = a.TempCopy();

            iProxyComp.mulInPlace(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator *(iProxy lhs, in iProxyMxN rhs) => rhs * lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator /(in iProxyMxN a, iProxy s)
        {
            iProxyMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            iProxyComp.divInPlace(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator /(iProxy s, in iProxyMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer div by zero).
            iProxyMxN matrix = a.TempCopy();
            iProxyComp.divInPlace(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator %(in iProxyMxN a, iProxy s)
        {
            iProxyMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            iProxyComp.modInPlace(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator %(iProxy s, in iProxyMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer mod by zero).
            iProxyMxN matrix = a.TempCopy();
            iProxyComp.modInPlace(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator ~(in iProxyMxN a) {

            iProxyMxN matrix = a.TempCopy();

            iProxyComp.bitwiseComplementInPlace(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator &(in iProxyMxN a, in iProxy s) {

            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseAndInPlace(matrix, s);
            return matrix;
        }

        public static iProxyMxN operator &(in iProxy s, in iProxyMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator |(in iProxyMxN a, in iProxy s) {
            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseOrInPlace(matrix, s);
            return matrix;
        }

        public static iProxyMxN operator |(in iProxy s, in iProxyMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator ^(in iProxyMxN a, in iProxy s) {
            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseXorInPlace(matrix, s);
            return matrix;
        }

        public static iProxyMxN operator ^(in iProxy s, in iProxyMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator <<(in iProxyMxN a, int shift) {
            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseLeftShiftInPlace(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator >>(in iProxyMxN a, int shift) {
            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseRightShiftInPlace(matrix, shift);
            return matrix;
        }

        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator +(in iProxyMxN lhs, in iProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            iProxyMxN matrix = lhs.TempCopy();

            iProxyComp.addInPlace(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

            return matrix;
        }

        /// <summary>Component-wise subtraction; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator -(in iProxyMxN lhs, in iProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            iProxyMxN matrix = lhs.TempCopy();

            iProxyComp.subInPlace(matrix, rhs);

            return matrix;
        }

        /// <summary>Component-wise multiplication; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator *(in iProxyMxN lhs, in iProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            iProxyMxN matrix = lhs.TempCopy();

            iProxyComp.mulInPlace(rhs, matrix);

            return matrix;
        }

        /// <summary>Component-wise division; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator /(in iProxyMxN dividend, in iProxyMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            iProxyMxN newDividendMatrix = dividend.TempCopy();

            iProxyComp.divInPlace(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>Component-wise modulo; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator %(in iProxyMxN dividend, in iProxyMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            iProxyComp.modInPlace(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator &(in iProxyMxN a, in iProxyMxN b) {
            
            Assume.SameDim(in a, in b);

            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseAndInPlace(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator |(in iProxyMxN a, in iProxyMxN b) {

            Assume.SameDim(in a, in b);

            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseOrInPlace(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator ^(in iProxyMxN a, in iProxyMxN b) {

            Assume.SameDim(in a, in b);

            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseXorInPlace(matrix, b);
            return matrix;
        }

        #endregion
    }
}