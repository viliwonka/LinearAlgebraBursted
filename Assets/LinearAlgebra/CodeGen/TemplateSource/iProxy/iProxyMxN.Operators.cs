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
            
            iProxyComp.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator +(in iProxyMxN lhs, iProxy rhs)
        {
            iProxyMxN matrix = lhs.TempCopy();
            
            iProxyComp.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator +(iProxy lhs, in iProxyMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator -(in iProxyMxN lhs, iProxy rhs)
        {
            iProxyMxN matrix = lhs.TempCopy();
            
            iProxyComp.addInpl(matrix, (iProxy)(-rhs));

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator -(iProxy lhs, in iProxyMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            iProxyMxN matrix = rhs.TempCopy();
            iProxyComp.subInpl(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator *(in iProxyMxN a, iProxy s)
        {
            iProxyMxN matrix = a.TempCopy();

            iProxyComp.mulInpl(matrix, s);

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

            iProxyComp.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator /(iProxy s, in iProxyMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer div by zero).
            iProxyMxN matrix = a.TempCopy();
            iProxyComp.divInpl(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator %(in iProxyMxN a, iProxy s)
        {
            iProxyMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            iProxyComp.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator %(iProxy s, in iProxyMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry still throws (integer mod by zero).
            iProxyMxN matrix = a.TempCopy();
            iProxyComp.modInpl(s, matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator ~(in iProxyMxN a) {

            iProxyMxN matrix = a.TempCopy();

            iProxyComp.bitwiseComplementInpl(matrix);

            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator &(in iProxyMxN a, in iProxy s) {

            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseAndInpl(matrix, s);
            return matrix;
        }

        public static iProxyMxN operator &(in iProxy s, in iProxyMxN a) => a & s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator |(in iProxyMxN a, in iProxy s) {
            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseOrInpl(matrix, s);
            return matrix;
        }

        public static iProxyMxN operator |(in iProxy s, in iProxyMxN a) => a | s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator ^(in iProxyMxN a, in iProxy s) {
            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseXorInpl(matrix, s);
            return matrix;
        }

        public static iProxyMxN operator ^(in iProxy s, in iProxyMxN a) => a ^ s;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator <<(in iProxyMxN a, int shift) {
            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseLeftShiftInpl(matrix, shift);
            return matrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator >>(in iProxyMxN a, int shift) {
            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseRightShiftInpl(matrix, shift);
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

            iProxyComp.addInpl(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

            return matrix;
        }

        /// <summary>Component-wise subtraction; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator -(in iProxyMxN lhs, in iProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            iProxyMxN matrix = lhs.TempCopy();

            iProxyComp.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>Component-wise multiplication; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator *(in iProxyMxN lhs, in iProxyMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            iProxyMxN matrix = lhs.TempCopy();

            iProxyComp.mulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>Component-wise division; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator /(in iProxyMxN dividend, in iProxyMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            iProxyMxN newDividendMatrix = dividend.TempCopy();

            iProxyComp.divInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>Component-wise modulo; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator %(in iProxyMxN dividend, in iProxyMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            iProxyComp.modInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator &(in iProxyMxN a, in iProxyMxN b) {
            
            Assume.SameDim(in a, in b);

            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseAndInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator |(in iProxyMxN a, in iProxyMxN b) {

            Assume.SameDim(in a, in b);

            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseOrInpl(matrix, b);
            return matrix;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static iProxyMxN operator ^(in iProxyMxN a, in iProxyMxN b) {

            Assume.SameDim(in a, in b);

            iProxyMxN matrix = a.TempCopy();
            iProxyComp.bitwiseXorInpl(matrix, b);
            return matrix;
        }

        #endregion
    }
}