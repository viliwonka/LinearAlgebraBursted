using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct doubleMxN : IDisposable, IUnsafedoubleArray {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator +(in doubleMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator -(in doubleMxN a)
        {
            doubleMxN matrix = a.TempCopy();
            
            doubleComp.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator +(in doubleMxN lhs, double rhs)
        {
            doubleMxN matrix = lhs.TempCopy();
            
            doubleComp.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator +(double lhs, in doubleMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator -(in doubleMxN lhs, double rhs)
        {
            doubleMxN matrix = lhs.TempCopy();
            
            doubleComp.addInpl(matrix, -rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator -(double lhs, in doubleMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            doubleMxN matrix = rhs.TempCopy();
            doubleComp.subInpl(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator *(in doubleMxN a, double s)
        {
            doubleMxN matrix = a.TempCopy();

            doubleComp.mulInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator *(double lhs, in doubleMxN rhs) => rhs * lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator /(in doubleMxN a, double s)
        {
            doubleMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            doubleComp.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator /(double s, in doubleMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); division by zero MATRIX entries is left to IEEE (Inf/NaN).
            doubleMxN matrix = a.TempCopy();
            doubleComp.divInpl(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator %(in doubleMxN a, double s)
        {
            doubleMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            doubleComp.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator %(double s, in doubleMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry is left to IEEE / runtime semantics.
            doubleMxN matrix = a.TempCopy();
            doubleComp.modInpl(s, matrix);

            return matrix;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator +(in doubleMxN lhs, in doubleMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            doubleMxN matrix = lhs.TempCopy();

            doubleComp.addInpl(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

            return matrix;
        }

        /// <summary>Component-wise subtraction; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator -(in doubleMxN lhs, in doubleMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            doubleMxN matrix = lhs.TempCopy();

            doubleComp.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>Component-wise multiplication; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator *(in doubleMxN lhs, in doubleMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            doubleMxN matrix = lhs.TempCopy();

            doubleComp.mulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>Component-wise division; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator /(in doubleMxN dividend, in doubleMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            doubleMxN newDividendMatrix = dividend.TempCopy();

            doubleComp.divInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>Component-wise modulo; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator %(in doubleMxN dividend, in doubleMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            doubleComp.modInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        #endregion
    }
}