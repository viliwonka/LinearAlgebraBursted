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
            
            doubleElem_OP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator +(in doubleMxN lhs, double rhs)
        {
            doubleMxN matrix = lhs.TempCopy();
            
            doubleElem_OP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator +(double lhs, in doubleMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator -(in doubleMxN lhs, double rhs)
        {
            doubleMxN matrix = lhs.TempCopy();
            
            doubleElem_OP.addInpl(matrix, -rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator -(double lhs, in doubleMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            doubleMxN matrix = rhs.TempCopy();
            doubleElem_OP.subInpl(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator *(in doubleMxN a, double s)
        {
            doubleMxN matrix = a.TempCopy();

            doubleElem_OP.mulInpl(matrix, s);

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

            doubleElem_OP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator /(double s, in doubleMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); division by zero MATRIX entries is left to IEEE (Inf/NaN).
            doubleMxN matrix = a.TempCopy();
            doubleElem_OP.divInpl(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator %(in doubleMxN a, double s)
        {
            doubleMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            doubleElem_OP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator %(double s, in doubleMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry is left to IEEE / runtime semantics.
            doubleMxN matrix = a.TempCopy();
            doubleElem_OP.modInpl(s, matrix);

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

            doubleElem_OP.addInpl(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

            return matrix;
        }

        /// <summary>Component-wise subtraction; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator -(in doubleMxN lhs, in doubleMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            doubleMxN matrix = lhs.TempCopy();

            doubleElem_OP.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>Component-wise multiplication; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator *(in doubleMxN lhs, in doubleMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            doubleMxN matrix = lhs.TempCopy();

            doubleElem_OP.mulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>Component-wise division; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator /(in doubleMxN dividend, in doubleMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            doubleMxN newDividendMatrix = dividend.TempCopy();

            doubleElem_OP.divInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>Component-wise modulo; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator %(in doubleMxN dividend, in doubleMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            doubleElem_OP.modInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        #endregion
    }
}