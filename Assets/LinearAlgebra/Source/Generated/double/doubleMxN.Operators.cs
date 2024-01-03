using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct doubleMxN : IDisposable, IUnsafedoubleArray {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator +(in doubleMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator -(in doubleMxN a)
        {
            doubleMxN matrix = a.TempCopy();
            
            doubleOP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator +(in doubleMxN lhs, double rhs)
        {
            doubleMxN matrix = lhs.TempCopy();
            
            doubleOP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator +(double lhs, in doubleMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator -(in doubleMxN lhs, double rhs)
        {
            doubleMxN matrix = lhs.TempCopy();
            
            doubleOP.addInpl(matrix, -rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator -(double lhs, in doubleMxN rhs)
        {
            return rhs - lhs;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator *(in doubleMxN a, double s)
        {
            doubleMxN matrix = a.TempCopy();

            doubleOP.mulInpl(matrix, s);

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

            doubleOP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator /(double s, in doubleMxN a)
        {
            doubleMxN matrix = a.TempCopy();
            
            if (s == 0f)
                throw new DivideByZeroException();

            doubleOP.divInpl(s, matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator %(in doubleMxN a, double s)
        {
            doubleMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            doubleOP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator %(double s, in doubleMxN a)
        {
            doubleMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            doubleOP.modInpl(s, matrix);

            return matrix;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>
        /// Component-wise addition
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator +(in doubleMxN lhs, in doubleMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            doubleMxN matrix = lhs.TempCopy();

            doubleOP.addInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>
        /// Component-wise subtraction
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator -(in doubleMxN lhs, in doubleMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            doubleMxN matrix = lhs.TempCopy();

            doubleOP.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>
        /// Component-wise multiplication
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator *(in doubleMxN lhs, in doubleMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            doubleMxN matrix = lhs.TempCopy();

            doubleOP.compMulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>
        /// Component-wise division
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator /(in doubleMxN dividend, in doubleMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            doubleMxN newDividendMatrix = dividend.TempCopy();

            doubleOP.compDivInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>
        /// Component-wise modulo
        /// Matrixs have to be same dimensions
        /// </summary>
        /// <returns>Same dimension Matrix</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN operator %(in doubleMxN dividend, in doubleMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            doubleOP.compModDiv(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        #endregion
    }
}