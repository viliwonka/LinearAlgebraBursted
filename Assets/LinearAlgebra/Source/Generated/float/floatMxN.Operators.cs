using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct floatMxN : IDisposable, IUnsafefloatArray {

        #region SCALAR OPERATIONS

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator +(in floatMxN a) => a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator -(in floatMxN a)
        {
            floatMxN matrix = a.TempCopy();
            
            floatElem_OP.signFlipInpl(matrix);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator +(in floatMxN lhs, float rhs)
        {
            floatMxN matrix = lhs.TempCopy();
            
            floatElem_OP.addInpl(matrix, rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator +(float lhs, in floatMxN rhs) => rhs + lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator -(in floatMxN lhs, float rhs)
        {
            floatMxN matrix = lhs.TempCopy();
            
            floatElem_OP.addInpl(matrix, -rhs);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator -(float lhs, in floatMxN rhs)
        {
            // subtraction is NOT commutative: lhs - rhs[i,j], not rhs[i,j] - lhs
            floatMxN matrix = rhs.TempCopy();
            floatElem_OP.subInpl(lhs, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator *(in floatMxN a, float s)
        {
            floatMxN matrix = a.TempCopy();

            floatElem_OP.mulInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator *(float lhs, in floatMxN rhs) => rhs * lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator /(in floatMxN a, float s)
        {
            floatMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            floatElem_OP.divInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator /(float s, in floatMxN a)
        {
            // 0 / M is valid (= 0 where M != 0); division by zero MATRIX entries is left to IEEE (Inf/NaN).
            floatMxN matrix = a.TempCopy();
            floatElem_OP.divInpl(s, matrix);
            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator %(in floatMxN a, float s)
        {
            floatMxN matrix = a.TempCopy();

            if (s == 0f)
                throw new DivideByZeroException();

            floatElem_OP.modInpl(matrix, s);

            return matrix;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator %(float s, in floatMxN a)
        {
            // 0 % M is valid (= 0 where M != 0); a zero MATRIX entry is left to IEEE / runtime semantics.
            floatMxN matrix = a.TempCopy();
            floatElem_OP.modInpl(s, matrix);

            return matrix;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS

        /// <summary>Component-wise addition; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator +(in floatMxN lhs, in floatMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            floatMxN matrix = lhs.TempCopy();

            floatElem_OP.addInpl(matrix, rhs);   // matrix += rhs  (matrix is the copy of lhs)

            return matrix;
        }

        /// <summary>Component-wise subtraction; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator -(in floatMxN lhs, in floatMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            floatMxN matrix = lhs.TempCopy();

            floatElem_OP.subInpl(matrix, rhs);

            return matrix;
        }

        /// <summary>Component-wise multiplication; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator *(in floatMxN lhs, in floatMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            floatMxN matrix = lhs.TempCopy();

            floatElem_OP.mulInpl(rhs, matrix);

            return matrix;
        }

        /// <summary>Component-wise division; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator /(in floatMxN dividend, in floatMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            floatMxN newDividendMatrix = dividend.TempCopy();

            floatElem_OP.divInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        /// <summary>Component-wise modulo; matrices must be the same dimensions.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatMxN operator %(in floatMxN dividend, in floatMxN divisor)
        {
            Assume.SameDim(in dividend, in divisor);

            var newDividendMatrix = dividend.TempCopy();

            floatElem_OP.modInpl(newDividendMatrix, divisor);
            return newDividendMatrix;
        }

        #endregion
    }
}