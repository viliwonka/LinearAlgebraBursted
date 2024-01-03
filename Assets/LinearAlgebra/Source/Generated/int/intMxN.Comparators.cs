using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct intMxN
    {
        #region SCALAR OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator <(in intMxN lhs, int rhs)
        {
            boolMxN res = lhs.tempBoolMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBoolOP.cmprLessScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator <(int lhs, in intMxN rhs) => rhs > lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator >(in intMxN lhs, int rhs)
        {
            boolMxN res = lhs.tempBoolMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBoolOP.cmprGreaterScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator >(int lhs, in intMxN rhs) => rhs < lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator <=(in intMxN lhs, int rhs)
        {
            
            boolMxN res = lhs.tempBoolMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBoolOP.cmprLessOrEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator <=(int lhs, in intMxN rhs) => rhs >= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator >=(in intMxN lhs, int rhs)
        {
            boolMxN res = lhs.tempBoolMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBoolOP.cmprGreaterOrEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator >=(int lhs, in intMxN rhs) => rhs <= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator ==(in intMxN lhs, int rhs)
        {
            boolMxN res = lhs.tempBoolMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBoolOP.cmprEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator ==(int lhs, in intMxN rhs) => rhs == lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator !=(in intMxN lhs, int rhs)
        {
            boolMxN res = lhs.tempBoolMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBoolOP.cmprNotEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator !=(int lhs, in intMxN rhs) => rhs != lhs;

        #endregion

        #region COMPONENT-WISE OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator <(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolMxN res = lhs.tempBoolMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBoolOP.cmprLess(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator >(in intMxN a, in intMxN b) => b < a;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator <=(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolMxN res = lhs.tempBoolMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBoolOP.cmprLessOrEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator >=(in intMxN lhs, in intMxN rhs) => rhs <= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator ==(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolMxN res = lhs.tempBoolMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBoolOP.cmprEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator !=(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolMxN res = lhs.tempBoolMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBoolOP.cmprNotEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.Length);
            }

            return res;
        }

        #endregion
    }
}