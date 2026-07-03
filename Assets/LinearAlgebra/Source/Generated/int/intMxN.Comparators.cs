using System.Runtime.CompilerServices;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{

    public partial struct intMxN
    {
        #region SCALAR OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator <(in intMxN lhs, int rhs)
        {
            boolMxN res = lhs.boolTempMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBool_OP.cmprLessScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator <(int lhs, in intMxN rhs) => rhs > lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator >(in intMxN lhs, int rhs)
        {
            boolMxN res = lhs.boolTempMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBool_OP.cmprGreaterScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator >(int lhs, in intMxN rhs) => rhs < lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator <=(in intMxN lhs, int rhs)
        {
            
            boolMxN res = lhs.boolTempMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBool_OP.cmprLessOrEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator <=(int lhs, in intMxN rhs) => rhs >= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator >=(in intMxN lhs, int rhs)
        {
            boolMxN res = lhs.boolTempMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBool_OP.cmprGreaterOrEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator >=(int lhs, in intMxN rhs) => rhs <= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator ==(in intMxN lhs, int rhs)
        {
            boolMxN res = lhs.boolTempMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBool_OP.cmprEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator ==(int lhs, in intMxN rhs) => rhs == lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator !=(in intMxN lhs, int rhs)
        {
            boolMxN res = lhs.boolTempMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBool_OP.cmprNotEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.Length);
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

            boolMxN res = lhs.boolTempMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBool_OP.cmprLess(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator >(in intMxN a, in intMxN b) => b < a;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator <=(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolMxN res = lhs.boolTempMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBool_OP.cmprLessOrEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator >=(in intMxN lhs, in intMxN rhs) => rhs <= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator ==(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolMxN res = lhs.boolTempMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBool_OP.cmprEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.Length);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolMxN operator !=(in intMxN lhs, in intMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolMxN res = lhs.boolTempMat(lhs.M_Rows, lhs.N_Cols, true);

            unsafe
            {
                UnsafeBool_OP.cmprNotEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.Length);
            }

            return res;
        }

        #endregion
    }
}