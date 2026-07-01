using System.Runtime.CompilerServices;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{

    public partial struct intN {

        #region SCALAR OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <(in intN lhs, int rhs) {

            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe {
                UnsafeBool_OP.cmprLessScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res; 
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <(int lhs, in intN rhs) => rhs > lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >(in intN lhs, int rhs) {
            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe {
                UnsafeBool_OP.cmprGreaterScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >(int lhs, in intN rhs) => rhs < lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <=(in intN lhs, int rhs)
        {
            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBool_OP.cmprLessOrEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <=(int lhs, in intN rhs) => rhs >= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >=(in intN lhs, int rhs)
        {
            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBool_OP.cmprGreaterOrEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >=(int lhs, in intN rhs) => rhs <= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator ==(in intN lhs, int rhs)
        {
            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBool_OP.cmprEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator ==(int lhs, in intN rhs) => rhs == lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator !=(in intN lhs, int rhs)
        {
            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBool_OP.cmprNotEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator !=(int lhs, in intN rhs) => rhs != lhs;

        #endregion

        #region COMPONENT-WISE OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <(in intN lhs, in intN rhs) {

            Assume.SameDim(in lhs, in rhs);

            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBool_OP.cmprLess(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >(in intN a, in intN b) => b < a;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <=(in intN lhs, in intN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBool_OP.cmprLessOrEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >=(in intN lhs, in intN rhs) => rhs <= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator ==(in intN lhs, in intN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBool_OP.cmprEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator !=(in intN lhs, in intN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBool_OP.cmprNotEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.N);
            }

            return res;
        }

        #endregion

    }
}