using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct shortN {

        #region SCALAR OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <(in shortN lhs, short rhs) {

            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe {
                UnsafeBoolOP.cmprLessScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res; 
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <(short lhs, in shortN rhs) => rhs > lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >(in shortN lhs, short rhs) {
            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe {
                UnsafeBoolOP.cmprGreaterScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >(short lhs, in shortN rhs) => rhs < lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <=(in shortN lhs, short rhs)
        {
            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprLessOrEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <=(short lhs, in shortN rhs) => rhs >= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >=(in shortN lhs, short rhs)
        {
            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprGreaterOrEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >=(short lhs, in shortN rhs) => rhs <= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator ==(in shortN lhs, short rhs)
        {
            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator ==(short lhs, in shortN rhs) => rhs == lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator !=(in shortN lhs, short rhs)
        {
            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprNotEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator !=(short lhs, in shortN rhs) => rhs != lhs;

        #endregion

        #region COMPONENT-WISE OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <(in shortN lhs, in shortN rhs) {

            Assume.SameDim(in lhs, in rhs);

            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprLess(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >(in shortN a, in shortN b) => b < a;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <=(in shortN lhs, in shortN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprLessOrEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >=(in shortN lhs, in shortN rhs) => rhs <= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator ==(in shortN lhs, in shortN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator !=(in shortN lhs, in shortN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolN res = lhs.tempBoolVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprNotEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.N);
            }

            return res;
        }

        #endregion

    }
}