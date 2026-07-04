using System.Runtime.CompilerServices;
using LinearAlgebra.Internal;


namespace LinearAlgebra
{

    public partial struct uintN {

        #region SCALAR OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <(in uintN lhs, uint rhs) {

            boolN res = lhs.boolTempVec(lhs.N, true);

            unsafe {
                UnsafeBoolOP.cmprLessScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res; 
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <(uint lhs, in uintN rhs) => rhs > lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >(in uintN lhs, uint rhs) {
            boolN res = lhs.boolTempVec(lhs.N, true);

            unsafe {
                UnsafeBoolOP.cmprGreaterScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >(uint lhs, in uintN rhs) => rhs < lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <=(in uintN lhs, uint rhs)
        {
            boolN res = lhs.boolTempVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprLessOrEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <=(uint lhs, in uintN rhs) => rhs >= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >=(in uintN lhs, uint rhs)
        {
            boolN res = lhs.boolTempVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprGreaterOrEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >=(uint lhs, in uintN rhs) => rhs <= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator ==(in uintN lhs, uint rhs)
        {
            boolN res = lhs.boolTempVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator ==(uint lhs, in uintN rhs) => rhs == lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator !=(in uintN lhs, uint rhs)
        {
            boolN res = lhs.boolTempVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprNotEqualScalar(lhs.Data.Ptr, rhs, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator !=(uint lhs, in uintN rhs) => rhs != lhs;

        #endregion

        #region PREDICATES
        /// <summary>Componentwise power-of-two test (mirrors Unity.Mathematics' math.ispow2): true
        /// where an element's bit pattern has exactly one bit set AND the element is positive (0 and
        /// negative values are never a power of two). See UnsafeBoolOP.uint.cs's ispow2 kernel for
        /// the per-type semantics - short/long need a hand-written formula since Unity.Mathematics
        /// defines no short/long overload for math.ispow2.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public boolN ispow2()
        {
            boolN res = this.boolTempVec(this.N, true);

            unsafe
            {
                UnsafeBoolOP.ispow2(this.Data.Ptr, res.Data.Ptr, this.N);
            }

            return res;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <(in uintN lhs, in uintN rhs) {

            Assume.SameDim(in lhs, in rhs);

            boolN res = lhs.boolTempVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprLess(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >(in uintN a, in uintN b) => b < a;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator <=(in uintN lhs, in uintN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolN res = lhs.boolTempVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprLessOrEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator >=(in uintN lhs, in uintN rhs) => rhs <= lhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator ==(in uintN lhs, in uintN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolN res = lhs.boolTempVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.N);
            }

            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static boolN operator !=(in uintN lhs, in uintN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            boolN res = lhs.boolTempVec(lhs.N, true);

            unsafe
            {
                UnsafeBoolOP.cmprNotEqual(lhs.Data.Ptr, rhs.Data.Ptr, res.Data.Ptr, lhs.N);
            }

            return res;
        }

        #endregion

    }
}