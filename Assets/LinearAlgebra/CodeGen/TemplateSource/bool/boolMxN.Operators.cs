
namespace LinearAlgebra
{

    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct boolMxN {

        #region SCALAR OPERATIONS
        public static boolMxN operator |(in boolMxN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            BoolOP.or(vec, rhs);
            return vec;
        }

        public static boolMxN operator |(bool lhs, in boolMxN rhs) => rhs | lhs;

        public static boolMxN operator &(in boolMxN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            BoolOP.and(vec, rhs);
            return vec;
        }

        public static boolMxN operator &(bool lhs, in boolMxN rhs) => rhs & lhs;

        public static boolMxN operator ^(in boolMxN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            BoolOP.xor(vec, rhs);
            return vec;
        }

        public static boolMxN operator ^(bool lhs, in boolMxN rhs) => rhs ^ lhs;

        public static boolMxN operator ==(in boolMxN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            BoolOP.equals(vec, rhs);
            return vec;
        }

        public static boolMxN operator ==(bool lhs, in boolMxN rhs) => rhs == lhs;

        public static boolMxN operator !=(in boolMxN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            BoolOP.equals(vec, !rhs);
            return vec;
        }

        public static boolMxN operator !=(bool lhs, in boolMxN rhs) => rhs != lhs;
        #endregion

        #region UNITARY OPERATIONS
        public static boolMxN operator !(in boolMxN lhs)
        {
            var vec = lhs.TempCopy();

            BoolOP.not(vec);

            return vec;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS

        public static boolMxN operator |(in boolMxN lhs, boolMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            BoolOP.or(vec, rhs);
            return vec;
        }

        public static boolMxN operator &(in boolMxN lhs, boolMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            BoolOP.and(vec, rhs);
            return vec;
        }

        public static boolMxN operator ^(in boolMxN lhs, boolMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            BoolOP.xor(vec, rhs);
            return vec;
        }

        public static boolMxN operator ==(in boolMxN lhs, boolMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            BoolOP.equals(vec, rhs);
            return vec;
        }

        public static boolMxN operator !=(in boolMxN lhs, boolMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            BoolOP.notEquals(vec, rhs);
            return vec;
        }
        #endregion
    }
}