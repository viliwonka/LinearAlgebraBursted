
namespace LinearAlgebra
{

    public partial struct boolMxN {

        #region SCALAR OPERATIONS
        public static boolMxN operator |(in boolMxN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            boolComp.orInPlace(vec, rhs);
            return vec;
        }

        public static boolMxN operator |(bool lhs, in boolMxN rhs) => rhs | lhs;

        public static boolMxN operator &(in boolMxN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            boolComp.andInPlace(vec, rhs);
            return vec;
        }

        public static boolMxN operator &(bool lhs, in boolMxN rhs) => rhs & lhs;

        public static boolMxN operator ^(in boolMxN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            boolComp.xorInPlace(vec, rhs);
            return vec;
        }

        public static boolMxN operator ^(bool lhs, in boolMxN rhs) => rhs ^ lhs;

        public static boolMxN operator ==(in boolMxN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            boolComp.equalsInPlace(vec, rhs);
            return vec;
        }

        public static boolMxN operator ==(bool lhs, in boolMxN rhs) => rhs == lhs;

        public static boolMxN operator !=(in boolMxN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            boolComp.equalsInPlace(vec, !rhs);
            return vec;
        }

        public static boolMxN operator !=(bool lhs, in boolMxN rhs) => rhs != lhs;
        #endregion

        #region UNITARY OPERATIONS
        public static boolMxN operator !(in boolMxN lhs)
        {
            var vec = lhs.TempCopy();

            boolComp.notInPlace(vec);

            return vec;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS

        public static boolMxN operator |(in boolMxN lhs, boolMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            boolComp.orInPlace(vec, rhs);
            return vec;
        }

        public static boolMxN operator &(in boolMxN lhs, boolMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            boolComp.andInPlace(vec, rhs);
            return vec;
        }

        public static boolMxN operator ^(in boolMxN lhs, boolMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            boolComp.xorInPlace(vec, rhs);
            return vec;
        }

        public static boolMxN operator ==(in boolMxN lhs, boolMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            boolComp.equalsInPlace(vec, rhs);
            return vec;
        }

        public static boolMxN operator !=(in boolMxN lhs, boolMxN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            boolComp.notEqualsInPlace(vec, rhs);
            return vec;
        }
        #endregion
    }
}