
namespace BULA
{

    public partial struct boolMxN {

        #region SCALAR OPERATIONS
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

        #region COMPONENT-WISE OPERATIONS

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