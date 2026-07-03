namespace LinearAlgebra
{

    // can optimize scalar bool operations by not computing (like vec & false is always false)

    public partial struct boolN {

        #region SCALAR OPERATIONS
        public static boolN operator |(in boolN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            boolComp.orInPlace(vec, rhs);
            return vec;
        }

        public static boolN operator |(bool lhs, in boolN rhs) => rhs | lhs;

        public static boolN operator &(in boolN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            boolComp.andInPlace(vec, rhs);
            return vec;
        }

        public static boolN operator &(bool lhs, in boolN rhs) => rhs & lhs;

        public static boolN operator ^(in boolN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            boolComp.xorInPlace(vec, rhs);
            return vec;
        }

        public static boolN operator ^(bool lhs, in boolN rhs) => rhs ^ lhs;

        public static boolN operator ==(in boolN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            boolComp.equalsInPlace(vec, rhs);
            return vec;
        }

        public static boolN operator ==(bool lhs, in boolN rhs) => rhs == lhs;

        public static boolN operator !=(in boolN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            boolComp.equalsInPlace(vec, !rhs);
            return vec;
        }

        public static boolN operator !=(bool lhs, in boolN rhs) => rhs != lhs;
        #endregion

        #region UNITARY OPERATIONS
        public static boolN operator !(in boolN lhs) {

            var vec = lhs.TempCopy();
            
            boolComp.notInPlace(vec);

            return vec;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS

        public static boolN operator |(in boolN lhs, boolN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            boolComp.orInPlace(vec, rhs);
            return vec;
        }

        public static boolN operator &(in boolN lhs, boolN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            boolComp.andInPlace(vec, rhs);
            return vec;
        }

        public static boolN operator ^(in boolN lhs, boolN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            boolComp.xorInPlace(vec, rhs);
            return vec;
        }

        public static boolN operator ==(in boolN lhs, boolN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            var vec = lhs.TempCopy();
            boolComp.equalsInPlace(vec, rhs);
            return vec;
        }

        public static boolN operator !=(in boolN lhs, boolN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            var vec = lhs.TempCopy();
            boolComp.notEqualsInPlace(vec, rhs);
            return vec;
        }
        #endregion

    }
}