namespace LinearAlgebra
{

    // can optimize scalar bool operations by not computing (like vec & false is always false)

    public partial struct boolN {

        #region SCALAR OPERATIONS
        public static boolN operator |(in boolN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            Bool_OP.or(vec, rhs);
            return vec;
        }

        public static boolN operator |(bool lhs, in boolN rhs) => rhs | lhs;

        public static boolN operator &(in boolN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            Bool_OP.and(vec, rhs);
            return vec;
        }

        public static boolN operator &(bool lhs, in boolN rhs) => rhs & lhs;

        public static boolN operator ^(in boolN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            Bool_OP.xor(vec, rhs);
            return vec;
        }

        public static boolN operator ^(bool lhs, in boolN rhs) => rhs ^ lhs;

        public static boolN operator ==(in boolN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            Bool_OP.equals(vec, rhs);
            return vec;
        }

        public static boolN operator ==(bool lhs, in boolN rhs) => rhs == lhs;

        public static boolN operator !=(in boolN lhs, bool rhs)
        {
            var vec = lhs.TempCopy();
            Bool_OP.equals(vec, !rhs);
            return vec;
        }

        public static boolN operator !=(bool lhs, in boolN rhs) => rhs != lhs;
        #endregion

        #region UNITARY OPERATIONS
        public static boolN operator !(in boolN lhs) {

            var vec = lhs.TempCopy();
            
            Bool_OP.not(vec);

            return vec;
        }
        #endregion

        #region COMPONENT-WISE OPERATIONS

        public static boolN operator |(in boolN lhs, boolN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            Bool_OP.or(vec, rhs);
            return vec;
        }

        public static boolN operator &(in boolN lhs, boolN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            Bool_OP.and(vec, rhs);
            return vec;
        }

        public static boolN operator ^(in boolN lhs, boolN rhs)
        {
            Assume.SameDim(in lhs, in rhs);

            var vec = lhs.TempCopy();
            Bool_OP.xor(vec, rhs);
            return vec;
        }

        public static boolN operator ==(in boolN lhs, boolN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            var vec = lhs.TempCopy();
            Bool_OP.equals(vec, rhs);
            return vec;
        }

        public static boolN operator !=(in boolN lhs, boolN rhs)
        {
            Assume.SameDim(in lhs, in rhs);
            
            var vec = lhs.TempCopy();
            Bool_OP.notEquals(vec, rhs);
            return vec;
        }
        #endregion

    }
}