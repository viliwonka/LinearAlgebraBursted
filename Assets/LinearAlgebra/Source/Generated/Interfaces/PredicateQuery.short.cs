namespace LinearAlgebra
{
    /// <summary>
    /// Scalar predicate for integer flat data (<see cref="IUnsafeshortArray"/>).
    /// Used by the short Group-A predicate ops: findFirst, count, any, all, findAll.
    /// Groups B, C, and D are fProxy-only; for integer matrix row/col filtering
    /// convert to the float or double variant.
    /// </summary>
    public interface IshortPredicate {
        bool Test(short x);
    }
}
