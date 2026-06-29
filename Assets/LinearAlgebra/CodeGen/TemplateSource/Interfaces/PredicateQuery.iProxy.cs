namespace LinearAlgebra
{
    /// <summary>
    /// Scalar predicate for integer flat data (<see cref="IUnsafeiProxyArray"/>).
    /// Used by the iProxy Group-A predicate ops: findFirst, count, any, all, findAll.
    /// Groups B, C, and D are fProxy-only; for integer matrix row/col filtering
    /// convert to the float or double variant.
    /// </summary>
    public interface IiProxyPredicate {
        bool Test(iProxy x);
    }
}
