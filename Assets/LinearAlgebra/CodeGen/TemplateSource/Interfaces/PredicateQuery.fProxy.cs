namespace BULA
{
    /// <summary>
    /// Row predicate for struct-functor predicate queries. Reads A[row, 0..N_Cols-1]
    /// directly — zero-alloc, no row extraction. Implement on a blittable struct;
    /// pass by <c>ref</c> to preserve any mutable state across calls.
    /// </summary>
    public interface IfProxyRowPredicate {
        bool Test(in fProxyMxN A, int row);
    }

    /// <summary>
    /// Column predicate: symmetric twin of <see cref="IfProxyRowPredicate"/>.
    /// Reads A[0..M_Rows-1, col] with stride. Zero-alloc, no column extraction.
    /// </summary>
    public interface IfProxyColPredicate {
        bool Test(in fProxyMxN A, int col);
    }

    /// <summary>
    /// Scalar / elementwise predicate for flat <see cref="IUnsafefProxyArray"/> data.
    /// Used by the generic Group-A ops: findFirst, count, any, all, findAll.
    /// </summary>
    public interface IfProxyPredicate {
        bool Test(fProxy x);
    }

    /// <summary>
    /// Row-score functor: returns a scalar score for row r of A.
    /// Used by Group-D score-based selection: argMaxRowBy, argMinRowBy, topKRowsBy.
    /// Higher score = better for argMaxRowBy / topKRowsBy.
    /// </summary>
    public interface IfProxyRowScore {
        fProxy Score(in fProxyMxN A, int row);
    }

    /// <summary>
    /// Column-score functor: symmetric twin of <see cref="IfProxyRowScore"/>.
    /// Used by argMaxColBy, argMinColBy, topKColsBy.
    /// </summary>
    public interface IfProxyColScore {
        fProxy Score(in fProxyMxN A, int col);
    }
}
