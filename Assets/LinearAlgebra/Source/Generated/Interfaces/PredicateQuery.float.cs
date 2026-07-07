namespace LinearAlgebra
{
    /// <summary>
    /// Row predicate for struct-functor predicate queries. Reads A[row, 0..N_Cols-1]
    /// directly — zero-alloc, no row extraction. Implement on a blittable struct;
    /// pass by <c>ref</c> to preserve any mutable state across calls.
    /// </summary>
    public interface IfloatRowPredicate {
        bool Test(in floatMxN A, int row);
    }

    /// <summary>
    /// Column predicate: symmetric twin of <see cref="IfloatRowPredicate"/>.
    /// Reads A[0..M_Rows-1, col] with stride. Zero-alloc, no column extraction.
    /// </summary>
    public interface IfloatColPredicate {
        bool Test(in floatMxN A, int col);
    }

    /// <summary>
    /// Scalar / elementwise predicate for flat <see cref="IUnsafefloatArray"/> data.
    /// Used by the generic Group-A ops: findFirst, count, any, all, findAll.
    /// </summary>
    public interface IfloatPredicate {
        bool Test(float x);
    }

    /// <summary>
    /// Row-score functor: returns a scalar score for row r of A.
    /// Used by Group-D score-based selection: argMaxRowBy, argMinRowBy, topKRowsBy.
    /// Higher score = better for argMaxRowBy / topKRowsBy.
    /// </summary>
    public interface IfloatRowScore {
        float Score(in floatMxN A, int row);
    }

    /// <summary>
    /// Column-score functor: symmetric twin of <see cref="IfloatRowScore"/>.
    /// Used by argMaxColBy, argMinColBy, topKColsBy.
    /// </summary>
    public interface IfloatColScore {
        float Score(in floatMxN A, int col);
    }
}
