namespace LinearAlgebra
{
    // Allocating (arena) wrappers for fProxyQueryOP search operations.
    // Zero-alloc ref-dest primitives (distancesToRow/Column) are in QueryOP;
    // these wrappers allocate from the matrix's own arena so callers don't
    // have to size buffers manually for the fProxy-typed outputs.
    //
    // intN-returning wrappers (nonzero, rowsWithinRadius, kNearestRows, WhichTrue)
    // live in Source/OP/QueryOP.Indices.cs (hand-maintained, cross-type).
    public static partial class ArenaExtensions
    {
        // -------------------------------------------------------------------------
        // distancesToRow / distancesToColumn — allocate fProxyN from A's arena
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates a fresh fProxyN (length A.M_Rows) from A's arena and fills it
        /// with the distance/similarity from each row of A to query q under metric m.
        /// </summary>
        public static fProxyN fProxyDistancesToRow(in fProxyMxN A, in fProxyN q, Metric m)
        {
            var dest = A.fProxyVec(A.M_Rows);
            fProxyQueryOP.distancesToRow(in A, in q, m, ref dest);
            return dest;
        }

        /// <summary>
        /// Allocates a fresh fProxyN (length A.N_Cols) from A's arena and fills it
        /// with the distance/similarity from each column of A to query q under metric m.
        /// </summary>
        public static fProxyN fProxyDistancesToColumn(in fProxyMxN A, in fProxyN q, Metric m)
        {
            var dest = A.fProxyVec(A.N_Cols);
            fProxyQueryOP.distancesToColumn(in A, in q, m, ref dest);
            return dest;
        }
    }
}
