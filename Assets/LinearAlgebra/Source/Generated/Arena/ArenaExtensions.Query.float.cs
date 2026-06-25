namespace LinearAlgebra
{
    // Allocating (arena) wrappers for floatQueryOP search operations.
    // Zero-alloc ref-dest primitives (distancesToRow/Column) are in QueryOP;
    // these wrappers allocate from the matrix's own arena so callers don't
    // have to size buffers manually for the float-typed outputs.
    //
    // intN-returning wrappers (nonzero, rowsWithinRadius, kNearestRows, WhichTrue)
    // live in Source/OP/QueryOP.Indices.cs (hand-maintained, cross-type).
    public static partial class ArenaExtensions
    {
        // -------------------------------------------------------------------------
        // distancesToRow / distancesToColumn — allocate floatN from A's arena
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates a fresh floatN (length A.M_Rows) from A's arena and fills it
        /// with the distance/similarity from each row of A to query q under metric m.
        /// </summary>
        public static floatN floatDistancesToRow(in floatMxN A, in floatN q, Metric m)
        {
            var dest = A.floatVec(A.M_Rows);
            floatQueryOP.distancesToRow(in A, in q, m, ref dest);
            return dest;
        }

        /// <summary>
        /// Allocates a fresh floatN (length A.N_Cols) from A's arena and fills it
        /// with the distance/similarity from each column of A to query q under metric m.
        /// </summary>
        public static floatN floatDistancesToColumn(in floatMxN A, in floatN q, Metric m)
        {
            var dest = A.floatVec(A.N_Cols);
            floatQueryOP.distancesToColumn(in A, in q, m, ref dest);
            return dest;
        }
    }
}
