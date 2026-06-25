namespace LinearAlgebra
{
    // Allocating (arena) wrappers for doubleQueryOP search operations.
    // Zero-alloc ref-dest primitives (distancesToRow/Column) are in QueryOP;
    // these wrappers allocate from the matrix's own arena so callers don't
    // have to size buffers manually for the double-typed outputs.
    //
    // intN-returning wrappers (nonzero, rowsWithinRadius, kNearestRows, WhichTrue)
    // live in Source/OP/QueryOP.Indices.cs (hand-maintained, cross-type).
    public static partial class ArenaExtensions
    {
        // -------------------------------------------------------------------------
        // distancesToRow / distancesToColumn — allocate doubleN from A's arena
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates a fresh doubleN (length A.M_Rows) from A's arena and fills it
        /// with the distance/similarity from each row of A to query q under metric m.
        /// </summary>
        public static doubleN doubleDistancesToRow(in doubleMxN A, in doubleN q, Metric m)
        {
            var dest = A.doubleVec(A.M_Rows);
            doubleQueryOP.distancesToRow(in A, in q, m, ref dest);
            return dest;
        }

        /// <summary>
        /// Allocates a fresh doubleN (length A.N_Cols) from A's arena and fills it
        /// with the distance/similarity from each column of A to query q under metric m.
        /// </summary>
        public static doubleN doubleDistancesToColumn(in doubleMxN A, in doubleN q, Metric m)
        {
            var dest = A.doubleVec(A.N_Cols);
            doubleQueryOP.distancesToColumn(in A, in q, m, ref dest);
            return dest;
        }
    }
}
