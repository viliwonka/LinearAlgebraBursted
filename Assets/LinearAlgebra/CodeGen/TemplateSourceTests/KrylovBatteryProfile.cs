namespace LinearAlgebra
{
    /// <summary>
    /// Tags a Krylov battery gallery matrix (what it structurally IS) and, on the solver side,
    /// what a solver family REQUIRES/FORBIDS of a matrix it is willing to run against. Two
    /// disjoint sub-groups:
    ///   KIND   (mutually exclusive per matrix): SPD, SymmetricIndefinite, Nonsymmetric.
    ///   SHAPE  (mutually exclusive per matrix): Square, Rectangular (+ Overdetermined /
    ///          Underdetermined as a Rectangular refinement).
    ///   MODIFIER (orthogonal, any combination): FullRank, RankDeficient, WellConditioned,
    ///          IllConditioned, Sparse (BSR-native vs dense literature-gallery).
    /// Every gallery matrix carries exactly one KIND flag, exactly one SHAPE flag (plus
    /// Overdetermined/Underdetermined when Rectangular), and any applicable MODIFIER flags.
    /// </summary>
    [System.Flags]
    public enum MatrixProfile : uint
    {
        None                 = 0,

        // KIND (exactly one per square matrix; rectangular matrices don't carry a KIND flag)
        SPD                  = 1 << 0,
        SymmetricIndefinite  = 1 << 1,
        Nonsymmetric         = 1 << 2,

        // SHAPE (exactly one of Square/Rectangular; Over/Under only set when Rectangular)
        Square               = 1 << 3,
        Rectangular          = 1 << 4,
        Overdetermined       = 1 << 5,
        Underdetermined      = 1 << 6,

        // MODIFIERS (orthogonal)
        FullRank             = 1 << 7,
        RankDeficient        = 1 << 8,
        WellConditioned      = 1 << 9,
        IllConditioned       = 1 << 10,
        Sparse               = 1 << 11,   // BSR-native gallery entry (unlocks the
                                           // preconditioned-convergence check; dense entries
                                           // never carry this flag)
    }

    /// <summary>Which preconditioner a solver invoker expects for the Sparse-only
    /// preconditioned-convergence check. Mirrors the symmetric/nonsymmetric routing
    /// PreconditionerBatteryTests already uses (BlockJacobi for cg-family, ILU0 for
    /// biCGStab/gmres/idr-family).</summary>
    public enum PreconditionerKind { None, SymmetricBSR, NonsymmetricBSR }

    /// <summary>Dense literature-gallery matrices driven by the Krylov battery, tagged via
    /// <see cref="GalleryProfiles.Of(GalleryDenseMatrix)"/>.</summary>
    public enum GalleryDenseMatrix
    {
        // SPD
        Laplacian1D_8, MinIJ_5, Pei5_2, Hilbert4, Pascal5, Lehmer5,
        // SymmetricIndefinite
        Fiedler5, Clement4, Rosser8,
        // Nonsymmetric (square)
        DenseNonsym20, ConvDiffDense40, Grcar8,
        // Rectangular (Overdetermined) -- least-squares family
        Lauchli3_05, Lauchli3_1e3,
        // Rectangular (Underdetermined) -- least-squares family
        WideRandom10x30,
        // Rank-deficient rectangular -- least-squares family
        RankDeficient20x10_Rank5,
        // Synthetic modifiers (Rand.*InPlace) -- clean, size-independent WellConditioned /
        // IllConditioned knobs the literature gallery doesn't give directly
        RandSPDWellCond20, RandSPDIllCond20,
    }

    /// <summary>Block-sparse (BSR) gallery matrices driven by the Krylov battery, tagged via
    /// <see cref="GalleryProfiles.Of(GalleryBSRMatrix)"/>.</summary>
    public enum GalleryBSRMatrix
    {
        Poisson2D_20x20,          // SPD, Sparse
        Laplacian2D_16x16,        // SPD, Sparse
        RandomSparseSPD_120_2,    // SPD, Sparse
        RandomSparseNonsym_80,    // Nonsymmetric, Sparse
    }

    /// <summary>Static tag lookup for every <see cref="GalleryDenseMatrix"/> / <see cref="GalleryBSRMatrix"/>
    /// entry -- pure enum-to-flags data, no matrix construction (see the templated
    /// fProxyKrylovBatteryGallery for that).</summary>
    public static class GalleryProfiles
    {
        public static MatrixProfile Of(GalleryDenseMatrix m)
        {
            switch (m)
            {
                case GalleryDenseMatrix.Laplacian1D_8: return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
                case GalleryDenseMatrix.MinIJ_5:        return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
                case GalleryDenseMatrix.Pei5_2:          return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
                case GalleryDenseMatrix.Hilbert4:        return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.IllConditioned;
                case GalleryDenseMatrix.Pascal5:         return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.IllConditioned;
                case GalleryDenseMatrix.Lehmer5:         return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;

                case GalleryDenseMatrix.Fiedler5:  return MatrixProfile.SymmetricIndefinite | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
                case GalleryDenseMatrix.Clement4:  return MatrixProfile.SymmetricIndefinite | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
                case GalleryDenseMatrix.Rosser8:   return MatrixProfile.SymmetricIndefinite | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.IllConditioned;

                case GalleryDenseMatrix.DenseNonsym20:   return MatrixProfile.Nonsymmetric | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
                case GalleryDenseMatrix.ConvDiffDense40: return MatrixProfile.Nonsymmetric | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
                case GalleryDenseMatrix.Grcar8:          return MatrixProfile.Nonsymmetric | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.IllConditioned;

                case GalleryDenseMatrix.Lauchli3_05:  return MatrixProfile.Rectangular | MatrixProfile.Overdetermined | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
                case GalleryDenseMatrix.Lauchli3_1e3: return MatrixProfile.Rectangular | MatrixProfile.Overdetermined | MatrixProfile.FullRank | MatrixProfile.IllConditioned;

                case GalleryDenseMatrix.WideRandom10x30: return MatrixProfile.Rectangular | MatrixProfile.Underdetermined | MatrixProfile.FullRank | MatrixProfile.WellConditioned;

                case GalleryDenseMatrix.RankDeficient20x10_Rank5: return MatrixProfile.Rectangular | MatrixProfile.Overdetermined | MatrixProfile.RankDeficient | MatrixProfile.WellConditioned;

                case GalleryDenseMatrix.RandSPDWellCond20: return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
                case GalleryDenseMatrix.RandSPDIllCond20:  return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.IllConditioned;

                default: return MatrixProfile.None;
            }
        }

        public static MatrixProfile Of(GalleryBSRMatrix m)
        {
            switch (m)
            {
                case GalleryBSRMatrix.Poisson2D_20x20:       return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned | MatrixProfile.Sparse;
                case GalleryBSRMatrix.Laplacian2D_16x16:     return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned | MatrixProfile.Sparse;
                case GalleryBSRMatrix.RandomSparseSPD_120_2: return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned | MatrixProfile.Sparse;
                case GalleryBSRMatrix.RandomSparseNonsym_80: return MatrixProfile.Nonsymmetric | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned | MatrixProfile.Sparse;
                default: return MatrixProfile.None;
            }
        }
    }

    /// <summary>Requires/Forbids refinement of "does this solver run on this matrix" -- a plain
    /// flags-intersection is wrong for the KIND group (e.g. Square alone would spuriously match
    /// cg against a Nonsymmetric matrix). A matrix is applicable iff it carries EVERY flag in
    /// <paramref name="requires"/> and NONE of the flags in <paramref name="forbids"/>.</summary>
    public static class MatrixProfileMatch
    {
        public static bool Applicable(MatrixProfile requires, MatrixProfile forbids, MatrixProfile matrixTags)
            => (matrixTags & requires) == requires && (matrixTags & forbids) == MatrixProfile.None;
    }
}
