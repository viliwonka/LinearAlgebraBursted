namespace BULA
{
    // Type-agnostic members of the Fit facade. `Fit` carries no per-type token, so every generated
    // per-dtype file partial-merges into this SAME class -- anything not parameterized by the element
    // type must live here once, or it collides as CS0102/CS0111 across the merged fragments. Same
    // arrangement as Hash.Shared.cs.
    //
    // Codegen hazard: this file must never contain the code generator's per-type placeholder
    // spellings, or it would be templated and reintroduce the collision it exists to prevent.
    public static partial class Fit
    {
        /// <summary>IRLS iteration budget used when a caller passes maxIter &lt;= 0.</summary>
        public const int DefaultIrlsIter = 50;
    }

    /// <summary>
    /// Surface family of a quadric, from the eigenvalue signature of its 3x3 quadratic form -- see
    /// <see cref="Fit.classify"/>. Cone and hyperboloid share a signature (two eigenvalues of one
    /// sign, one of the other) and are told apart only by the constant term after translating to the
    /// centre, which is not scale-invariant enough to decide reliably on fitted data; they are
    /// therefore reported together rather than guessed at.
    /// </summary>
    public enum QuadricKind
    {
        /// <summary>The eigensolve failed; nothing can be said about the surface.</summary>
        Unknown = 0,
        /// <summary>All three eigenvalues share a sign: an ellipsoid (or an imaginary one).</summary>
        Ellipsoid = 1,
        /// <summary>Mixed signs, no zero eigenvalue: a hyperboloid of one or two sheets, or a cone.</summary>
        HyperboloidOrCone = 2,
        /// <summary>A zero eigenvalue: no centre exists -- a paraboloid or a cylinder.</summary>
        Paraboloid = 3,
        /// <summary>No quadratic part survives: the fit collapsed to a plane or worse.</summary>
        Degenerate = 4,
    }
}
