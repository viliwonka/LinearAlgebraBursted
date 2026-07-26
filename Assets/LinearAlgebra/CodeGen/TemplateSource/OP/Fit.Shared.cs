using Unity.Collections;
using Unity.Mathematics;

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

        /// <summary>Cap on RANSAC draws, and the budget used when the adaptive rule cannot bound itself.</summary>
        public const int DefaultRansacIter = 2000;

        // Attempt cap for the rejection samplers. Their bounds are tight enough that acceptance is a
        // healthy constant fraction, so this is not reachable on a well-formed shape; it exists so a
        // degenerate one (a torus whose MinorRadius swallows its MajorRadius, an ellipsoid with a
        // collapsed axis) cannot spin forever inside a job.
        internal const int SampleTries = 64;

        // Draws m DISTINCT indices in [0, n). Rejection sampling: m is tiny (2-4) next to n in every
        // real use, so collisions are rare; the bounded retry keeps a pathological draw from spinning.
        internal static bool DrawDistinct(ref Random rng, int n, int m, ref NativeArray<int> idx)
        {
            for (int j = 0; j < m; j++)
            {
                bool placed = false;
                for (int attempt = 0; attempt < 32 && !placed; attempt++)
                {
                    int c = rng.NextInt(0, n);
                    bool dup = false;
                    for (int k = 0; k < j && !dup; k++) if (idx[k] == c) dup = true;
                    if (!dup) { idx[j] = c; placed = true; }
                }
                if (!placed) return false;
            }
            return true;
        }

        // Iterations for `confidence` probability that at least one drawn sample was all-inlier:
        // N = log(1 - p) / log(1 - w^m), w the inlier ratio. Returns the cap when w is too small for
        // that to be meaningful.
        internal static int AdaptiveIterations(int inliers, int n, int m, double confidence)
        {
            if (inliers <= 0 || n <= 0) return DefaultRansacIter;
            if (confidence <= 0 || confidence >= 1) confidence = 0.99;

            double w = (double)inliers / n;
            double wm = math.pow(w, m);
            if (wm <= 0 || wm >= 1) return wm >= 1 ? 1 : DefaultRansacIter;

            double num = math.log(1.0 - confidence);
            double den = math.log(1.0 - wm);
            if (den >= 0) return DefaultRansacIter;

            double N = num / den;
            if (!(N > 0) || N > DefaultRansacIter) return DefaultRansacIter;
            return (int)math.ceil(N);
        }
    }

    /// <summary>
    /// Outcome of <see cref="Fit.ransac"/>. Implicitly converts to bool (== <see cref="found"/>).
    /// </summary>
    public struct RansacInfo
    {
        /// <summary>A consensus set of at least the minimal sample size was found.</summary>
        public bool found;
        /// <summary>Points within the threshold of the returned model.</summary>
        public int inliers;
        /// <summary>Draws actually performed -- below the budget when the adaptive rule stopped early.</summary>
        public int iterations;
        /// <summary>MSAC score of the returned model, sum of min(d², t²). LOWER is better.</summary>
        public double score;

        public static implicit operator bool(RansacInfo i) => i.found;

        public override string ToString()
            => $"RansacInfo(found={found}, inliers={inliers}, iterations={iterations}, score={score:G6})";
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
