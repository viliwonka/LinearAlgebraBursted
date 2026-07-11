using System;

using Unity.Collections;

namespace LinearAlgebra
{
    /// <summary>
    /// Warm-solve factor/weight persistence cache for <see cref="LP.solve(in fProxyMxN, in fProxyN, in fProxyN, in NativeArray{ConstraintSense}, ref fProxyN, out double, ref LPBasis, ref fProxyLPCache, int)"/>.
    /// Persists the dual simplex's computational form (M/lower/upper/cost/rhs) and basis
    /// factorization (B/P/eta) AND DSE pricing weights across separate top-level solve calls, so a
    /// re-solve against an unchanged constraint structure skips both BuildComputationalForm (O(mN)
    /// copy) and Refactorize (O(m^3) LU).
    ///
    /// LIFECYCLE: standalone, user-allocated -- mirrors <see cref="LPBasis"/> (needs to persist across
    /// separate solve calls, not arena-scoped). <c>new fProxyLPCache(n, m, allocator)</c> +
    /// <see cref="Dispose"/>. <c>default</c> (or any not-yet-<see cref="IsCreated"/> instance) means "no
    /// cache": <see cref="LP.solve"/>'s cache-taking overload then behaves byte-identically to the plain
    /// <c>ref LPBasis</c> overload.
    ///
    /// INVALIDATION CONTRACT: <see cref="matrixVersion"/> is bumped BY THE CALLER (plain increment)
    /// whenever it changes the constraint matrix's structure (coefficients or per-row sense) or the
    /// objective <c>c</c> -- an rhs/bound-only change does not need a bump. <see cref="builtVersion"/> is
    /// LP.solve-owned: the <see cref="matrixVersion"/> value in effect when <see cref="M"/>/<see cref="B"/>/
    /// <see cref="P"/>/the eta file were last (re)built. A cache HIT requires
    /// <c>builtVersion == matrixVersion</c> AND <see cref="factorsValid"/>.
    ///
    /// BASIS-UNCHANGED DETECTION: no separate snapshot of <c>LPBasis.basis</c> is kept.
    /// <see cref="factorsValid"/>/<see cref="weightsValid"/> are trusted BY CONTRACT instead: the SAME
    /// (<see cref="LPBasis"/>, cache) pair must be threaded through every solve call in sequence (e.g.
    /// MIP.SearchCore: one basis + one cache, passed to every node/trial solve). A caller that
    /// hand-edits the basis directly, or interleaves solves against the same basis through a different
    /// cache (or none), must invalidate this one itself (<c>factorsValid = weightsValid = false</c>).
    ///
    /// VERIFICATION: under ENABLE_UNITY_COLLECTIONS_CHECKS, every cache HIT rebuilds the computational
    /// form into scratch and compares it entrywise against <see cref="M"/>/<see cref="lower"/>/
    /// <see cref="upper"/>/<see cref="cost"/>, throwing on any mismatch -- a caller that forgot to bump
    /// <see cref="matrixVersion"/> fails loudly in tests instead of silently solving the wrong problem.
    /// Compiled out (costs nothing) in release.
    /// </summary>
    public struct fProxyLPCache
    {
        /// <summary>Computational form, m x N (N = n + m). Rebuilt on a miss; unchanged (only
        /// <see cref="rhs"/> patched) on a hit.</summary>
        public fProxyMxN M;

        /// <summary>Length m. Re-copied from the caller's b every call, hit or miss (O(m), cheap) --
        /// rhs changes between calls even when the structure does not.</summary>
        public fProxyN rhs;

        /// <summary>Length N. Computational-form bounds -- depend only on A's structure/senses, so valid
        /// whenever <see cref="M"/> is.</summary>
        public fProxyN lower, upper;

        /// <summary>Length N. Computational-form cost.</summary>
        public fProxyN cost;

        /// <summary>m x m. LU factors of the basis matrix, in place -- exactly DualSimplexCore's local
        /// <c>B</c>, persisted.</summary>
        public fProxyMxN B;

        /// <summary>Partial-pivoting permutation paired with <see cref="B"/>.</summary>
        public Pivot P;

        /// <summary>REFACTOR_INTERVAL x m eta rows (product-form-of-the-inverse), persisted alongside
        /// <see cref="B"/>/<see cref="P"/>.</summary>
        public fProxyMxN etaAlpha;

        /// <summary>Length REFACTOR_INTERVAL. Leaving row per eta slot.</summary>
        public NativeArray<int> etaRow;

        /// <summary>Number of eta entries currently live (0..REFACTOR_INTERVAL).</summary>
        public int etaCount;

        /// <summary>Length m. Dual steepest-edge weights -- carried terminal state of the solve that
        /// last left the basis at its current content.</summary>
        public fProxyN weight;

        /// <summary>Caller-owned generation counter -- bump on every structural (A/senses/c) change.
        /// See the type's own doc comment for the full invalidation contract.</summary>
        public int matrixVersion;

        /// <summary><see cref="matrixVersion"/>'s value the moment <see cref="M"/>/<see cref="B"/>/
        /// <see cref="P"/>/the eta file were last (re)built -- LP.solve-owned, never written by the
        /// caller.</summary>
        public int builtVersion;

        /// <summary>True iff <see cref="B"/>/<see cref="P"/>/the eta file describe the basis matrix
        /// formed by <c>LPBasis.basis</c> as of <see cref="builtVersion"/> -- a cache hit resumes
        /// FTRAN/BTRAN from here instead of calling Refactorize.</summary>
        public bool factorsValid;

        /// <summary>True iff <see cref="weight"/> is the terminal DSE weight state of a solve that ended
        /// at the basis <c>LPBasis.basis</c> currently describes -- seeds pricing instead of the w=1
        /// approximation.</summary>
        public bool weightsValid;

        /// <summary>
        /// Allocates a cache sized for a computational form with <paramref name="n"/> structural
        /// variables and <paramref name="m"/> constraints (N = n + m). <see cref="factorsValid"/>/
        /// <see cref="weightsValid"/> start false, so the first <see cref="LP.solve"/> call using this
        /// cache always takes the cold path (and populates it).
        /// </summary>
        public fProxyLPCache(int n, int m, Allocator allocator)
        {
            int N = n + m;
            M = new fProxyMxN(m, N, allocator);
            rhs = new fProxyN(m, allocator);
            lower = new fProxyN(N, allocator);
            upper = new fProxyN(N, allocator);
            cost = new fProxyN(N, allocator);
            B = new fProxyMxN(m, m, allocator);
            P = new Pivot(m, allocator);
            etaAlpha = new fProxyMxN(LP.REFACTOR_INTERVAL, m, allocator);
            etaRow = new NativeArray<int>(LP.REFACTOR_INTERVAL, allocator);
            etaCount = 0;
            weight = new fProxyN(m, allocator);
            matrixVersion = 0;
            builtVersion = -1;       // never matches matrixVersion==0 -- first call always cold
            factorsValid = false;
            weightsValid = false;
        }

        /// <summary>True once every buffer is allocated. <c>default</c>/not-yet-constructed reads false
        /// -- <see cref="LP.solve"/>'s cache-taking overload treats that as "no cache".</summary>
        public bool IsCreated => etaRow.IsCreated;

        /// <summary>True iff created AND sized for exactly <paramref name="n"/> structural variables /
        /// <paramref name="m"/> constraints.</summary>
        public bool IsValid(int n, int m) => IsCreated && M.M_Rows == m && M.N_Cols == n + m;

        /// <summary>Releases every buffer. Safe on an empty/already-disposed instance.</summary>
        public void Dispose()
        {
            if (!IsCreated) return;
            M.Dispose(); rhs.Dispose(); lower.Dispose(); upper.Dispose(); cost.Dispose();
            B.Dispose(); P.Dispose(); etaAlpha.Dispose(); etaRow.Dispose(); weight.Dispose();
        }
    }
}
