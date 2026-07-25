using Unity.Mathematics;

namespace BULA
{
    public static partial class Krylov {

        // ================= Single-RHS verify-at-exit / final-residual core =================
        // Shared by cg, fcg, minres (two sites), minresQLP, biCGStab (one of its two sites), idr
        // (two sites), gmres, fgmres, gcrodr (one site each, post-back-substitution), and tfqmr
        // (verify-then-report, no idle scratch to fall through on): recompute a FRESH true residual
        // from A and x rather than trust a tracked/estimated residual. Callers own the convergence
        // decision -- some fall through and keep iterating on a failed verify, others use it
        // unconditionally as a final report -- so this returns the squared norm and bakes in no
        // return/branch of its own.
        //
        // biCGStab's TRIAL-x verify site (checks a not-yet-committed x before the stabilization
        // step, sign-flipped into A·x_trial - b) and every scalar Golub-Kahan / block verify site
        // are NOT this shape and are not routed here -- see the file DEVLOG.

        // Recomputes Ax = A·x and r = b - Ax (both caller-owned scratch, fully overwritten), and
        // returns ‖r‖². Ax and r must be distinct from each other and from b/x.
        static fProxy VerifyTrueResidual<TOp>(in TOp A, in fProxyN b, in fProxyN x, ref fProxyN Ax, ref fProxyN r)
            where TOp : struct, IfProxyLinearOperator
        {
            A.Apply(in x, ref Ax);
            r.CopyFrom(in b);
            r.addScaledInPlace((fProxy)(-1), Ax);
            return Blas.dot(r, r);
        }
    }
}
