using System;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    // ================================================================================================
    // Convex quadratic programming: a HiGHS-style dense primal null-space active-set method, ported per
    // Nocedal & Wright, Numerical Optimization (2nd ed.), ch. 16 ("Quadratic Programming"), 16.2 "The
    // Null-Space Method" and 16.5 "Active-Set Methods for Convex QP".
    //
    // This file: the EQUALITY-constrained QP core (EQP) -- a FIXED working set, the null-space Newton
    // step + multiplier recovery only (no add/drop loop, no ratio test, no phase 1).
    //
    //     minimize    ½xᵀQx + cᵀx   subject to   A_W x = b_W
    //     (A_W: k x n, k <= n, rows independent -- "the working set")
    //
    // Q is symmetric PSD (v1 contract): a singular reduced Hessian is regularized (δ·‖Q‖∞·I retry)
    // rather than handled via negative-curvature machinery. Indefinite Q is out of scope (NP-hard in
    // general). One null-space Newton step is exact for this problem (Nocedal & Wright eq.
    // 16.16-16.19): substituting x = x0 + Zy for an orthonormal null(A_W) basis Z reduces it to an
    // unconstrained quadratic in y, which Newton's method solves in one step from any start.
    //
    // The dense n x k QR factor Q1 is never materialized -- the null-space machinery drives QR's own
    // per-step primitives directly instead of calling QR.decompInPlace's public API. Z (n x (n-k)) IS
    // formed explicitly, since the reduced Hessian ZᵀQZ and the step p = Zy need it as a GEMM/GEMV
    // operand.
    //
    // Every function below is `internal static`, not a buried local, so a future inequality active-set
    // loop can call eqpNullSpaceStep (or its constituent pieces) once per iteration without
    // refactoring this file.
    // ================================================================================================
    public static partial class QP
    {
        // ============================================================================================
        // The PUBLIC FACADE -- QP.solve, mirroring LP.solve's doc voice, validation style, and layering
        // (validate -> phase 1 -> hand off to the internal core).
        //
        //   * Dimension/shape validation (ArgumentException, matching LP.solve's per-argument style)
        //     PLUS the v1 CONVEXITY CONTRACT itself (Q symmetric PSD) -- this is the one place in the
        //     whole QP stack that actually CHECKS symmetry (qpActiveSetCore and eqpNullSpaceStep both
        //     only ever READ Q via matrix products against both triangles, per their own doc comments,
        //     and never verify it): a cheap max|Q[i,j]-Q[j,i]| scan, scaled the same way every other
        //     tolerance in this file is (relative to ||Q||_inf). PSD itself is NOT checked (no cheap
        //     certificate exists short of a full factorization the solver would have to pay for
        //     anyway; a non-PSD Q surfaces indirectly through spurious Unbounded reports or a CHO retry
        //     that never stops regularizing -- out of v1 scope, matching HiGHS's own PSD assumption).
        //
        //   * Phase 1: PhaseOneFeasibleStart below finds ANY point satisfying A x {<=,=,>=} b,
        //     xl <= x <= xu via a zero-cost LP over the identical region (LP.solve, LPMethod.
        //     DualSimplex) -- see that function's own doc comment for the shift/split reformulation
        //     LP.solve's x>=0-only computational form requires. Anything other than LPStatus.Optimal
        //     from that LP maps straight to QPStatus.Infeasible.
        //
        // Neither piece touches qpActiveSetCore's own contract or tolerances -- the facade is purely an
        // outer layer, exactly like LP.solve is an outer layer over simplexCore/revisedSimplexCore/
        // dualSimplexCore/interiorCore.
        // ============================================================================================

        /// <summary>
        /// Solve the convex quadratic program  min ½xᵀQx + cᵀx  s.t.  A x {≤,=,≥} b (per-row
        /// <paramref name="senses"/>), xl ≤ x ≤ xu -- the public entry point, mirroring
        /// <see cref="LP.solve"/>'s doc voice and validation style. Q must be symmetric
        /// (checked here, cheaply -- see the facade header comment above) and positive semidefinite
        /// (the v1 convexity contract, NOT checked -- see the same comment); a genuinely non-PSD Q is
        /// out of scope (indefinite QP is NP-hard in general, matching HiGHS's own v1 assumption).
        ///
        /// Finds its own feasible starting point via a zero-cost LP over the same constraints+bounds
        /// (see <see cref="PhaseOneFeasibleStart"/>) -- <paramref name="x"/> is OUTPUT ONLY, its entry
        /// contents are ignored, matching <see cref="LP.solve"/>'s own "x: Output solution" convention
        /// (there is no warm-start overload in v1; qpActiveSetCore itself, reached via
        /// InternalsVisibleTo, is the seam a future warm-start entry point would call directly).
        /// </summary>
        /// <param name="Q">Symmetric PSD Hessian, n x n.</param>
        /// <param name="c">Linear cost, length n.</param>
        /// <param name="A">Constraint coefficients, m x n (m may be 0).</param>
        /// <param name="b">Right-hand sides, length m.</param>
        /// <param name="senses">Per-row constraint sense, length m.</param>
        /// <param name="xl">Variable lower bounds, length n. Use a large-magnitude negative sentinel
        /// (&lt;= -1e29) for a variable unbounded below -- or use the overload below that fills both
        /// bound arrays with +-infinity sentinels for you.</param>
        /// <param name="xu">Variable upper bounds, length n. Use a large-magnitude positive sentinel
        /// (&gt;= 1e29) for a variable unbounded above.</param>
        /// <param name="x">Output solution, length n (overwritten; entry contents ignored).</param>
        /// <param name="objective">Output ½xᵀQx + cᵀx at the returned x. 0 on
        /// <see cref="QPStatus.Infeasible"/> (no usable x -- matches <see cref="LPInfo"/>'s own
        /// convention).</param>
        /// <param name="maxIter">Pivot budget for the active-set loop; &lt;=0 picks a size-based
        /// default. Phase 1's own feasibility LP always uses its own size-based default, independent
        /// of this budget.</param>
        public static QPInfo solve(in fProxyMxN Q, in fProxyN c, in fProxyMxN A, in fProxyN b,
                                   in NativeArray<ConstraintSense> senses,
                                   in fProxyN xl, in fProxyN xu,
                                   ref fProxyN x, out double objective, int maxIter = 0)
        {
            int n = Q.M_Rows, m = A.M_Rows;

            if (!Q.IsSquare) throw new ArgumentException("QP.solve: Q must be square");
            if (c.N != n) throw new ArgumentException("QP.solve: c.N must equal Q.M_Rows");
            if (A.N_Cols != n) throw new ArgumentException("QP.solve: A.N_Cols must equal Q.M_Rows");
            if (b.N != m) throw new ArgumentException("QP.solve: b.N must equal A.M_Rows");
            if (senses.Length != m) throw new ArgumentException("QP.solve: senses.Length must equal A.M_Rows");
            if (xl.N != n) throw new ArgumentException("QP.solve: xl.N must equal Q.M_Rows");
            if (xu.N != n) throw new ArgumentException("QP.solve: xu.N must equal Q.M_Rows");
            if (x.N != n) throw new ArgumentException("QP.solve: x.N must equal Q.M_Rows");

            for (int j = 0; j < n; j++)
                if (xl[j] > xu[j]) throw new ArgumentException("QP.solve: xl must be <= xu componentwise");

            fProxy normInfQ = Norms.LInf(in Q);
            fProxy symTol = Consts.fProxyZeroThreshold * math.max(normInfQ, (fProxy)1);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    if (math.abs(Q[i, j] - Q[j, i]) > symTol)
                        throw new ArgumentException("QP.solve: Q must be symmetric");

            bool feasible = PhaseOneFeasibleStart(in A, in b, in senses, in xl, in xu, ref x);
            if (!feasible)
            {
                for (int j = 0; j < n; j++) x[j] = (fProxy)0;
                objective = 0;
                return new QPInfo { status = QPStatus.Infeasible, iterations = 0, objective = 0, stationarityResidual = 0, feasibilityResidual = 0 };
            }

            return qpActiveSetCore(in Q, in c, in A, in b, in senses, in xl, in xu, ref x, out objective, maxIter);
        }

        /// <summary>
        /// Convenience overload of the bounded <see cref="solve"/> for the common BOX-FREE case --
        /// every variable unbounded both directions (xl = -infinity, xu = +infinity, the library's
        /// 1e30 sentinel convention). A separate overload rather than default parameter values for
        /// <c>xl</c>/<c>xu</c>: <c>fProxyN</c> is a struct wrapping a native allocation, so there is no
        /// compile-time-constant default value to give it (the same reason proxy-typed parameters
        /// elsewhere in this codebase use a forwarding overload instead of a default value).
        /// </summary>
        public static QPInfo solve(in fProxyMxN Q, in fProxyN c, in fProxyMxN A, in fProxyN b,
                                   in NativeArray<ConstraintSense> senses,
                                   ref fProxyN x, out double objective, int maxIter = 0)
        {
            int n = Q.M_Rows;
            fProxy INF = (fProxy)1e30;
            var xl = new fProxyN(n, Allocator.Temp, true);
            var xu = new fProxyN(n, Allocator.Temp, true);
            for (int j = 0; j < n; j++) { xl[j] = -INF; xu[j] = INF; }

            var info = solve(in Q, in c, in A, in b, in senses, in xl, in xu, ref x, out objective, maxIter);

            xu.Dispose(); xl.Dispose();
            return info;
        }

        // Phase 1: find ANY point satisfying A x {<=,=,>=} b AND xl <= x <= xu via a ZERO-COST LP over
        // the SAME feasible region, reusing LP.solve (LPMethod.DualSimplex) instead of writing a
        // dedicated QP feasibility routine.
        //
        // LP.solve's computational form is `A x {<=,=,>=} b, x >= 0` ONLY (no general bounds) -- QP's
        // xl/xu can be negative or +-infinite (the library's 1e30 sentinel convention, matching
        // qpActiveSetCore's BuildRowBounds), so every variable is re-expressed in a shifted/split
        // non-negative variable y before handing off:
        //   * xl[j] finite                 -- anchor low:  x_j = xl[j] + y_j,  y_j >= 0 (native).
        //   * xl[j] infinite, xu[j] finite -- anchor high: x_j = xu[j] - y_j,  y_j >= 0 (native).
        //   * both infinite (free)         -- split:       x_j = y_j+ - y_j-, both >= 0 (native).
        //   * both finite (boxed)          -- anchor low (arbitrary pick) PLUS one extra row
        //                                      y_j <= xu[j]-xl[j] (LP.solve has no native upper bound
        //                                      -- "bounded-above adds rows").
        // Each original row's constant term picked up by a shift substitution moves to that row's RHS;
        // a split contributes a +/- coefficient pair on the SAME row, no RHS change. Sense is never
        // flipped (a per-variable linear substitution, not a row-scale). The LP's objective is the zero
        // vector -- finding ANY vertex of the feasible region is the whole point, not optimizing
        // anything.
        //
        // mprime == 0 (no general rows AND no boxed variable) means every variable is unconstrained in
        // at least one direction with nothing to intersect against -- trivially feasible at its own
        // single remaining bound (or 0 if free) with no LP needed at all; handled as an early return
        // rather than asking LP.solve to factor a zero-row system.
        //
        // Returns false (infeasible, x left untouched) or true (x overwritten with a feasible point).
        // Anything other than LPStatus.Optimal from the phase-1 LP (Infeasible, or the far rarer
        // MaxIterations/Unbounded -- the latter cannot actually happen for a zero-cost objective, kept
        // only as a defensive catch-all) is treated as infeasible.
        internal static bool PhaseOneFeasibleStart(in fProxyMxN A, in fProxyN b, in NativeArray<ConstraintSense> senses,
                                                    in fProxyN xl, in fProxyN xu, ref fProxyN x)
        {
            int m = A.M_Rows, n = A.N_Cols;

            var kind = new NativeArray<byte>(math.max(n, 1), Allocator.Temp);   // 0=anchor-low, 1=anchor-high, 2=free-split
            var col = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
            int nExtraCols = 0, nExtraRows = 0;
            for (int j = 0; j < n; j++)
            {
                bool loFinite = (double)xl[j] > -1e29;
                bool hiFinite = (double)xu[j] < 1e29;
                if (loFinite) { kind[j] = 0; if (hiFinite) nExtraRows++; }
                else if (hiFinite) { kind[j] = 1; }
                else { kind[j] = 2; nExtraCols++; }
            }
            int nprime = n + nExtraCols;
            int mprime = m + nExtraRows;
            { int nextCol = 0; for (int j = 0; j < n; j++) { col[j] = nextCol; nextCol += (kind[j] == 2 ? 2 : 1); } }

            if (mprime == 0)
            {
                for (int j = 0; j < n; j++)
                    x[j] = kind[j] == 0 ? xl[j] : (kind[j] == 1 ? xu[j] : (fProxy)0);
                col.Dispose(); kind.Dispose();
                return true;
            }

            var Anew = new fProxyMxN(mprime, nprime, Allocator.Temp);      // zero-initialized
            var bnew = new fProxyN(mprime, Allocator.Temp, true);
            var sensesNew = new NativeArray<ConstraintSense>(mprime, Allocator.Temp);
            var cZero = new fProxyN(nprime, Allocator.Temp);               // zero-initialized

            for (int i = 0; i < m; i++)
            {
                fProxy shiftSum = (fProxy)0;
                for (int j = 0; j < n; j++)
                {
                    fProxy a = A[i, j];
                    if (a == (fProxy)0) continue;
                    if (kind[j] == 0) { Anew[i, col[j]] = a; shiftSum += a * xl[j]; }
                    else if (kind[j] == 1) { Anew[i, col[j]] = -a; shiftSum += a * xu[j]; }
                    else { Anew[i, col[j]] = a; Anew[i, col[j] + 1] = -a; }
                }
                bnew[i] = b[i] - shiftSum;
                sensesNew[i] = senses[i];
            }

            int rowIdx = m;
            for (int j = 0; j < n; j++)
            {
                if (kind[j] == 0 && (double)xu[j] < 1e29)
                {
                    Anew[rowIdx, col[j]] = (fProxy)1;
                    bnew[rowIdx] = xu[j] - xl[j];
                    sensesNew[rowIdx] = ConstraintSense.LessEqual;
                    rowIdx++;
                }
            }

            var y = new fProxyN(nprime, Allocator.Temp, true);
            var lpInfo = LP.solve(in Anew, in bnew, in cZero, in sensesNew, ref y, out double _, LPMethod.DualSimplex, 0);

            bool feasible = lpInfo;
            if (feasible)
            {
                for (int j = 0; j < n; j++)
                {
                    if (kind[j] == 0) x[j] = xl[j] + y[col[j]];
                    else if (kind[j] == 1) x[j] = xu[j] - y[col[j]];
                    else x[j] = y[col[j]] - y[col[j] + 1];
                }
            }

            y.Dispose(); cZero.Dispose(); sensesNew.Dispose(); bnew.Dispose(); Anew.Dispose(); col.Dispose(); kind.Dispose();
            return feasible;
        }

        /// <summary>
        /// Solve the EQUALITY-constrained QP  min ½xᵀQx + cᵀx  s.t. A_W x = b_W  EXACTLY, from
        /// scratch: reaches a feasible point via <see cref="LQ.minNormSolve(ref fProxyMxN, ref fProxyN, ref fProxyN)"/>
        /// (A_W is k x n, k &lt;= n, independent rows -- exactly the "wide, full row rank"
        /// underdetermined system that targets), then takes ONE exact null-space Newton step -- see
        /// <see cref="eqpNullSpaceStep"/> and this file's header comment for why one step suffices.
        ///
        /// A FIXED working set (algorithm steps 2-3 only). INTERNAL: the public surface for QP is
        /// <c>QP.solve</c>; this is the reusable EQP kernel entry a future inequality active-set loop
        /// would call once the active set is pinned down for a given iteration. Hand-written tests
        /// reach this via InternalsVisibleTo (BurstLinearAlgebra.Tests, see AssemblyInfo.cs), the same
        /// route as LP.ladFrischNewtonCore / LadFrischNewtonQuantileTests.cs.
        /// </summary>
        /// <param name="Q">Symmetric PSD Hessian, n x n. Only referenced via matrix products (Qx,
        /// ZᵀQZ) -- not verified to be exactly symmetric in storage (both triangles are read).</param>
        /// <param name="c">Linear cost, length n.</param>
        /// <param name="A_W">Working-set constraint matrix, k x n (1 &lt;= k &lt;= n), rows
        /// independent. Not modified.</param>
        /// <param name="b_W">Working-set right-hand side, length k. Not modified.</param>
        /// <param name="x">Output only; prior contents ignored; safe to allocate with uninit: true.
        /// Solution, length n.</param>
        /// <param name="lambda">Output only; prior contents ignored; safe to allocate with uninit:
        /// true. Multipliers for A_W's rows (A_Wᵀlambda = Qx+c at the optimum), length k. NOT written
        /// on <see cref="QPStatus.Unbounded"/>.</param>
        internal static QPInfo eqpSolve(in fProxyMxN Q, in fProxyN c, in fProxyMxN A_W, in fProxyN b_W,
                                        ref fProxyN x, ref fProxyN lambda)
        {
            // LQ.minNormSolve does not modify its A/b arguments (they are copied into its own working
            // buffers internally) -- a local struct copy (same handle, zero-cost) is enough to satisfy
            // its `ref` parameters from our `in` ones.
            var A_W_rw = A_W;
            var b_W_rw = b_W;
            LQ.minNormSolve(ref A_W_rw, ref b_W_rw, ref x);

            return eqpNullSpaceStep(in Q, in c, in A_W, in b_W, ref x, ref lambda);
        }

        /// <summary>
        /// Given an ALREADY FEASIBLE <paramref name="x"/> (A_W x = b_W) and a fixed working set
        /// <paramref name="A_W"/>, take the exact null-space Newton step to the equality-constrained
        /// optimum for THIS working set: assemble A_Wᵀ, factor it (<see cref="FactorWorkingSetTranspose"/>),
        /// form the reduced gradient and Z implicitly from the same factor
        /// (<see cref="ApplyWorkingSetQtForward"/>, <see cref="FormNullSpaceBasis"/>), solve the reduced
        /// Hessian system with regularization on breakdown (<see cref="SolveReducedNewtonStep"/>), take
        /// the step, and recover the multipliers from the SAME QR factor (R-solve of the reduced
        /// gradient's leading k entries). Updates <paramref name="x"/> and <paramref name="lambda"/> in
        /// place; returns the diagnostics (see <see cref="QPInfo"/>).
        ///
        /// Called once by <see cref="eqpSolve"/> and that is the whole algorithm for a fixed working
        /// set (no ratio test / add-drop loop). Always applies the full computed step and recovers
        /// multipliers in the same call.
        /// </summary>
        /// <param name="Q">Symmetric PSD Hessian, n x n.</param>
        /// <param name="c">Linear cost, length n.</param>
        /// <param name="A_W">Working-set constraint matrix, k x n (1 &lt;= k &lt;= n), rows
        /// independent. Not modified.</param>
        /// <param name="b_W">Working-set right-hand side, length k. Not modified. Used only for the
        /// cheap feasibility-residual diagnostic (the step itself never needs it -- it stays within
        /// null(A_W) by construction).</param>
        /// <param name="x">On entry a feasible point (A_W x = b_W); on exit the equality-constrained
        /// optimum (unchanged from entry on <see cref="QPStatus.Unbounded"/>).</param>
        /// <param name="lambda">Output only; prior contents ignored. Multipliers, length k. NOT
        /// written on <see cref="QPStatus.Unbounded"/>.</param>
        internal static QPInfo eqpNullSpaceStep(in fProxyMxN Q, in fProxyN c, in fProxyMxN A_W, in fProxyN b_W,
                                                ref fProxyN x, ref fProxyN lambda)
        {
            int n = Q.M_Rows;
            int k = A_W.M_Rows;
            int nz = n - k;

            if (!Q.IsSquare)
                throw new ArgumentException("QP.eqpNullSpaceStep: Q must be square");
            if (A_W.N_Cols != n)
                throw new ArgumentException("QP.eqpNullSpaceStep: A_W.N_Cols must equal Q.M_Rows");
            if (c.N != n)
                throw new ArgumentException("QP.eqpNullSpaceStep: c.N must equal Q.M_Rows");
            if (b_W.N != k)
                throw new ArgumentException("QP.eqpNullSpaceStep: b_W.N must equal A_W.M_Rows");
            if (x.N != n)
                throw new ArgumentException("QP.eqpNullSpaceStep: x.N must equal Q.M_Rows");
            if (lambda.N != k)
                throw new ArgumentException("QP.eqpNullSpaceStep: lambda.N must equal A_W.M_Rows");
            // k == 0 (an empty working set) is deliberately unsupported -- it would need a separate
            // Z-is-implicitly-identity fast path (no A_Wᵀ to factor at all) that the current test
            // matrix (k in {1, n/4, n/2, n-1}) never exercises; revisit if a future active-set loop
            // can transiently reach an empty working set.
            if (k < 1 || k > n)
                throw new ArgumentException("QP.eqpNullSpaceStep: A_W.M_Rows (k) must be between 1 and Q.M_Rows (n) inclusive");

            fProxy normInfQ = Norms.LInf(in Q);
            fProxy zeroThreshold = Consts.fProxyZeroThreshold * math.max(normInfQ, (fProxy)1);

            var AWT = new fProxyMxN(n, k, Allocator.Temp, true);
            Blas.trans(in A_W, ref AWT);

            var R = new fProxyMxN(k, k, Allocator.Temp, true);
            var u = new fProxyN(n, Allocator.Temp, false);
            var w = new fProxyN(math.max(k, math.max(nz, 1)), Allocator.Temp, false);

            FactorWorkingSetTranspose(ref AWT, ref R, ref u, ref w, zeroThreshold);

            var Qx = new fProxyN(n, Allocator.Temp, false);
            var g = new fProxyN(n, Allocator.Temp, false);
            Blas.dot(in Q, in x, ref Qx);
            for (int i = 0; i < n; i++) g[i] = Qx[i] + c[i];

            // g[0..k) <- Q1ᵀg, g[k..n) <- Zᵀg  (at the feasible start x) -- see file header.
            ApplyWorkingSetQtForward(ref AWT, ref g, ref u, k);

            QPStatus status = QPStatus.Optimal;

            if (nz == 0)
            {
                // Fully determined working set (k == n): A_W is n x n and full rank, so x is ALREADY
                // pinned down exactly by the constraints alone -- there is no null space to step
                // through. lambda solves A_Wᵀlambda = g directly (R-solve of the now-full-length
                // transformed g).
                for (int i = 0; i < k; i++) lambda[i] = g[i];
                Blas.triUpper(ref R, ref lambda);
            }
            else
            {
                var gz = new fProxyN(nz, Allocator.Temp, false);
                for (int j = 0; j < nz; j++) gz[j] = g[k + j];

                var Z = new fProxyMxN(n, nz, Allocator.Temp, true);
                FormNullSpaceBasis(ref AWT, ref Z, ref u, ref w, k);

                var QZ = new fProxyMxN(n, nz, Allocator.Temp, true);
                var Hz = new fProxyMxN(nz, nz, Allocator.Temp, true);
                var y = new fProxyN(nz, Allocator.Temp, false);

                var choInfo = SolveReducedNewtonStep(in Q, ref Z, ref QZ, ref Hz, ref gz, ref y, normInfQ, out bool regularized);

                bool unbounded = !choInfo.Solved;
                if (!unbounded && regularized)
                {
                    // Descent-direction check on the REGULARIZED step: a successful regularized solve
                    // mathematically guarantees pᵀg = yᵀgz <= 0 (H_Z+δI is PD, so yᵀ(H_Z+δI)y >= 0 =>
                    // -yᵀgz >= 0), so in exact arithmetic this branch is a formality that should never
                    // fire for a genuinely PSD Q -- it is a catch-all for "Q singular along an unbounded
                    // ray", detectable only if roundoff or a not-quite-PSD Q defeats the guarantee above.
                    double pdotg = 0;
                    for (int j = 0; j < nz; j++) pdotg += (double)y[j] * (double)gz[j];
                    if (pdotg > (double)zeroThreshold)
                        unbounded = true;
                }

                if (unbounded)
                {
                    status = QPStatus.Unbounded;
                }
                else
                {
                    var p = new fProxyN(n, Allocator.Temp, false);
                    Blas.dot(in Z, in y, ref p);
                    for (int i = 0; i < n; i++) x[i] += p[i];
                    p.Dispose();

                    // Recompute g at the NEW x and replay the SAME stashed reflectors -- the
                    // multipliers must satisfy A_Wᵀlambda = g(x_new), not g(x0) (see file header: the
                    // null-space step changes g by Q*(step), which the multiplier solve at x0 alone
                    // cannot account for).
                    Blas.dot(in Q, in x, ref Qx);
                    for (int i = 0; i < n; i++) g[i] = Qx[i] + c[i];
                    ApplyWorkingSetQtForward(ref AWT, ref g, ref u, k);

                    for (int i = 0; i < k; i++) lambda[i] = g[i];
                    Blas.triUpper(ref R, ref lambda);
                }

                y.Dispose(); Hz.Dispose(); QZ.Dispose(); Z.Dispose(); gz.Dispose();
            }

            // Stationarity residual: ‖Zᵀg‖∞ at the FINAL x -- already sitting in g's tail after the
            // last ApplyWorkingSetQtForward call above (both branches leave it there); 0 on Unbounded
            // (no valid step was taken, the residual would be stale) or when nz == 0 (no null space).
            double stationarity = 0;
            if (status == QPStatus.Optimal)
                for (int j = k; j < n; j++)
                    stationarity = math.max(stationarity, math.abs((double)g[j]));

            // Objective + feasibility at the FINAL x. Qx already equals Q*x_final in every path above
            // (nz==0: x never changed after the first Qx; Unbounded: x never changed, same Qx still
            // valid; the normal step path: Qx was refreshed right after x += p) -- no redundant GEMV.
            double objective = 0;
            for (int i = 0; i < n; i++)
                objective += 0.5 * (double)x[i] * (double)Qx[i] + (double)c[i] * (double)x[i];

            var Ax = new fProxyN(k, Allocator.Temp, false);
            Blas.dot(in A_W, in x, ref Ax);
            double feasibility = 0;
            for (int i = 0; i < k; i++)
                feasibility = math.max(feasibility, math.abs((double)Ax[i] - (double)b_W[i]));
            Ax.Dispose();

            g.Dispose(); Qx.Dispose(); w.Dispose(); u.Dispose(); R.Dispose(); AWT.Dispose();

            return new QPInfo
            {
                status = status,
                iterations = 1,
                objective = objective,
                stationarityResidual = stationarity,
                feasibilityResidual = feasibility,
            };
        }

        // Factor A_Wᵀ (AWT, n x k) into R (k x k, upper triangular, caller-allocated and expected
        // zero-initialized -- only entries with c >= r are written) plus k stashed Householder
        // reflectors left in AWT's own columns (row i, column d holds u_d[i] for i >= d) -- exactly
        // decompInPlace's OWN storage convention, WITHOUT its subsequent Q-reconstruction phase (see
        // file header for why: this keeps Q1 implicit). u is scratch of length AWT.M_Rows (= n); w is
        // scratch of length >= AWT.N_Cols (= k). Reuses QR's per-step primitives (QR.genHouseholder /
        // QR.applyReflectorRight, both `internal`) -- not a re-derivation of the Householder math.
        //
        // Always re-factors A_Wᵀ from scratch (no incremental update) -- dense target sizes make an
        // O(n²k) re-factor per change cheap enough. If an incremental version is ever needed, the QRCP
        // downdating machinery is the natural donor (it already removes a column from a QR factor
        // without refactoring; an add/drop of a working-set constraint is a column add/drop on A_Wᵀ).
        internal static void FactorWorkingSetTranspose(ref fProxyMxN AWT, ref fProxyMxN R, ref fProxyN u, ref fProxyN w, fProxy zeroThreshold)
        {
            int n = AWT.M_Rows;
            int k = AWT.N_Cols;

            for (int d = 0; d < k; d++)
            {
                QR.genHouseholder(ref AWT, ref u, d, zeroThreshold);
                QR.applyReflectorRight(ref AWT, ref u, ref w, d);

                R[d, d] = AWT[d, d];
                for (int i = d; i < n; i++)
                    AWT[i, d] = u[i];
            }

            // R's strict upper triangle (c > r): read AFTER the full loop, since row r's entries in
            // columns c > r are finalized by step d = r's own reflector apply and never touched again
            // by any later step (which only ever modifies rows >= its own d > r) -- see file header.
            for (int r = 0; r < k; r++)
            for (int c = r + 1; c < k; c++)
                R[r, c] = AWT[r, c];
        }

        // v[0..k) <- Q1ᵀv, v[k..n) <- Zᵀv, applying the k reflectors STASHED by
        // FactorWorkingSetTranspose (read back from AWT's columns, NOT regenerated) in FORWARD order
        // (d = 0..k-1). See file header for why this single sweep produces both halves at once and why
        // it never needs to form Q1. u is scratch of length AWT.M_Rows (= n), overwritten (used to
        // stage each reflector as it is read back). Mirrors QR.solveInPlace's own `b`-transform loop
        // (plain scalar accumulation, not the matrix-apply's vectorised UnsafeOP.axpy path -- v is a
        // single vector here, not a trailing submatrix).
        internal static void ApplyWorkingSetQtForward(ref fProxyMxN AWT, ref fProxyN v, ref fProxyN u, int k)
        {
            int n = AWT.M_Rows;

            for (int d = 0; d < k; d++)
            {
                for (int r = d; r < n; r++) u[r] = AWT[r, d];

                fProxy dot = 0;
                for (int r = d; r < n; r++) dot += u[r] * v[r];
                for (int r = d; r < n; r++) v[r] -= u[r] * dot;
            }
        }

        // Z (n x (n-k)): an orthonormal basis for null(A_W) -- the trailing n-k columns of the FULL
        // orthogonal factor Q_full = H_0 H_1 ... H_{k-1} -- formed by REVERSE-sweeping (d = k-1..0)
        // the reflectors FactorWorkingSetTranspose stashed in AWT's columns over the seed [0; I_{n-k}]
        // (n x (n-k)). Z must be caller-allocated n x (n-k) and is fully overwritten (any prior
        // contents are irrelevant -- it is zeroed here). u is scratch of length AWT.M_Rows (= n); w is
        // scratch of length >= Z.N_Cols (= n-k).
        //
        // Deliberately does NOT call QR.applyReflectorRight (unlike FactorWorkingSetTranspose above) --
        // that primitive restricts each reflector d's column range to [d, N_Cols), which is exact ONLY
        // for decompInPlace's own reconstruction, seeded with the LEADING identity: there, column j's
        // seed is e_j, and H_i (support on rows >= i) satisfies H_i·e_j = e_j EXACTLY whenever i > j
        // (row j < i is outside u_i's support), so skipping columns < d when applying H_d is a true
        // per-column no-op, not just an optimization. Z's seed is the TRAILING identity block instead
        // (column j = e_{k+j}), and every reflector index d < k <= k+j, so u_d's support (rows >= d)
        // ALWAYS includes row k+j -- H_d is generally NOT a no-op on any column of Z, for any d. The
        // column restriction would therefore silently skip real work. ApplyReflectorFullWidth below is
        // the same rank-1 update with that restriction removed -- full column width every step.
        internal static void FormNullSpaceBasis(ref fProxyMxN AWT, ref fProxyMxN Z, ref fProxyN u, ref fProxyN w, int k)
        {
            int n = AWT.M_Rows;
            int nz = Z.N_Cols;

            unsafe { UnsafeUtility.MemClear(Z.Data.Ptr, (long)Z.Data.Length * UnsafeUtility.SizeOf<fProxy>()); }
            for (int j = 0; j < nz; j++)
                Z[k + j, j] = (fProxy)1;

            for (int d = k - 1; d >= 0; d--)
            {
                for (int r = d; r < n; r++) u[r] = AWT[r, d];
                ApplyReflectorFullWidth(ref Z, ref u, ref w, d);
            }
        }

        // Apply Householder reflector H_d = I - u·uᵀ (u supported on rows >= d) to M's FULL column
        // width, rows [d, M_Rows): M[d:, :] -= u·(uᵀ·M[d:, :]). Same two-pass vectorised shape as
        // QR.applyReflectorRightCols (UnsafeOP.axpy, the GEMM pointer path), but WITHOUT that
        // primitive's "columns >= d" restriction -- see FormNullSpaceBasis's doc comment for why that
        // restriction does not generalize to a seed other than the leading identity. w is scratch of
        // length >= M.N_Cols.
        internal static unsafe void ApplyReflectorFullWidth(ref fProxyMxN M, ref fProxyN u, ref fProxyN w, int d)
        {
            int rows = M.M_Rows;
            int cols = M.N_Cols;
            if (cols <= 0) return;

            fProxy* mp = M.Data.Ptr;
            fProxy* up = u.Data.Ptr;
            fProxy* wp = w.Data.Ptr;

            UnsafeUtility.MemClear(wp, (long)cols * UnsafeUtility.SizeOf<fProxy>());
            for (int r = d; r < rows; r++)
                UnsafeOP.axpy(wp, mp + (long)r * cols, up[r], cols);
            for (int r = d; r < rows; r++)
                UnsafeOP.axpy(mp + (long)r * cols, wp, -up[r], cols);
        }

        // Solve the reduced-Hessian Newton system H_Z y = -gz for y, H_Z = ZᵀQZ (nz x nz, PSD since Q
        // is PSD -- two matMatDot-shaped calls via Blas.dot). On a
        // Cholesky breakdown (H_Z numerically singular -- possible even though Q is only PSD, not PD:
        // Q's null space can overlap Z's span), retries ONCE with H_Z + delta*normInfQ*I,
        // delta = sqrt(Consts.fProxyEpsilon) -- a PSD matrix plus a strictly positive multiple of I is
        // always PD, so this retry cannot itself break down for a genuinely PSD Q. QZ/Hz are
        // caller-allocated scratch (n x nz / nz x nz respectively, sized by the caller since it also
        // owns Z); y is the caller-allocated destination (length nz). Returns the CHO status from the
        // (possibly regularized) solve and, via <paramref name="regularized"/>, whether the retry
        // path was taken -- the caller uses that to run the descent-direction / Unbounded check, which
        // needs gz and y regardless of which path produced them.
        internal static DirectSolveInfo SolveReducedNewtonStep(in fProxyMxN Q, ref fProxyMxN Z, ref fProxyMxN QZ, ref fProxyMxN Hz,
                                                                ref fProxyN gz, ref fProxyN y, fProxy normInfQ, out bool regularized)
        {
            int nz = y.N;

            Blas.dot(in Q, in Z, ref QZ);
            Blas.dot(in Z, in QZ, ref Hz, transposeA: true);

            for (int j = 0; j < nz; j++) y[j] = -gz[j];
            var info = CHO.solveInPlace(ref Hz, ref y);

            regularized = false;
            if (!info.Solved)
            {
                regularized = true;
                fProxy delta = math.sqrt(Consts.fProxyEpsilon) * math.max(normInfQ, (fProxy)1);

                // CHO.decompInPlace leaves a failed factor PARTIALLY overwritten (documented
                // "destroyed on failure"); rebuild H_Z cleanly from the still-intact Z/QZ before
                // adding the regularizer, rather than trying to patch the partial factor in place.
                Blas.dot(in Z, in QZ, ref Hz, transposeA: true);
                for (int j = 0; j < nz; j++) Hz[j, j] += delta;

                for (int j = 0; j < nz; j++) y[j] = -gz[j];
                info = CHO.solveInPlace(ref Hz, ref y);
            }

            return info;
        }

        // ============================================================================================
        // The INEQUALITY-constrained active-set LOOP -- algorithm steps 1-5 minus phase 1. This
        // stage's entry point, qpActiveSetCore, takes a CALLER-SUPPLIED feasible x0 and validates it
        // rather than manufacturing one. Built directly on the constituent kernel functions above
        // (FactorWorkingSetTranspose / ApplyWorkingSetQtForward / FormNullSpaceBasis /
        // SolveReducedNewtonStep, all still `internal static`, UNCHANGED) instead of through
        // eqpSolve/eqpNullSpaceStep, since the ratio test must see the step p BEFORE it is applied, to
        // know how far it is safe to go (and possibly not apply it at all).
        //
        // Problem solved:
        //
        //     minimize    1/2 xᵀQx + cᵀx
        //     subject to  A x {<=,=,>=} b     (per-row senses, LP.solve's ConstraintSense)
        //                 xl <= x <= xu
        //
        // Unified row/bound representation: every constraint -- general row AND variable bound alike --
        // is one range L_t <= (row t).x <= U_t over T = m + n rows (t < m: general row t; t >= m:
        // variable bound j = t - m, normal e_j, L/U = xl[j]/xu[j]). Each row's WorkingSetStatus
        // (QP.Info.cs) records which side it is pinned to; ActiveLower/Equality rows enter A_W AS-IS
        // (+row), ActiveUpper rows enter NEGATED (-row), so the whole working set shares one
        // "row.x >= bound" sign convention (Nocedal & Wright's convention, matching the multiplier
        // recovery A_Wᵀlambda = g) -- one uniform "lambda >= 0" test works for every row without an
        // ActiveLower/ActiveUpper case split.
        //
        // Working-set rank guard: A_W's rows must stay independent. TryAddToWorkingSet tests a
        // candidate row by tentatively appending it as the last column of a from-scratch QR and
        // checking |R[k,k]| against a scale-relative threshold. A row found dependent is left Inactive
        // -- since it is then a linear combination of rows already in W, its activity gradient
        // (row).p is exactly 0 for any p in null(A_W), so excluding it costs nothing.
        //
        // Unbounded detection: declared exactly when all four hold:
        //   1. regularized      -- SolveReducedNewtonStep's Cholesky retry fired (H_Z numerically
        //                          singular).
        //   2. zero curvature   -- the Rayleigh quotient pᵀQp / pᵀp <= zeroThreshold.
        //   3. no blocker       -- RatioTest, run with an UNCAPPED self-limit, finds no inactive
        //                          constraint anywhere along p.
        //   4. descent          -- gᵀp < 0 (scaled by ||p|| for the same scale-invariance as #2).
        // When #1-3 hold but #4 does not (a genuinely FLAT direction), the step is simply not taken and
        // this working set is treated as converged instead -- not Unbounded, since the objective does
        // not in fact decrease without bound there. Verified against Nocedal & Wright, Numerical
        // Optimization (2nd ed.), section 16.5 ("Active-Set Methods for Indefinite QP").
        // ============================================================================================

        /// <summary>
        /// Solve the inequality-constrained convex QP  min 1/2 xᵀQx + cᵀx  s.t. A x {&lt;=,=,&gt;=} b
        /// (per-row <paramref name="senses"/>), xl &lt;= x &lt;= xu, from a CALLER-SUPPLIED feasible
        /// starting point (<paramref name="x"/> on entry) -- the primal null-space active-set method,
        /// HiGHS / Nocedal &amp; Wright ch. 16 lineage (see this file's header comments). Q must be
        /// symmetric PSD (v1 contract, same as <see cref="eqpSolve"/>).
        ///
        /// The inequality add/drop loop (algorithm steps 1-5 minus phase 1). INTERNAL: phase 1 (an
        /// LP-powered feasible start, so callers need not supply one themselves) is the public
        /// <c>QP.solve</c> facade, which calls this once a feasible x0 is in hand.
        /// </summary>
        /// <param name="Q">Symmetric PSD Hessian, n x n.</param>
        /// <param name="c">Linear cost, length n.</param>
        /// <param name="A">Constraint coefficients, m x n.</param>
        /// <param name="b">Right-hand sides, length m.</param>
        /// <param name="senses">Per-row constraint sense, length m.</param>
        /// <param name="xl">Variable lower bounds, length n. Use a large-magnitude negative sentinel
        /// (&lt;= -1e29) for a variable unbounded below.</param>
        /// <param name="xu">Variable upper bounds, length n. Use a large-magnitude positive sentinel
        /// (&gt;= 1e29) for a variable unbounded above.</param>
        /// <param name="x">On entry a FEASIBLE point (A x {&lt;=,=,&gt;=} b AND xl &lt;= x &lt;= xu,
        /// checked up front to <see cref="QPStatus.Infeasible"/> tolerance); on exit the optimum (on
        /// <see cref="QPStatus.MaxIterations"/>, the last feasible iterate; unchanged from entry on
        /// <see cref="QPStatus.Infeasible"/> or <see cref="QPStatus.Unbounded"/>).</param>
        /// <param name="objective">Output 1/2 xᵀQx + cᵀx at the returned x, computed fresh regardless
        /// of status (matches <see cref="LP.solve"/>'s convention).</param>
        /// <param name="maxIter">Pivot budget; &lt;= 0 picks a size-based default.</param>
        internal static QPInfo qpActiveSetCore(in fProxyMxN Q, in fProxyN c, in fProxyMxN A, in fProxyN b,
                                               in NativeArray<ConstraintSense> senses,
                                               in fProxyN xl, in fProxyN xu,
                                               ref fProxyN x, out double objective, int maxIter)
        {
            int n = Q.M_Rows, m = A.M_Rows, T = m + n;

            if (!Q.IsSquare) throw new ArgumentException("QP.qpActiveSetCore: Q must be square");
            if (A.N_Cols != n) throw new ArgumentException("QP.qpActiveSetCore: A.N_Cols must equal Q.M_Rows");
            if (b.N != m) throw new ArgumentException("QP.qpActiveSetCore: b.N must equal A.M_Rows");
            if (c.N != n) throw new ArgumentException("QP.qpActiveSetCore: c.N must equal Q.M_Rows");
            if (senses.Length != m) throw new ArgumentException("QP.qpActiveSetCore: senses.Length must equal A.M_Rows");
            if (xl.N != n) throw new ArgumentException("QP.qpActiveSetCore: xl.N must equal Q.M_Rows");
            if (xu.N != n) throw new ArgumentException("QP.qpActiveSetCore: xu.N must equal Q.M_Rows");
            if (x.N != n) throw new ArgumentException("QP.qpActiveSetCore: x.N must equal Q.M_Rows");

            fProxy INF = (fProxy)1e30;
            fProxy normInfQ = Norms.LInf(in Q);
            fProxy normInfA = Norms.LInf(in A);
            // Q-space tolerance (curvature / regularization-delta / descent-direction checks) -- same
            // scale stage 1's eqpNullSpaceStep already derives for the SAME purposes.
            fProxy zeroThreshold = Consts.fProxyZeroThreshold * math.max(normInfQ, (fProxy)1);
            // Constraint-space tolerance (the working-set QR rank guard) -- A_W's own natural scale
            // (bound rows contribute norm 1, general rows contribute normInfA), deliberately SEPARATE
            // from zeroThreshold above since it factors a different matrix.
            fProxy zeroThresholdAW = Consts.fProxyZeroThreshold * math.max(normInfA, (fProxy)1);
            fProxy feasTol = (fProxy)(math.max(math.sqrt((double)Consts.fProxyEpsilon), 1e-7)) * math.max((fProxy)1, normInfA);
            fProxy pivTol = math.max(Consts.fProxyZeroThreshold, (fProxy)1e-9);
            fProxy dualTol = feasTol;

            var L = new fProxyN(T, Allocator.Temp, true);
            var U = new fProxyN(T, Allocator.Temp, true);
            BuildRowBounds(in b, in senses, in xl, in xu, m, n, ref L, ref U);

            // ---- validate x0's feasibility up front ----
            var Ax0 = new fProxyN(math.max(m, 1), Allocator.Temp, true);
            if (m > 0) Blas.dot(in A, in x, ref Ax0);
            bool feasible = true;
            double worstViol = 0;
            for (int t = 0; t < T; t++)
            {
                double act = t < m ? (double)Ax0[t] : (double)x[t - m];
                double lo = (double)L[t] - (double)feasTol, hi = (double)U[t] + (double)feasTol;
                if (act < lo) { feasible = false; worstViol = math.max(worstViol, lo - act); }
                else if (act > hi) { feasible = false; worstViol = math.max(worstViol, act - hi); }
            }

            if (!feasible)
            {
                Ax0.Dispose(); L.Dispose(); U.Dispose();
                var Qxi = new fProxyN(n, Allocator.Temp, true);
                Blas.dot(in Q, in x, ref Qxi);
                double objInfeas = 0;
                for (int i = 0; i < n; i++) objInfeas += 0.5 * (double)x[i] * (double)Qxi[i] + (double)c[i] * (double)x[i];
                Qxi.Dispose();
                objective = objInfeas;
                return new QPInfo { status = QPStatus.Infeasible, iterations = 0, objective = objInfeas, stationarityResidual = 0, feasibilityResidual = worstViol };
            }

            var wstatus = new NativeArray<byte>(T, Allocator.Temp);   // zero-init -> every row Inactive
            SeedWorkingSet(in A, in L, in U, m, n, T, in x, in Ax0, wstatus, feasTol, zeroThresholdAW);
            Ax0.Dispose();

            int budget = maxIter > 0 ? maxIter : 50 * T + 200;
            int degenCap = 3 * math.max(n, 1);
            int degenCount = 0;
            // Anti-cycling hardening: HiGHS-style deterministic bound perturbation. Once a run of
            // alpha=0 (degenerate) steps reaches degenCap, usePerturbation switches the ratio test
            // (both call sites below) from the TRUE L/U to a lazily-built, SLIGHTLY WIDENED pair
            // (BuildPerturbedBounds -- perturbedL <= L <= U <= perturbedU always, so nothing
            // feasible under the true bounds ever becomes infeasible), which breaks the EXACT
            // ties that cause a zero-length step in the first place. The multiplier sign check just
            // below (the "optimality decision") never reads L/U at all -- it depends only on
            // g = Qx+c and the working-set geometry (see this file's
            // "unified row/bound representation" header note) -- so it is, structurally, already
            // "deciding on ORIGINAL data" without needing a Bland-style special case. What perturbation
            // CAN leave behind is a perturbation-sized drift in x itself (it took a step to a
            // slightly-off bound); that is REMOVED at the end by one more exact null-space Newton step
            // against the TRUE bounds, once the loop reaches Optimal -- see the cleanup pass right
            // after this loop.
            bool usePerturbation = false;
            bool perturbationEverUsed = false;
            fProxyN perturbedL = default, perturbedU = default;
            bool havePerturbedBuffers = false;
            QPStatus status = QPStatus.Optimal;
            int iterations = 0;

            while (true)
            {
                if (iterations >= budget) { status = QPStatus.MaxIterations; break; }

                var curL = usePerturbation ? perturbedL : L;
                var curU = usePerturbation ? perturbedU : U;

                // ---- factor the CURRENT working set from scratch (v1 judgment: no incremental
                // update, see this file's Stage-1 header comment on FactorWorkingSetTranspose) ----
                var rowOfCol = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
                int k = AssembleWorkingSetTranspose(in A, in L, in U, m, n, T, wstatus, -1, WorkingSetStatus.Inactive, rowOfCol, out var AWT, out var bW);
                int nz = n - k;

                var R = new fProxyMxN(k, k, Allocator.Temp);
                var u = new fProxyN(n, Allocator.Temp, true);
                var w = new fProxyN(math.max(k, math.max(nz, 1)), Allocator.Temp, true);
                FactorWorkingSetTranspose(ref AWT, ref R, ref u, ref w, zeroThresholdAW);

                var Qx = new fProxyN(n, Allocator.Temp, true);
                var g = new fProxyN(n, Allocator.Temp, true);
                Blas.dot(in Q, in x, ref Qx);
                for (int i = 0; i < n; i++) g[i] = Qx[i] + c[i];
                ApplyWorkingSetQtForward(ref AWT, ref g, ref u, k);

                // ---- compute the null-space Newton step p ----
                bool haveNullSpace = nz > 0;
                fProxyN gz = default, y = default, p = default;
                fProxyMxN Z = default, QZ = default, Hz = default;
                bool regularized = false, haveP = false;
                double pInf = 0, pNormSq = 0, gp = 0;
                QPStatus? exitStatus = null;

                if (haveNullSpace)
                {
                    gz = new fProxyN(nz, Allocator.Temp, true);
                    for (int j = 0; j < nz; j++) gz[j] = g[k + j];
                    Z = new fProxyMxN(n, nz, Allocator.Temp, true);
                    FormNullSpaceBasis(ref AWT, ref Z, ref u, ref w, k);
                    QZ = new fProxyMxN(n, nz, Allocator.Temp, true);
                    Hz = new fProxyMxN(nz, nz, Allocator.Temp, true);
                    y = new fProxyN(nz, Allocator.Temp, true);

                    var choInfo = SolveReducedNewtonStep(in Q, ref Z, ref QZ, ref Hz, ref gz, ref y, normInfQ, out regularized);
                    if (!choInfo.Solved)
                    {
                        // Stage 1's own hard-failure bail (the regularized retry ITSELF failed --
                        // "should never happen for genuinely PSD Q", see SolveReducedNewtonStep's doc
                        // comment): unconditional Unbounded, same as stage 1.
                        exitStatus = QPStatus.Unbounded;
                    }
                    else
                    {
                        p = new fProxyN(n, Allocator.Temp, true);
                        haveP = true;
                        Blas.dot(in Z, in y, ref p);
                        for (int i = 0; i < n; i++) { double pi = (double)p[i]; pInf = math.max(pInf, math.abs(pi)); pNormSq += pi * pi; }
                        for (int j = 0; j < nz; j++) gp += (double)gz[j] * (double)y[j];
                    }
                }

                bool small = !haveNullSpace || pInf <= (double)feasTol;

                fProxy thetaSelf = (fProxy)1;
                fProxy pScale = (fProxy)1;
                fProxy alphaTake = (fProxy)0;
                int addRow = -1; bool addUpper = false;
                bool doTakeStep = false;
                bool doMultiplierCheck = small;
                fProxyN Ax = default, Ap = default;
                NativeArray<bool> excluded = default;
                bool haveRatioBufs = false;

                if (exitStatus == null && !small)
                {
                    // ---- curvature test + ratio test ----
                    var Qp = new fProxyN(n, Allocator.Temp, true);
                    Blas.dot(in Q, in p, ref Qp);
                    double curvature = 0;
                    for (int i = 0; i < n; i++) curvature += (double)p[i] * (double)Qp[i];
                    Qp.Dispose();

                    bool zeroCurv = regularized && pNormSq > 0 && (curvature / pNormSq) <= (double)zeroThreshold;

                    // Rescale p to unit inf-norm for the ratio test (RatioTest's `pScale` divides d by
                    // this internally): along the regularized/zero-curvature path p = -Z(H_Z+deltaI)^-1
                    // gz can be enormous (~1/delta), which would otherwise make alpha come out tiny
                    // (~delta-scaled) and make feasTol's Harris tie-window (calibrated for an O(1) alpha)
                    // FAR too coarse relative to the true spacing between distinct blocking points --
                    // several genuinely different blockers would look "tied" and the wrong (overstepping)
                    // one could win, corrupting feasibility. Scale-invariant fix: run the ratio test on
                    // p/pInf (alpha then O(1)-scaled regardless of p's raw magnitude) and convert back
                    // (alpha_original = alpha_hat / pInf).
                    pScale = (fProxy)math.max(pInf, 1e-30);
                    thetaSelf = zeroCurv ? INF : pScale;

                    Ax = new fProxyN(math.max(m, 1), Allocator.Temp, true);
                    Ap = new fProxyN(math.max(m, 1), Allocator.Temp, true);
                    if (m > 0) { Blas.dot(in A, in x, ref Ax); Blas.dot(in A, in p, ref Ap); }
                    excluded = new NativeArray<bool>(T, Allocator.Temp);   // zero-init -> none excluded
                    haveRatioBufs = true;

                    RatioTest(in curL, in curU, m, n, T, wstatus, excluded, in Ax, in Ap, in x, in p, pScale, thetaSelf, feasTol, pivTol, out fProxy alphaHat, out int winnerRow, out bool winnerUpper);
                    fProxy alpha = alphaHat / pScale;

                    if (zeroCurv && winnerRow < 0)
                    {
                        // See this file's header comment for the full 4-conjunct derivation
                        // (conditions 1-3 all hold here; this is condition 4, descent).
                        if (gp <= -(double)zeroThreshold * math.sqrt(pNormSq))
                        {
                            exitStatus = QPStatus.Unbounded;
                        }
                        else
                        {
                            // Flat, non-improving direction with no blocker: further movement along p
                            // cannot help. Converged for this working set -- multiplier check, no step.
                            doMultiplierCheck = true;
                        }
                    }
                    else
                    {
                        doTakeStep = true;
                        alphaTake = winnerRow >= 0 ? alpha : thetaSelf / pScale;
                        addRow = winnerRow; addUpper = winnerUpper;
                    }
                }

                // ---- act (mutate x / wstatus) BEFORE disposing this iteration's scratch ----
                if (exitStatus == null && doTakeStep)
                {
                    bool degenerate = alphaTake <= pivTol;
                    degenCount = degenerate ? degenCount + 1 : 0;
                    usePerturbation = degenCount >= degenCap;
                    if (usePerturbation)
                    {
                        perturbationEverUsed = true;
                        if (!havePerturbedBuffers)
                        {
                            perturbedL = new fProxyN(T, Allocator.Temp, true);
                            perturbedU = new fProxyN(T, Allocator.Temp, true);
                            BuildPerturbedBounds(in L, in U, T, feasTol, ref perturbedL, ref perturbedU);
                            havePerturbedBuffers = true;
                        }
                    }

                    for (int i = 0; i < n; i++) x[i] += alphaTake * p[i];

                    if (addRow >= 0)
                    {
                        int tryRow = addRow; bool tryUpper = addUpper;
                        int guardAttempts = 0;
                        // Bounded rank-guard retry: a naive ratio-test winner that would make A_W
                        // rank-deficient (degenerate/redundant-constraint instances) is excluded and
                        // the next-best candidate tried instead. Capped
                        // (not unbounded) -- if every candidate fails, W is simply left unchanged this
                        // iteration; the degenerate-step counter / iteration budget are the backstop.
                        while (guardAttempts < 8)
                        {
                            var cand = tryUpper ? WorkingSetStatus.ActiveUpper : WorkingSetStatus.ActiveLower;
                            if (TryAddToWorkingSet(in A, in L, in U, m, n, T, wstatus, tryRow, cand, zeroThresholdAW))
                                break;
                            excluded[tryRow] = true;
                            guardAttempts++;
                            RatioTest(in curL, in curU, m, n, T, wstatus, excluded, in Ax, in Ap, in x, in p, pScale, thetaSelf, feasTol, pivTol, out _, out int nextRow, out bool nextUpper);
                            if (nextRow < 0) break;
                            tryRow = nextRow; tryUpper = nextUpper;
                        }
                    }
                    iterations++;
                }
                else if (exitStatus == null && doMultiplierCheck)
                {
                    // ---- multiplier recovery + sign check ----
                    var lamBuf = new fProxyN(math.max(k, 1), Allocator.Temp, true);
                    for (int i = 0; i < k; i++) lamBuf[i] = g[i];
                    if (k > 0) Blas.triUpper(ref R, ref lamBuf);

                    // Dantzig pricing (most-negative multiplier) unconditionally -- no Bland-style
                    // tie-break needed here: this decision never reads L/U at all (see the
                    // usePerturbation comment above the loop), so bound-perturbation hardening cannot
                    // corrupt it, and cycling risk lives entirely in the ratio test's degenerate
                    // zero-length steps, which usePerturbation now addresses directly.
                    int worstCol = -1; fProxy worstLam = -dualTol;
                    for (int kk = 0; kk < k; kk++)
                    {
                        int t = rowOfCol[kk];
                        var st = (WorkingSetStatus)wstatus[t];
                        if (st == WorkingSetStatus.Equality) continue;   // no sign constraint -- never a drop candidate
                        fProxy lam = lamBuf[kk];
                        if (lam < -dualTol && lam < worstLam) { worstLam = lam; worstCol = kk; }
                    }

                    if (worstCol < 0) exitStatus = QPStatus.Optimal;
                    else { wstatus[rowOfCol[worstCol]] = (byte)WorkingSetStatus.Inactive; iterations++; }

                    lamBuf.Dispose();
                }

                // ---- dispose this iteration's scratch (every path reaches here) ----
                if (haveNullSpace)
                {
                    if (haveP) p.Dispose();
                    y.Dispose(); Hz.Dispose(); QZ.Dispose(); Z.Dispose(); gz.Dispose();
                }
                g.Dispose(); Qx.Dispose(); w.Dispose(); u.Dispose(); R.Dispose(); AWT.Dispose(); bW.Dispose(); rowOfCol.Dispose();
                if (haveRatioBufs) { Ax.Dispose(); Ap.Dispose(); excluded.Dispose(); }

                if (exitStatus.HasValue) { status = exitStatus.Value; break; }
            }

            // ---- one exact null-space Newton step against the TRUE (unperturbed) bounds removes any
            // perturbation drift left in x -- no-op (skipped) whenever perturbation was never engaged.
            // ----
            if (perturbationEverUsed && status == QPStatus.Optimal)
            {
                var rowOfColC = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
                int kC = AssembleWorkingSetTranspose(in A, in L, in U, m, n, T, wstatus, -1, WorkingSetStatus.Inactive, rowOfColC, out var AWTc, out var bWc);
                if (kC > 0)
                {
                    var A_Wc = new fProxyMxN(kC, n, Allocator.Temp, true);
                    Blas.trans(in AWTc, ref A_Wc);
                    var lambdaC = new fProxyN(kC, Allocator.Temp, true);
                    // eqpSolve's status is only ever Optimal or Unbounded (see QPInfo/eqpSolve's own
                    // doc comments); Unbounded here would mean Q lost PSD-ness along a direction the
                    // multiplier check should already have caught (defensive-only, see this file's own
                    // "should never fire for genuinely PSD Q" framing elsewhere) -- x is left as the
                    // perturbed-but-already-near-optimal iterate in that case (eqpSolve does not
                    // modify x on Unbounded).
                    var cleanupInfo = eqpSolve(in Q, in c, in A_Wc, in bWc, ref x, ref lambdaC);
                    if (cleanupInfo.status == QPStatus.Optimal) iterations += 1;
                    lambdaC.Dispose();
                    A_Wc.Dispose();
                }
                rowOfColC.Dispose(); AWTc.Dispose(); bWc.Dispose();
            }
            if (havePerturbedBuffers) { perturbedL.Dispose(); perturbedU.Dispose(); }

            // ---- final diagnostics (fresh, matching LP.solve's "recompute from original data") ----
            double stationarity = 0;
            if (status == QPStatus.Optimal)
            {
                var rowOfColF = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
                int kf = AssembleWorkingSetTranspose(in A, in L, in U, m, n, T, wstatus, -1, WorkingSetStatus.Inactive, rowOfColF, out var AWTf, out var bWf);
                int nzf = n - kf;
                var Rf = new fProxyMxN(kf, kf, Allocator.Temp);
                var uf = new fProxyN(n, Allocator.Temp, true);
                var wf = new fProxyN(math.max(kf, math.max(nzf, 1)), Allocator.Temp, true);
                FactorWorkingSetTranspose(ref AWTf, ref Rf, ref uf, ref wf, zeroThresholdAW);

                var Qxf = new fProxyN(n, Allocator.Temp, true);
                var gf = new fProxyN(n, Allocator.Temp, true);
                Blas.dot(in Q, in x, ref Qxf);
                for (int i = 0; i < n; i++) gf[i] = Qxf[i] + c[i];
                ApplyWorkingSetQtForward(ref AWTf, ref gf, ref uf, kf);
                for (int j = kf; j < n; j++) stationarity = math.max(stationarity, math.abs((double)gf[j]));

                gf.Dispose(); Qxf.Dispose(); wf.Dispose(); uf.Dispose(); Rf.Dispose(); AWTf.Dispose(); bWf.Dispose(); rowOfColF.Dispose();
            }

            double feasibilityResidual = 0;
            {
                var Axf = new fProxyN(math.max(m, 1), Allocator.Temp, true);
                if (m > 0) Blas.dot(in A, in x, ref Axf);
                for (int t = 0; t < T; t++)
                {
                    double act = t < m ? (double)Axf[t] : (double)x[t - m];
                    if (act < (double)L[t]) feasibilityResidual = math.max(feasibilityResidual, (double)L[t] - act);
                    else if (act > (double)U[t]) feasibilityResidual = math.max(feasibilityResidual, act - (double)U[t]);
                }
                Axf.Dispose();
            }

            wstatus.Dispose(); L.Dispose(); U.Dispose();

            var Qxo = new fProxyN(n, Allocator.Temp, true);
            Blas.dot(in Q, in x, ref Qxo);
            double obj = 0;
            for (int i = 0; i < n; i++) obj += 0.5 * (double)x[i] * (double)Qxo[i] + (double)c[i] * (double)x[i];
            Qxo.Dispose();
            objective = obj;

            return new QPInfo
            {
                status = status,
                iterations = iterations,
                objective = obj,
                stationarityResidual = stationarity,
                feasibilityResidual = feasibilityResidual,
            };
        }

        // Builds L (T) / U (T) row-bound arrays from the problem data: t < m is general row t
        // (ConstraintSense-derived range), t >= m is variable bound j = t - m (xl[j]/xu[j] directly).
        // See qpActiveSetCore's file-header comment for the unified-representation rationale.
        internal static void BuildRowBounds(in fProxyN b, in NativeArray<ConstraintSense> senses,
                                            in fProxyN xl, in fProxyN xu, int m, int n,
                                            ref fProxyN L, ref fProxyN U)
        {
            fProxy INF = (fProxy)1e30;
            for (int i = 0; i < m; i++)
            {
                switch (senses[i])
                {
                    case ConstraintSense.LessEqual: L[i] = -INF; U[i] = b[i]; break;
                    case ConstraintSense.Equal: L[i] = b[i]; U[i] = b[i]; break;
                    default: L[i] = b[i]; U[i] = INF; break;   // GreaterEqual
                }
            }
            for (int j = 0; j < n; j++) { L[m + j] = xl[j]; U[m + j] = xu[j]; }
        }

        // Widens L/U deterministically to break ratio-test ties (perturbedL <= L <= U <= perturbedU
        // always -- never TIGHTEN, so anything feasible under the TRUE bounds stays feasible under the
        // perturbed ones); see qpActiveSetCore's usePerturbation comment for when/why this is called.
        // Sentinel (+-1e29) sides are left untouched. perturbedL/perturbedU must be caller-allocated,
        // length T; every entry is (re)written.
        internal static void BuildPerturbedBounds(in fProxyN L, in fProxyN U, int T, fProxy feasTol,
                                                   ref fProxyN perturbedL, ref fProxyN perturbedU)
        {
            fProxy mag = (fProxy)0.1 * feasTol;
            for (int t = 0; t < T; t++)
            {
                uint h = (uint)t * 2654435761u + 0x9E3779B9u;
                h ^= h >> 15; h *= 0x85EBCA6Bu;
                h ^= h >> 13; h *= 0xC2B2AE35u;
                h ^= h >> 16;
                fProxy widen = mag * (fProxy)(0.5 + 0.5 * (h * (1.0 / 4294967295.0)));

                perturbedL[t] = (double)L[t] > -1e29 ? L[t] - widen : L[t];
                perturbedU[t] = (double)U[t] < 1e29 ? U[t] + widen : U[t];
            }
        }

        // Seeds the working set from x0's tight constraints. Pass 1: every equality row (L_t == U_t --
        // general Equal-sense rows AND fixed bounds xl[j]==xu[j]) is permanently in W, added via the
        // SAME rank guard as everything else (a redundant/duplicated equality is simply left Inactive;
        // see TryAddToWorkingSet's doc comment for why that is safe, not a lost constraint). Pass 2:
        // every remaining row tight at
        // x0 within feasTol (general row or bound) is added as ActiveLower/ActiveUpper, independence-
        // guarded the same way. wstatus must be caller-allocated, length T; every entry is (re)written.
        internal static void SeedWorkingSet(in fProxyMxN A, in fProxyN L, in fProxyN U, int m, int n, int T,
                                            in fProxyN x0, in fProxyN Ax0, NativeArray<byte> wstatus,
                                            fProxy feasTol, fProxy zeroThresholdAW)
        {
            for (int t = 0; t < T; t++) wstatus[t] = (byte)WorkingSetStatus.Inactive;

            for (int t = 0; t < T; t++)
                if (L[t] == U[t])
                    TryAddToWorkingSet(in A, in L, in U, m, n, T, wstatus, t, WorkingSetStatus.Equality, zeroThresholdAW);

            for (int t = 0; t < T; t++)
            {
                if (wstatus[t] != (byte)WorkingSetStatus.Inactive) continue;
                double act = t < m ? (double)Ax0[t] : (double)x0[t - m];
                bool atLower = (double)L[t] > -1e29 && math.abs(act - (double)L[t]) <= (double)feasTol;
                bool atUpper = (double)U[t] < 1e29 && math.abs(act - (double)U[t]) <= (double)feasTol;
                if (atLower)
                    TryAddToWorkingSet(in A, in L, in U, m, n, T, wstatus, t, WorkingSetStatus.ActiveLower, zeroThresholdAW);
                else if (atUpper)
                    TryAddToWorkingSet(in A, in L, in U, m, n, T, wstatus, t, WorkingSetStatus.ActiveUpper, zeroThresholdAW);
            }
        }

        // Assembles A_Wᵀ (AWT, n x k) and b_W (bW, length k) from wstatus, in ascending row-index order
        // (t = 0..T-1, skipping Inactive rows), sign-oriented per WorkingSetStatus (ActiveLower/Equality:
        // +row; ActiveUpper: -row -- see qpActiveSetCore's file-header comment). If extraT >= 0, ONE
        // more column is appended AFTER all of wstatus's active rows for row extraT with status
        // extraStatus -- WITHOUT reading or writing wstatus[extraT] itself (a pure query: the caller
        // decides whether to commit it, see TryAddToWorkingSet). rowOfCol (caller-allocated, length >= n)
        // is filled with the row index t that produced each column (rowOfCol[kk] = t); only the first
        // (returned) k entries are meaningful. AWT/bW are allocated fresh (Allocator.Temp, uninit) at
        // EXACTLY the returned k -- no reshape/view capability exists for fProxyMxN (see this file's
        // header comment on why per-iteration re-assembly, not incremental update, is v1's design).
        internal static int AssembleWorkingSetTranspose(in fProxyMxN A, in fProxyN L, in fProxyN U, int m, int n, int T,
                                                         NativeArray<byte> wstatus, int extraT, WorkingSetStatus extraStatus,
                                                         NativeArray<int> rowOfCol, out fProxyMxN AWT, out fProxyN bW)
        {
            int k = 0;
            for (int t = 0; t < T; t++) if (wstatus[t] != (byte)WorkingSetStatus.Inactive) k++;
            if (extraT >= 0) k++;

            AWT = new fProxyMxN(n, k, Allocator.Temp, true);
            bW = new fProxyN(k, Allocator.Temp, true);

            int kk = 0;
            for (int t = 0; t < T; t++)
            {
                var st = (WorkingSetStatus)wstatus[t];
                if (st == WorkingSetStatus.Inactive) continue;
                WriteWorkingSetColumn(in A, in L, in U, m, n, t, st, ref AWT, ref bW, kk);
                rowOfCol[kk] = t;
                kk++;
            }
            if (extraT >= 0)
            {
                WriteWorkingSetColumn(in A, in L, in U, m, n, extraT, extraStatus, ref AWT, ref bW, kk);
                rowOfCol[kk] = extraT;
                kk++;
            }
            return k;
        }

        // Writes column `col` of AWT/bW for row t under the given status (sign-oriented, see
        // AssembleWorkingSetTranspose's doc comment). t < m: A's row t; t >= m: the unit row e_{t-m}.
        internal static void WriteWorkingSetColumn(in fProxyMxN A, in fProxyN L, in fProxyN U, int m, int n,
                                                    int t, WorkingSetStatus status,
                                                    ref fProxyMxN AWT, ref fProxyN bW, int col)
        {
            fProxy sign = status == WorkingSetStatus.ActiveUpper ? (fProxy)(-1) : (fProxy)1;
            if (t < m)
                for (int i = 0; i < n; i++) AWT[i, col] = sign * A[t, i];
            else
            {
                int j = t - m;
                for (int i = 0; i < n; i++) AWT[i, col] = (fProxy)0;
                AWT[j, col] = sign;
            }
            bW[col] = sign > (fProxy)0 ? L[t] : -U[t];
        }

        // Tests whether adding candidate row t (oriented per candStatus) to the CURRENT wstatus keeps
        // A_W's rows independent, via a throwaway trial factor (AssembleWorkingSetTranspose's extraT
        // path places the candidate as the LAST Householder column regardless of its own row index, so
        // R[k-1,k-1] is exactly its component orthogonal to everything already accepted -- see this
        // file's header comment). On success, COMMITS (sets wstatus[t] = candStatus) and returns true;
        // on failure, leaves wstatus untouched and returns false. Used by SeedWorkingSet and by the main
        // loop's blocking-constraint add.
        internal static bool TryAddToWorkingSet(in fProxyMxN A, in fProxyN L, in fProxyN U, int m, int n, int T,
                                                NativeArray<byte> wstatus, int t, WorkingSetStatus candStatus,
                                                fProxy zeroThresholdAW)
        {
            var rowOfCol = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
            int k = AssembleWorkingSetTranspose(in A, in L, in U, m, n, T, wstatus, t, candStatus, rowOfCol, out var AWT, out var bW);

            var R = new fProxyMxN(k, k, Allocator.Temp);
            var u = new fProxyN(n, Allocator.Temp, true);
            var w = new fProxyN(math.max(k, 1), Allocator.Temp, true);
            FactorWorkingSetTranspose(ref AWT, ref R, ref u, ref w, zeroThresholdAW);

            bool ok = math.abs(R[k - 1, k - 1]) > zeroThresholdAW;

            w.Dispose(); u.Dispose(); R.Dispose(); AWT.Dispose(); bW.Dispose(); rowOfCol.Dispose();

            if (ok) wstatus[t] = (byte)candStatus;
            return ok;
        }

        // Harris-shaped two-pass ratio test over INACTIVE rows (the SHAPE of LP.RevisedSimplex's
        // HarrisRatioTest, not its code: x is ALREADY feasible for every row here
        // (not just W), so there is no "healing an infeasible basic variable" case to handle, unlike
        // that LP phase-1 ratio test -- every inactive row's current activity already sits within
        // [L_t, U_t] to feasTol). d_t = (row t).p / pScale is the RESCALED rate the row's activity moves
        // per unit of the returned alpha (Ap[t]/pScale for t < m, p[t-m]/pScale for a bound row) --
        // pScale (the caller's ||p||_inf, or 1 if the caller already knows p is unit-scale) makes alpha
        // come out O(1)-scaled regardless of p's own raw magnitude, which matters a lot along the
        // regularized/zero-curvature path where p can be enormous (~1/delta): WITHOUT this rescaling,
        // alpha would come out correspondingly tiny and feasTol's Harris tie-window below (calibrated
        // for an O(1) alpha) would be far too coarse relative to the true spacing between distinct
        // blocking points, corrupting the winner choice -- see qpActiveSetCore's call site comment.
        // The caller un-rescales the returned alpha (alpha_original = alpha / pScale) and thetaSelf must
        // already be pre-scaled by the SAME pScale (INF is its own rescale, unaffected). Rows with
        // |d_t| <= pivTol, or whose relevant bound is the +-1e29 unbounded sentinel, can never block and
        // are skipped. winnerRow is -1 (no block within thetaSelf) or the winning row, tie-broken by
        // largest |d_t| among candidates within feasTol of the relaxed (pass-1) threshold, matching
        // HarrisRatioTest's own stability rationale. `excluded` (caller-allocated, length T) lets the
        // rank-guard retry re-run this test skipping already-rejected rows without mutating wstatus.
        internal static void RatioTest(in fProxyN L, in fProxyN U, int m, int n, int T,
                                       NativeArray<byte> wstatus, NativeArray<bool> excluded,
                                       in fProxyN Ax, in fProxyN Ap, in fProxyN x, in fProxyN p,
                                       fProxy pScale, fProxy thetaSelf, fProxy feasTol, fProxy pivTol,
                                       out fProxy alpha, out int winnerRow, out bool winnerUpper)
        {
            fProxy thetaRelaxed = thetaSelf;

            for (int t = 0; t < T; t++)
            {
                if (wstatus[t] != (byte)WorkingSetStatus.Inactive || excluded[t]) continue;
                fProxy d = (t < m ? Ap[t] : p[t - m]) / pScale;
                if (math.abs(d) <= pivTol) continue;
                fProxy act = t < m ? Ax[t] : x[t - m];

                if (d > (fProxy)0)
                {
                    if (U[t] >= (fProxy)1e29) continue;
                    fProxy tcand = (U[t] + feasTol - act) / d;
                    if (tcand < (fProxy)0) tcand = (fProxy)0;
                    if (tcand < thetaRelaxed) thetaRelaxed = tcand;
                }
                else
                {
                    if (L[t] <= (fProxy)(-1e29)) continue;
                    fProxy tcand = (L[t] - feasTol - act) / d;
                    if (tcand < (fProxy)0) tcand = (fProxy)0;
                    if (tcand < thetaRelaxed) thetaRelaxed = tcand;
                }
            }

            if (thetaRelaxed >= thetaSelf)
            {
                alpha = thetaSelf; winnerRow = -1; winnerUpper = false; return;
            }

            int winner = -1; fProxy winnerMag = (fProxy)(-1); fProxy winnerExact = (fProxy)0; bool winnerUp = false;
            for (int t = 0; t < T; t++)
            {
                if (wstatus[t] != (byte)WorkingSetStatus.Inactive || excluded[t]) continue;
                fProxy d = (t < m ? Ap[t] : p[t - m]) / pScale;
                fProxy absd = math.abs(d);
                if (absd <= pivTol) continue;
                fProxy act = t < m ? Ax[t] : x[t - m];
                bool isUp = d > (fProxy)0;
                if (isUp && U[t] >= (fProxy)1e29) continue;
                if (!isUp && L[t] <= (fProxy)(-1e29)) continue;
                fProxy bound = isUp ? U[t] : L[t];

                fProxy texact = (bound - act) / d;
                if (texact < (fProxy)0) texact = (fProxy)0;
                if (texact <= thetaRelaxed + feasTol && absd > winnerMag)
                {
                    winnerMag = absd; winner = t; winnerExact = texact; winnerUp = isUp;
                }
            }

            if (winner < 0)
            {
                alpha = thetaRelaxed; winnerRow = -1; winnerUpper = false; return;
            }
            alpha = winnerExact; winnerRow = winner; winnerUpper = winnerUp;
        }
    }
}
