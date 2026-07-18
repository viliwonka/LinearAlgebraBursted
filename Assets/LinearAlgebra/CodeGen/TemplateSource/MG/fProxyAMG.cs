using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Unsmoothed nodal-aggregation algebraic multigrid hierarchy over a square SPD BSR. Built once on
    /// the main thread (<see cref="Arena.fProxyAMG(in fProxyBSR, in AMGOptions, out AMGSetupInfo)"/>);
    /// each level carries its operator A_l, tentative prolongator P_l (level l+1 -> l), a Chebyshev
    /// smoother, and scratch vectors, with the coarsest level solved by a dense Cholesky. Applies a
    /// symmetric V-cycle as a standalone solver (<see cref="MG.solve(in fProxyAMG, in fProxyN, ref fProxyN, int, fProxy)"/>)
    /// or as an SPD preconditioner (<see cref="fProxyAMGPreconditioner"/>).
    ///
    /// The level DATA (operators, prolongators, smoothers, vectors) is arena-owned; only the small
    /// per-level handle CONTAINERS are held directly and released by <see cref="Dispose"/>. No mutable
    /// scalar fields — all cycle state lives in the vector buffers, so an IJob struct copy is safe.
    /// </summary>
    public struct fProxyAMG : IDisposable
    {
        // Level handle containers (allocator-owned; freed by Dispose). The referenced BSR/vector/
        // smoother storage is arena-owned and freed with the arena.
        UnsafeList<fProxyBSR> _A;        // [Levels]   operators, A[0] = fine
        UnsafeList<fProxyBSR> _P;        // [Levels-1] P[l]: level l+1 -> l
        UnsafeList<fProxyChebyshev> _S;  // [Levels-1] smoother for level l
        UnsafeList<fProxyN> _X, _B, _R, _Z;  // [Levels] per-level solution/rhs/residual/correction

        // K-cycle only: per-level 2-step Flexible-CG scratch (restricted residual, two search
        // directions, their operator images, and the accumulated correction). Empty for a V-cycle.
        UnsafeList<fProxyN> _krc, _kc1, _kc2, _kv1, _kv2, _ke;

        fProxyMxN _coarseChol;           // dense Cholesky factor of A[Levels-1]
        fProxyN _coarseRhs;              // coarsest solve scratch

        int _levels;
        int _pre, _post;
        int _cycle;                     // 0 = V, 1 = K
        bool _usable;                   // false when the coarsest Cholesky failed (do not solve)
        Allocator _alloc;

        public int Levels => _levels;
        public int Rows => _A[0].M_Rows;
        /// <summary>Pre-smoothing sweeps per level per cycle.</summary>
        public int Pre => _pre;
        /// <summary>Post-smoothing sweeps per level per cycle.</summary>
        public int Post => _post;
        /// <summary>True iff the build succeeded (coarsest Cholesky SPD). Solving an unusable
        /// hierarchy would emit NaN — the entry points throw instead.</summary>
        public bool Usable => _usable;
        /// <summary>True for a K-cycle hierarchy: the cycle is a VARIABLE operator, so it must be
        /// driven by <see cref="LinearAlgebra.Krylov"/>.fcg (not pcg / not a fixed-M pcg precond).</summary>
        public bool IsKCycle => _cycle == 1;
        /// <summary>True iff the cycle is a fixed SPD operator valid for pcg: a symmetric (Pre == Post)
        /// V-cycle. A K-cycle is never pcg-valid.</summary>
        public bool IsCycleSymmetric => _cycle == 0 && _pre == _post;

        internal fProxyAMG(UnsafeList<fProxyBSR> A, UnsafeList<fProxyBSR> P, UnsafeList<fProxyChebyshev> S,
            UnsafeList<fProxyN> X, UnsafeList<fProxyN> B, UnsafeList<fProxyN> R, UnsafeList<fProxyN> Z,
            UnsafeList<fProxyN> krc, UnsafeList<fProxyN> kc1, UnsafeList<fProxyN> kc2,
            UnsafeList<fProxyN> kv1, UnsafeList<fProxyN> kv2, UnsafeList<fProxyN> ke,
            fProxyMxN coarseChol, fProxyN coarseRhs, int levels, int pre, int post, int cycle, bool usable, Allocator alloc)
        {
            _A = A; _P = P; _S = S; _X = X; _B = B; _R = R; _Z = Z;
            _krc = krc; _kc1 = kc1; _kc2 = kc2; _kv1 = kv1; _kv2 = kv2; _ke = ke;
            _coarseChol = coarseChol; _coarseRhs = coarseRhs;
            _levels = levels; _pre = pre; _post = post; _cycle = cycle; _usable = usable; _alloc = alloc;
        }

        /// <summary>Frees the per-level handle containers (not the arena-owned level data).</summary>
        public void Dispose()
        {
            if (_A.IsCreated) _A.Dispose();
            if (_P.IsCreated) _P.Dispose();
            if (_S.IsCreated) _S.Dispose();
            if (_X.IsCreated) _X.Dispose();
            if (_B.IsCreated) _B.Dispose();
            if (_R.IsCreated) _R.Dispose();
            if (_Z.IsCreated) _Z.Dispose();
            if (_krc.IsCreated) _krc.Dispose();
            if (_kc1.IsCreated) _kc1.Dispose();
            if (_kc2.IsCreated) _kc2.Dispose();
            if (_kv1.IsCreated) _kv1.Dispose();
            if (_kv2.IsCreated) _kv2.Dispose();
            if (_ke.IsCreated) _ke.Dispose();
        }

        // One smoothing sweep on level l: X += M^-1 (B - A X), M the level's Chebyshev smoother.
        readonly void Smooth(int l, int times)
        {
            fProxyBSR A = _A[l];
            fProxyN x = _X[l], b = _B[l], r = _R[l], z = _Z[l];
            fProxyChebyshev s = _S[l];
            for (int t = 0; t < times; t++)
            {
                BSR.spMV(in A, in x, ref r);           // r = A x
                r.scaleAddInPlace((fProxy)(-1), b);     // r = b - A x
                s.Apply(in r, ref z);                    // z = M^-1 r
                x.addScaledInPlace((fProxy)1, z);        // x += z
            }
        }

        // One symmetric V-cycle over the level buffers: X[0] (current iterate) and B[0] (rhs) must be
        // set by the caller; coarse levels are zeroed internally. Updates X[0] in place.
        readonly void VCycle()
        {
            int L = _levels;

            // Down: pre-smooth, form residual, restrict to the next coarser level, zero its iterate.
            for (int l = 0; l < L - 1; l++)
            {
                if (_pre > 0) Smooth(l, _pre);

                fProxyBSR A = _A[l];
                fProxyN x = _X[l], b = _B[l], r = _R[l];
                BSR.spMV(in A, in x, ref r);
                r.scaleAddInPlace((fProxy)(-1), b);     // r = b - A x

                fProxyBSR P = _P[l];
                fProxyN bc = _B[l + 1];
                BSR.spMVT(in P, in r, ref bc);           // b_{l+1} = P^T r

                fProxyN xc = _X[l + 1];
                for (int i = 0; i < xc.N; i++) xc[i] = (fProxy)0;
            }

            // Coarsest: dense Cholesky solve. chol/crhs are local handle copies (a readonly method
            // cannot pass a field of `this` by ref); they alias the same arena buffers.
            {
                int lc = L - 1;
                fProxyN bc = _B[lc], xc = _X[lc];
                fProxyN crhs = _coarseRhs;
                fProxyMxN chol = _coarseChol;
                crhs.Data.CopyFrom(bc.Data);
                CHO.decompSolve(ref chol, ref crhs);
                xc.Data.CopyFrom(crhs.Data);
            }

            // Up: prolongate the coarse correction and post-smooth.
            for (int l = L - 2; l >= 0; l--)
            {
                fProxyBSR P = _P[l];
                fProxyN x = _X[l], xc = _X[l + 1], z = _Z[l];
                BSR.spMV(in P, in xc, ref z);            // z = P x_{l+1}
                x.addScaledInPlace((fProxy)1, z);        // x += z

                if (_post > 0) Smooth(l, _post);
            }
        }

        // K-cycle (Notay 2008 / AMGCL): like the V-cycle, but the coarse correction at each level is
        // computed by TWO steps of Flexible CG on the coarse operator, preconditioned by the next
        // level's K-cycle — the per-level Krylov acceleration that recovers the grid-independence
        // unsmoothed aggregation loses under a plain V-cycle. Recursive (calls itself twice per
        // level); the branching tree is 2^level but per-level work shrinks faster than 2x, so total
        // work stays O(N) with a larger constant than a V-cycle. Solves A_l _X[l] = _B[l] in place
        // (caller sets _X[l] — warm at the top, zeroed for an inner apply). Coarse correction breaks
        // down gracefully to a single unaccelerated apply on non-positive curvature.
        readonly void KCycle(int l)
        {
            int L = _levels;
            if (l == L - 1)
            {
                fProxyN bcz = _B[l], xcz = _X[l];
                fProxyN crhs = _coarseRhs;
                fProxyMxN chol = _coarseChol;
                crhs.Data.CopyFrom(bcz.Data);
                CHO.decompSolve(ref chol, ref crhs);
                xcz.Data.CopyFrom(crhs.Data);
                return;
            }

            if (_pre > 0) Smooth(l, _pre);

            fProxyBSR A = _A[l];
            fProxyBSR Pl = _P[l];                         // local copies: a readonly method cannot
            fProxyN x = _X[l], b = _B[l], r = _R[l], zl = _Z[l];   // pass a field's indexer by ref/in
            BSR.spMV(in A, in x, ref r);
            r.scaleAddInPlace((fProxy)(-1), b);           // r = b - A x

            int c = l + 1;
            fProxyBSR Ac = _A[c];
            fProxyN bc = _B[c], xc = _X[c];
            fProxyN rc = _krc[c], c1 = _kc1[c], c2 = _kc2[c], v1 = _kv1[c], v2 = _kv2[c], e = _ke[c];

            BSR.spMVT(in Pl, in r, ref bc);               // b_c = P^T r  (the restricted residual)
            rc.Data.CopyFrom(bc.Data);                    // keep a copy of rc (b_c gets rewritten)

            // c1 = K(rc): one K-cycle apply from a zero coarse guess.
            for (int i = 0; i < xc.N; i++) xc[i] = (fProxy)0;
            KCycle(c);
            c1.Data.CopyFrom(xc.Data);
            BSR.spMV(in Ac, in c1, ref v1);               // v1 = A_c c1

            fProxy d1 = Blas.dot(c1, v1);
            if (d1 > (fProxy)0)
            {
                fProxy a1 = Blas.dot(c1, rc) / d1;
                for (int i = 0; i < e.N; i++) e[i] = (fProxy)0;
                e.addScaledInPlace(a1, c1);               // e = a1 c1
                rc.addScaledInPlace(-a1, v1);             // rc <- rc - a1 v1

                // c2 = K(rc'): second flexible direction.
                bc.Data.CopyFrom(rc.Data);
                for (int i = 0; i < xc.N; i++) xc[i] = (fProxy)0;
                KCycle(c);
                c2.Data.CopyFrom(xc.Data);
                BSR.spMV(in Ac, in c2, ref v2);           // v2 = A_c c2

                fProxy beta = Blas.dot(c2, v1) / d1;      // A-orthogonalize c2 against c1
                c2.addScaledInPlace(-beta, c1);
                v2.addScaledInPlace(-beta, v1);

                fProxy d2 = Blas.dot(c2, v2);
                if (d2 > (fProxy)0)
                {
                    fProxy a2 = Blas.dot(c2, rc) / d2;
                    e.addScaledInPlace(a2, c2);           // e += a2 c2
                }
            }
            else
            {
                e.Data.CopyFrom(c1.Data);                 // breakdown: single unaccelerated apply
            }

            BSR.spMV(in Pl, in e, ref zl);                // prolong: x += P e
            x.addScaledInPlace((fProxy)1, zl);
            if (_post > 0) Smooth(l, _post);
        }

        // Runs one cycle of the configured shape over the level buffers (X[0]/B[0] set by the caller).
        readonly void Cycle()
        {
            if (_cycle == 1) KCycle(0);
            else VCycle();
        }

        // z = one cycle solving A z = r from a zero initial guess (the preconditioner apply).
        internal readonly void ApplyCycleFromZero(in fProxyN r, ref fProxyN z)
        {
            if (!_usable) throw new InvalidOperationException("fProxyAMG: build failed (coarsest not SPD); do not Apply — check AMGSetupInfo.Solved");
            fProxyN b0 = _B[0], x0 = _X[0];
            b0.Data.CopyFrom(r.Data);
            for (int i = 0; i < x0.N; i++) x0[i] = (fProxy)0;
            Cycle();
            z.Data.CopyFrom(x0.Data);
        }

        // Standalone V-cycle iteration to a relative true-residual tolerance; x is warm-startable.
        internal readonly SolveInfo Solve(in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            if (!_usable) throw new InvalidOperationException("MG.solve: AMG build failed (coarsest not SPD); check AMGSetupInfo.Solved");
            if (b.N != Rows) throw new ArgumentException("MG.solve: b.N must equal amg.Rows");
            if (x.N != Rows) throw new ArgumentException("MG.solve: x.N must equal amg.Rows");
            if (maxIter < 1) throw new ArgumentException("MG.solve: maxIter must be >= 1");

            fProxy bb = Blas.dot(b, b);
            if (bb == (fProxy)0)
            {
                x.Data.CopyFrom(b.Data);
                return new SolveInfo { rnorm = 0, iterations = 0, status = IterativeSolveStatus.Converged };
            }

            fProxy threshold = tol * tol * bb;

            fProxyN x0 = _X[0], b0 = _B[0], r0 = _R[0];
            fProxyBSR A0 = _A[0];
            x0.Data.CopyFrom(x.Data);
            b0.Data.CopyFrom(b.Data);

            fProxy rr = 0;
            for (int it = 0; it < maxIter; it++)
            {
                Cycle();                                  // one MG iteration on X[0]

                BSR.spMV(in A0, in x0, ref r0);
                r0.scaleAddInPlace((fProxy)(-1), b0);     // r = b - A x
                rr = Blas.dot(r0, r0);
                if (rr <= threshold)
                {
                    x.Data.CopyFrom(x0.Data);
                    return new SolveInfo { rnorm = math.sqrt(rr), iterations = it + 1, status = IterativeSolveStatus.Converged };
                }
            }

            x.Data.CopyFrom(x0.Data);
            return new SolveInfo { rnorm = math.sqrt(rr), iterations = maxIter, status = IterativeSolveStatus.MaxIterations };
        }
    }
}

namespace LinearAlgebra
{
    public unsafe partial struct Arena
    {
        /// <summary>
        /// Builds an <see cref="fProxyAMG"/> hierarchy from a square SPD BSR with the scalar constant
        /// near-nullspace (m=1). Coarsens by unsmoothed nodal aggregation until a level has &lt;=
        /// opts.coarseMax scalar unknowns, aggregation stops reducing, or opts.maxLevels is reached;
        /// the coarsest operator is factored by dense Cholesky. info.status is Success, or
        /// NotPositiveDefinite when the coarsest Cholesky fails. A Symmetric-storage A is mirrored to
        /// full transiently. Throws if A is not square. The returned hierarchy must be Disposed.
        /// </summary>
        internal fProxyAMG fProxyAMGBuild(in fProxyBSR A, in fProxyMxN B0, in AMGOptions opts, out AMGSetupInfo info)
        {
            if (A.BlockRows != A.BlockCols || A.BR != A.BC)
                throw new ArgumentException("Arena.fProxyAMG: A must be square (BlockRows==BlockCols, BR==BC)");
            if (B0.M_Rows != A.M_Rows || B0.N_Cols < 1)
                throw new ArgumentException("Arena.fProxyAMG: near-nullspace B must be A.M_Rows x m with m >= 1");
            if (opts.pre < 0 || opts.post < 0)
                throw new ArgumentException("Arena.fProxyAMG: pre/post must be >= 0");
            if (opts.coarseMax < 1 || opts.maxLevels < 1)
                throw new ArgumentException("Arena.fProxyAMG: coarseMax/maxLevels must be >= 1");

            var self = this;
            var alloc = self.Allocator;

            // Declared up front so the finally can free them if a level build (aggregation /
            // prolongator / Galerkin / Chebyshev) throws before the hierarchy is constructed —
            // the containers are allocator-owned, not arena-tracked, so they would otherwise leak.
            var levA = default(UnsafeList<fProxyBSR>);
            var levP = default(UnsafeList<fProxyBSR>);
            var levS = default(UnsafeList<fProxyChebyshev>);
            var levX = default(UnsafeList<fProxyN>);
            var levB = default(UnsafeList<fProxyN>);
            var levR = default(UnsafeList<fProxyN>);
            var levZ = default(UnsafeList<fProxyN>);
            var kRc = default(UnsafeList<fProxyN>);
            var kC1 = default(UnsafeList<fProxyN>);
            var kC2 = default(UnsafeList<fProxyN>);
            var kV1 = default(UnsafeList<fProxyN>);
            var kV2 = default(UnsafeList<fProxyN>);
            var kE = default(UnsafeList<fProxyN>);
            bool ok = false;
            try
            {
                levA = new UnsafeList<fProxyBSR>(4, alloc);
                levP = new UnsafeList<fProxyBSR>(4, alloc);
                levS = new UnsafeList<fProxyChebyshev>(4, alloc);

                fProxyBSR A0 = A.Symmetric ? self.fProxyBSRMirrorToFull(in A) : A;
                levA.Add(A0);

                fProxyMxN Bcur = B0;                     // level-0 near-nullspace (m = B0.N_Cols)

                while (levA[levA.Length - 1].M_Rows > opts.coarseMax && levA.Length < opts.maxLevels)
                {
                    fProxyBSR cur = levA[levA.Length - 1];
                    var aggId = self.Indices(cur.BlockRows);
                    AMG.aggregate(in cur, (fProxy)opts.theta, ref aggId, out int numAgg);
                    if (numAgg >= cur.BlockRows) break;      // aggregation did not coarsen -> stop

                    var T = AMG.tentativeProlongator(in cur, in aggId, numAgg, in Bcur, ref self, out var Bc);
                    var Ac = AMG.galerkinRAP(in cur, in T, in aggId, numAgg, ref self);
                    var sm = new fProxyChebyshev(in cur, ref self);   // smoother for the CURRENT level

                    levP.Add(T);
                    levS.Add(sm);
                    levA.Add(Ac);
                    Bcur = Bc;
                }

                int L = levA.Length;
                fProxyBSR coarse = levA[L - 1];
                fProxyMxN chol = coarse.ToDense(ref self);
                var cinfo = CHO.decompInPlace(ref chol);

                levX = new UnsafeList<fProxyN>(L, alloc);
                levB = new UnsafeList<fProxyN>(L, alloc);
                levR = new UnsafeList<fProxyN>(L, alloc);
                levZ = new UnsafeList<fProxyN>(L, alloc);
                for (int l = 0; l < L; l++)
                {
                    int nl = levA[l].M_Rows;
                    levX.Add(self.fProxyVec(nl));
                    levB.Add(self.fProxyVec(nl));
                    levR.Add(self.fProxyVec(nl));
                    levZ.Add(self.fProxyVec(nl));
                }
                var coarseRhs = self.fProxyVec(coarse.M_Rows);

                int cycle = opts.cycle == MGCycle.K ? 1 : 0;
                if (cycle == 1)
                {
                    // Per-level 2-step-FCG scratch (level 0's slot is unused but kept for indexing).
                    kRc = new UnsafeList<fProxyN>(L, alloc);
                    kC1 = new UnsafeList<fProxyN>(L, alloc);
                    kC2 = new UnsafeList<fProxyN>(L, alloc);
                    kV1 = new UnsafeList<fProxyN>(L, alloc);
                    kV2 = new UnsafeList<fProxyN>(L, alloc);
                    kE = new UnsafeList<fProxyN>(L, alloc);
                    for (int l = 0; l < L; l++)
                    {
                        int nl = levA[l].M_Rows;
                        kRc.Add(self.fProxyVec(nl));
                        kC1.Add(self.fProxyVec(nl));
                        kC2.Add(self.fProxyVec(nl));
                        kV1.Add(self.fProxyVec(nl));
                        kV2.Add(self.fProxyVec(nl));
                        kE.Add(self.fProxyVec(nl));
                    }
                }

                info = new AMGSetupInfo
                {
                    levels = L,
                    coarseRows = coarse.M_Rows,
                    status = cinfo.Solved ? DirectSolveStatus.Success : DirectSolveStatus.NotPositiveDefinite,
                };

                var result = new fProxyAMG(levA, levP, levS, levX, levB, levR, levZ,
                    kRc, kC1, kC2, kV1, kV2, kE, chol, coarseRhs,
                    L, opts.pre, opts.post, cycle, cinfo.Solved, alloc);
                ok = true;
                return result;
            }
            finally
            {
                if (!ok)
                {
                    if (levA.IsCreated) levA.Dispose();
                    if (levP.IsCreated) levP.Dispose();
                    if (levS.IsCreated) levS.Dispose();
                    if (levX.IsCreated) levX.Dispose();
                    if (levB.IsCreated) levB.Dispose();
                    if (levR.IsCreated) levR.Dispose();
                    if (levZ.IsCreated) levZ.Dispose();
                    if (kRc.IsCreated) kRc.Dispose();
                    if (kC1.IsCreated) kC1.Dispose();
                    if (kC2.IsCreated) kC2.Dispose();
                    if (kV1.IsCreated) kV1.Dispose();
                    if (kV2.IsCreated) kV2.Dispose();
                    if (kE.IsCreated) kE.Dispose();
                }
            }
        }

        /// <summary>
        /// Builds an AMG hierarchy with the SCALAR default near-nullspace (B = the constant vector,
        /// m = 1) — correct for scalar PDEs (Poisson, diffusion). For a vector problem (elasticity /
        /// structures) pass the rigid-body modes via the <c>Bnear</c> overload instead. See that
        /// overload for the coarsening/failure/dispose contract.
        /// </summary>
        public fProxyAMG fProxyAMG(in fProxyBSR A, in AMGOptions opts, out AMGSetupInfo info)
        {
            int n = A.M_Rows;
            var ones = fProxyMat(n, 1);
            for (int i = 0; i < n; i++) ones[i, 0] = (fProxy)1;
            return fProxyAMGBuild(in A, in ones, in opts, out info);
        }

        /// <summary>Scalar-default AMG with <see cref="AMGOptions.Default"/>.</summary>
        public fProxyAMG fProxyAMG(in fProxyBSR A, out AMGSetupInfo info)
            => fProxyAMG(in A, AMGOptions.Default, out info);

        /// <summary>
        /// Builds an AMG hierarchy with a user-supplied near-nullspace <paramref name="Bnear"/>
        /// (A.M_Rows x m, row-major) — the low-energy modes the coarse grid must represent. Scalar
        /// problems use m = 1 (the constant, i.e. the overload without Bnear); a vector problem passes
        /// its rigid-body modes (e.g. 3D elasticity: the 3 rigid translations, m = 3, or 6 with
        /// rotations). The coarse block size becomes m. Coarsens by unsmoothed nodal aggregation until
        /// a level has &lt;= opts.coarseMax scalar unknowns / aggregation stops / opts.maxLevels; the
        /// coarsest is factored by dense Cholesky. info.status is Success or NotPositiveDefinite. A
        /// Symmetric-storage A is mirrored to full transiently. Throws if A is not square or Bnear is
        /// not A.M_Rows x (m &gt;= 1). The returned hierarchy must be Disposed.
        /// </summary>
        public fProxyAMG fProxyAMG(in fProxyBSR A, in fProxyMxN Bnear, in AMGOptions opts, out AMGSetupInfo info)
            => fProxyAMGBuild(in A, in Bnear, in opts, out info);

        /// <summary>Near-nullspace AMG with <see cref="AMGOptions.Default"/>.</summary>
        public fProxyAMG fProxyAMG(in fProxyBSR A, in fProxyMxN Bnear, out AMGSetupInfo info)
            => fProxyAMG(in A, in Bnear, AMGOptions.Default, out info);
    }
}
