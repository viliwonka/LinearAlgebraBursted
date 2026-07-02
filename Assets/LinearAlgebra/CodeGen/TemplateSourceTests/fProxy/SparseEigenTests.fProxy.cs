using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Sparse (BSM) power-iteration test suite for the matrix-free Eigen.powerIteration<TOp> refactor:
// the new powerIteration(in fProxyBSM, ...) overloads forward through fProxyBSMOperator into the
// same generic core the dense powerIteration(in fProxyMxN, ...) path uses. Every sparse result is
// cross-checked against the pre-existing dense path (same recipe as fProxySparseSolverTests: build
// the SAME operator in both forms and compare), plus one literature known-spectrum case.
//
// Correctness cases run inside a [BurstCompile] IJob (matches fProxySparseSolverTests /
// fProxyEigenTests) and use fProxyEigenTests' Fail-NativeArray diagnostic convention: a failed
// Assert inside a Burst job aborts silently without surfacing to the runner, so every check first
// records [0]=flag, [1]=got, [2]=expected/limit, [3]=diff into Fail. Guard/exception cases run on
// the managed test thread with Assert.Throws, since NUnit's Assert.Throws cannot execute in Burst.
public class fProxySparseEigenTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SparseEigenTestJob : IJob
    {
        public enum TestType
        {
            DenseVsSparseCrossCheck,
            LaplacianKnownSpectrum,
            InverseLaplacianCrossCheck,
            InverseVsEigenvaluesSymmetric,
            LanczosFullSpectrum,
            LanczosPartialExtremal,
            LanczosDenseVsBSM,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/code
        public NativeArray<fProxy> Fail;

        // Two INDEPENDENTLY-converged iterative eigenpairs (dense manual-matvec vs BSM spMV) are
        // compared, so a machine-epsilon threshold is inappropriate: mirror fProxySparseSolverTests'
        // choose-marker tolerance for iterative-vs-iterative cross-checks (looser on float).
        static fProxy LooseTol() => /*+choose[1e-2f|1e-5]*/1e-2f/*-choose*/;

        // Full-spectrum Lanczos (steps == n, full reorthogonalization TWICE) makes the tridiagonal
        // T orthogonally similar to A, so its Ritz values are as accurate as running
        // eigenvaluesSymmetric on A directly -- i.e. bounded by the QL eigensolver's own floor plus
        // the reorthogonalization roundoff. This is tighter than the iterative LooseTol but a touch
        // looser than EvSymLaplacian's 1000*ZeroThreshold, to absorb that extra roundoff on the
        // larger n=16 tridiagonal used here. Applied as a (1+|lambda|)-scaled absolute tolerance.
        static fProxy FullSpectrumTol() => /*+choose[1e-2f|1e-9]*/1e-2f/*-choose*/;

        // Partial-spectrum Lanczos (steps < n): only the EXTREMAL Ritz values (largest at index 0,
        // smallest at index produced-1) are asserted. Their Kaniel-Paige-Saad convergence in ~n/2
        // steps on this well-separated Laplacian is fast but NOT machine-exact, so this is
        // deliberately looser than FullSpectrumTol. Interior Ritz values are not asserted at all.
        // Applied as a (1+|lambda|)-scaled absolute tolerance (the smallest Laplacian eigenvalue is
        // ~0.034, so the scale is ~1 there and this stays a meaningful bound, not a free pass).
        // Partial Lanczos (steps < n) resolves the EXTREMAL Ritz values only approximately -- the
        // error shrinks as steps->n. At steps=n/2 on the n=16 Laplacian the largest/smallest Ritz
        // values land ~7e-4 (absolute) from the closed-form extremes, so the double band is 5e-3
        // (scaled), NOT the ~machine-eps band the FULL-spectrum test can use. This is honest partial-
        // convergence accuracy (~0.5%), not a loose catch-all.
        static fProxy PartialExtremalTol() => /*+choose[2e-2f|5e-3]*/2e-2f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.DenseVsSparseCrossCheck: DenseVsSparseCrossCheck(); break;
                case TestType.LaplacianKnownSpectrum: LaplacianKnownSpectrum(); break;
                case TestType.InverseLaplacianCrossCheck: InverseLaplacianCrossCheck(); break;
                case TestType.InverseVsEigenvaluesSymmetric: InverseVsEigenvaluesSymmetric(); break;
                case TestType.LanczosFullSpectrum: LanczosFullSpectrum(); break;
                case TestType.LanczosPartialExtremal: LanczosPartialExtremal(); break;
                case TestType.LanczosDenseVsBSM: LanczosDenseVsBSM(); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        // Same recipe as fProxySparseSolverTests.BuildDenseSPD: A = M^T M + dim*I -> strictly SPD /
        // diagonally dominant, so it has a single clearly dominant (positive) eigenvalue that power
        // iteration converges to unambiguously.
        static fProxyMxN BuildDenseSPD(ref Arena arena, int dim, uint seed)
        {
            var M = arena.fProxyRandomMat(dim, dim, -1f, 1f, seed);
            var A = Linear_OP.dot(M, M, true);
            for (int d = 0; d < dim; d++)
                A[d, d] += dim;
            return A;
        }

        // 1x1-block BSM built from a dense matrix's nonzero entries via AddValue (identical to
        // fProxySparseSolverTests.DenseToBSM1x1). nnzHint bounds the known nonzero pattern purely as
        // a perf hint; growth past it is safe (the builder's triplet state lives behind a shared
        // heap pointer). Encodes the SAME numeric operator as the dense form, so spMV(bsm,.) and the
        // dense matvec agree up to floating-point reassociation only.
        static fProxyBSM DenseToBSM1x1(ref Arena arena, in fProxyMxN A, int nnzHint)
        {
            var builder = arena.fProxyBSMBuilder(A.M_Rows, A.N_Cols, 1, 1, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (fProxy)0)
                        builder.AddValue(r, c, A[r, c]);
            return builder.ToBSM(ref arena);
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        // Boolean-return guard: record a distinguishing `code` in [1] so a silent Burst abort is
        // still diagnosable ([2]/[3] unused). Used for the powerIteration convergence flags.
        void AssertTrue(bool cond, fProxy code)
        {
            if (!cond && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = code;
                Fail[2] = (fProxy)0;
                Fail[3] = (fProxy)0;
            }
            Assert.IsTrue(cond);
        }

        // Two unit eigenvectors are equal up to an overall sign: align the sign on a's
        // largest-magnitude component (robust to a near-zero pivot), then compare elementwise.
        void AssertVecEqUpToSign(in fProxyN a, in fProxyN b, int n, fProxy absTol)
        {
            int piv = 0;
            fProxy best = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy m = math.abs(a[i]);
                if (m > best) { best = m; piv = i; }
            }
            fProxy sign = (a[piv] * b[piv] >= (fProxy)0) ? (fProxy)1 : (fProxy)(-1);

            fProxy maxErr = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy e = math.abs(a[i] - sign * b[i]);
                if (e > maxErr) maxErr = e;
            }
            if (!(maxErr <= absTol) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = maxErr;
                Fail[2] = absTol;
                Fail[3] = maxErr - absTol;
            }
            Assert.IsTrue(maxErr <= absTol);
        }

        // Residual property ||Av - lambda*v||_inf <= limit, where Av is supplied precomputed (here
        // from Sparse_OP.spMV on the BSM) and limit scales with max(1,|lambda|). Mirrors
        // fProxyEigenTests.AssertPowerResidual but takes Av directly so the BSM matvec is the thing
        // under test.
        void AssertResidual(in fProxyN Av, in fProxyN v, fProxy lambda, fProxy limitBase, int n)
        {
            fProxy maxRes = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy ri = math.abs(Av[i] - lambda * v[i]);
                if (ri > maxRes) maxRes = ri;
            }
            fProxy scale = math.abs(lambda);
            if (scale < (fProxy)1) scale = (fProxy)1;
            fProxy limit = limitBase * scale;
            if (!(maxRes <= limit) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = maxRes;
                Fail[2] = limit;
                Fail[3] = maxRes - limit;
            }
            Assert.IsTrue(maxRes <= limit);
        }

        // ---- (a) dense-vs-sparse cross-check ---------------------------------------------
        //
        // Build one SPD operator, run powerIteration on the dense form and on the 1x1-block BSM form
        // from the SAME zero-seeded v (deterministic internal seeding -> both iterations start from
        // the identical vector). Both must converge; the eigenvalues must agree closely and the
        // eigenvectors up to an overall sign. This is the sparse path's core acceptance criterion:
        // the BSM overload must reproduce the trusted dense overload's dominant eigenpair.
        void DenseVsSparseCrossCheck()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 12;
            var A = BuildDenseSPD(ref arena, dim, 20240702);
            var bsm = DenseToBSM1x1(ref arena, in A, dim * dim);

            fProxy tol = (fProxy)10 * Consts.fProxyZeroThreshold;

            // Dense reference (v starts at zero -> deterministic seeding).
            var vDense = arena.fProxyVec(dim);
            var wDense = arena.fProxyVec(dim);
            bool okDense = Eigen.powerIteration(in A, ref vDense, ref wDense, out fProxy lamDense, tol, 2000);
            AssertTrue(okDense, (fProxy)1);

            // Sparse (BSM) path from an identically zero-seeded v.
            var vSparse = arena.fProxyVec(dim);
            var wSparse = arena.fProxyVec(dim);
            bool okSparse = Eigen.powerIteration(in bsm, ref vSparse, ref wSparse, out fProxy lamSparse, tol, 2000);
            AssertTrue(okSparse, (fProxy)2);

            // Eigenvalues agree (magnitude up to ~dim+order-of-M; scale the loose tolerance).
            fProxy scale = (fProxy)1 + math.abs(lamDense);
            AssertClose(lamSparse, lamDense, LooseTol() * scale);

            // Eigenvectors agree up to an overall sign (both are unit vectors).
            AssertVecEqUpToSign(in vDense, in vSparse, dim, LooseTol());

            arena.Dispose();
        }

        // ---- (b) literature known-spectrum on the BSM path -------------------------------
        //
        // n x n 1D-Laplacian tridiagonal (diag 2, off-diag -1): eigenvalues are EXACTLY
        // lambda_k = 2 - 2*cos(k*pi/(n+1)), k=1..n. The DOMINANT (largest) is k=n. Encode it as a
        // 1x1-block BSM (tridiagonal -> nnzHint = 3*n bounds the pattern) and run powerIteration on
        // the BSM form. Assert convergence, the closed-form dominant eigenvalue (computed in double
        // then cast, mirroring fProxyEigenTests.EvSymLaplacian), and the residual A*v ~= lambda*v
        // using Sparse_OP.spMV on the BSM itself.
        void LaplacianKnownSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16;
            var A = arena.fProxyLaplacian1D(n);
            var bsm = DenseToBSM1x1(ref arena, in A, 3 * n);

            fProxy tol = (fProxy)10 * Consts.fProxyZeroThreshold;

            var v = arena.fProxyVec(n);   // zero -> deterministic seeding
            var w = arena.fProxyVec(n);
            bool ok = Eigen.powerIteration(in bsm, ref v, ref w, out fProxy lambda, tol, 4000);
            AssertTrue(ok, (fProxy)1);

            // Closed-form dominant eigenvalue (k = n), computed in double precision then cast.
            double lamD = 2.0 - 2.0 * math.cos(n * math.PI_DBL / (n + 1));
            fProxy scale = (fProxy)1 + math.abs((fProxy)lamD);
            AssertClose(lambda, (fProxy)lamD, (fProxy)1000 * Consts.fProxyZeroThreshold * scale);

            // Residual property on the BSM operator: A*v ~= lambda*v (A*v via spMV on the BSM).
            var Av = Sparse_OP.spMV(in bsm, in v);
            AssertResidual(in Av, in v, lambda, (fProxy)100 * Consts.fProxyZeroThreshold, n);

            arena.Dispose();
        }

        // ---- Milestone C2: Eigen.inversePowerIteration<TOp> (smallest eigenpair, generic over
        // IfProxyLinearOperator, inner solve via Solvers.cg<TOp>) -----------------------------
        //
        // (a)+(b) literature known-spectrum AND dense-vs-BSM cross-check on the 1D Laplacian.
        //
        // The 1D Laplacian's SMALLEST eigenvalues are well-separated (lambda_2/lambda_1 ~= 4 for
        // small k, since lambda_k ~= (k*pi/(n+1))^2 for small k/n), so inverse iteration converges
        // quickly and reliably. This is deliberately NOT built from BuildDenseSPD (M^T M + dim*I):
        // that construction is great for the DOMINANT-eigenvalue powerIteration tests above (the
        // largest eigenvalues of a square Wishart-like M^T M are well separated) but is a poor
        // fixture for inverse iteration -- a square Wishart matrix's smallest eigenvalues cluster
        // near zero, so the ratio driving inverse iteration's convergence rate is close to 1 and
        // convergence can be arbitrarily slow. The Laplacian avoids that pitfall entirely.
        //
        // Runs inversePowerIteration on BOTH the dense matrix and an equivalent 1x1-block BSM
        // (same recipe as LaplacianKnownSpectrum/DenseVsSparseCrossCheck above): asserts both
        // converge, both match the closed-form smallest eigenvalue
        // lambda_1 = 2 - 2*cos(pi/(n+1)), the two eigenvector estimates agree up to an overall
        // sign, and the BSM path's own residual A*v ~= lambda*v holds via Sparse_OP.spMV.
        //
        // Tolerances here use LooseTol() (NOT the tight "1000*zeroThreshold"/"100*zeroThreshold"
        // constants LaplacianKnownSpectrum uses for pure-matvec powerIteration): inverse iteration
        // is mediated by an INEXACT inner CG solve (bounded by cgTol, not machine epsilon), so its
        // eigenpair floor is many orders coarser than a solver that only ever does matvecs.
        void InverseLaplacianCrossCheck()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 12;
            var Adense = arena.fProxyLaplacian1D(n);
            var bsm = DenseToBSM1x1(ref arena, in Adense, 3 * n);

            // tol is a multiple of cgTol (not the much tighter Consts.fProxyZeroThreshold): the
            // outer convergence checks compare consecutive eigenpair estimates, each from its own
            // fresh CG solve accurate only to ~cgTol, so an outer tolerance tighter than that noise
            // floor could spin to maxIter without ever detecting convergence (see
            // Eigen.inversePowerIteration's no-scratch convenience overload doc comment).
            fProxy cgTol = Consts.fProxySqrtEps;
            fProxy tol = (fProxy)10 * cgTol;

            // Closed-form smallest eigenvalue (k = 1), computed in double precision then cast.
            double lamD = 2.0 - 2.0 * math.cos(1.0 * math.PI_DBL / (n + 1));
            fProxy scale = (fProxy)1 + math.abs((fProxy)lamD);

            // Dense inverse power iteration (v starts at zero -> deterministic seeding).
            var vDense = arena.fProxyVec(n);
            bool okDense = Eigen.inversePowerIteration(in Adense, ref vDense, out fProxy lamDense, tol, 200, n, cgTol);
            AssertTrue(okDense, (fProxy)1);
            AssertClose(lamDense, (fProxy)lamD, LooseTol() * scale);

            // Sparse (BSM) inverse power iteration, from an identically zero-seeded v.
            var vSparse = arena.fProxyVec(n);
            bool okSparse = Eigen.inversePowerIteration(in bsm, ref vSparse, out fProxy lamSparse, tol, 200, n, cgTol);
            AssertTrue(okSparse, (fProxy)2);
            AssertClose(lamSparse, (fProxy)lamD, LooseTol() * scale);

            // Dense-vs-BSM agreement: two INDEPENDENTLY-converged eigenvectors, up to overall sign.
            AssertVecEqUpToSign(in vDense, in vSparse, n, LooseTol());

            // Residual property on the BSM operator: A*v ~= lambda*v (A*v via spMV on the BSM).
            var Av = Sparse_OP.spMV(in bsm, in vSparse);
            AssertResidual(in Av, in vSparse, lamSparse, LooseTol(), n);

            arena.Dispose();
        }

        // ---- (c) cross-check inversePowerIteration's lambda_min against the dense full-spectrum
        // eigenvaluesSymmetric (Householder tridiagonalization + QL) on the SAME operator.
        // eigenvaluesSymmetric DESTROYS its input matrix, so it runs on an independently-built
        // copy of the Laplacian -- fProxyLaplacian1D is a pure generator, so calling it twice
        // yields two separate fProxyMxN instances encoding the identical numeric operator.
        void InverseVsEigenvaluesSymmetric()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 10;
            var A = arena.fProxyLaplacian1D(n);
            var ARef = arena.fProxyLaplacian1D(n);   // independent copy; destroyed below

            // tol is a multiple of cgTol (not the much tighter Consts.fProxyZeroThreshold): the
            // outer convergence checks compare consecutive eigenpair estimates, each from its own
            // fresh CG solve accurate only to ~cgTol, so an outer tolerance tighter than that noise
            // floor could spin to maxIter without ever detecting convergence (see
            // Eigen.inversePowerIteration's no-scratch convenience overload doc comment).
            fProxy cgTol = Consts.fProxySqrtEps;
            fProxy tol = (fProxy)10 * cgTol;

            var v = arena.fProxyVec(n);   // zero -> deterministic seeding
            bool ok = Eigen.inversePowerIteration(in A, ref v, out fProxy lambda, tol, 200, n, cgTol);
            AssertTrue(ok, (fProxy)1);

            var eigenvalues = arena.fProxyVec(n);
            bool okEig = Eigen.eigenvaluesSymmetric(ref ARef, ref eigenvalues);
            AssertTrue(okEig, (fProxy)2);

            // eigenvaluesSymmetric sorts DESCENDING -> the smallest eigenvalue is the last entry.
            fProxy smallestRef = eigenvalues[n - 1];

            fProxy scale = (fProxy)1 + math.abs(smallestRef);
            AssertClose(lambda, smallestRef, LooseTol() * scale);

            arena.Dispose();
        }

        // ---- Milestone C3: Eigen.lanczos (symmetric Lanczos tridiagonalization + Ritz values via
        // eigenvaluesSymmetric on the small tridiagonal T), generic over IfProxyLinearOperator with
        // dense (fProxyMxN) and BSM (fProxyBSM) forwarders. Same fixture philosophy as the
        // power/inverse-power suites above: the 1D Laplacian's spectrum is closed-form, and the BSM
        // path is cross-checked against the dense path encoding the SAME numeric operator.
        //
        // (a) FULL-SPECTRUM REPRODUCTION. With steps == n and full reorthogonalization, T is
        // orthogonally similar to A, so the n Ritz values reproduce A's ENTIRE spectrum. Run lanczos
        // with steps == n on BOTH the dense Laplacian and its 1x1-block BSM encoding; assert both
        // produced == n, both converged, and every Ritz value matches the closed-form eigenvalue
        // lambda_k = 2 - 2*cos(k*pi/(n+1)). lanczos sorts DESCENDING (eigenvaluesSymmetric's
        // convention) and 2-2cos is INCREASING in k, so descending output index i corresponds to
        // k = n - i (identical mapping to EvSymLaplacian). ALSO cross-check the dense Ritz values
        // against the trusted dense eigenvaluesSymmetric run on an INDEPENDENT copy of the same
        // Laplacian (eigenvaluesSymmetric destroys its input; fProxyLaplacian1D is a pure generator,
        // so a second call yields a separate instance encoding the identical operator).
        void LanczosFullSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16;
            var Adense = arena.fProxyLaplacian1D(n);
            var bsm    = DenseToBSM1x1(ref arena, in Adense, 3 * n);
            var ARef   = arena.fProxyLaplacian1D(n);   // independent copy; destroyed by eigenvaluesSymmetric below

            // Full spectrum: steps == n.
            var eigDense = Eigen.lanczos(ref arena, in Adense, n, out int producedDense, out bool convDense);
            AssertTrue(convDense, (fProxy)1);
            AssertTrue(producedDense == n, (fProxy)2);

            var eigBsm = Eigen.lanczos(ref arena, in bsm, n, out int producedBsm, out bool convBsm);
            AssertTrue(convBsm, (fProxy)3);
            AssertTrue(producedBsm == n, (fProxy)4);

            // Trusted dense reference spectrum on the independent copy.
            var eigRef = arena.fProxyVec(n);
            bool okEig = Eigen.eigenvaluesSymmetric(ref ARef, ref eigRef);
            AssertTrue(okEig, (fProxy)5);

            for (int i = 0; i < n; i++)
            {
                // Descending output -> index i corresponds to k = n - i.
                int k = n - i;
                double lamD = 2.0 - 2.0 * math.cos(k * math.PI_DBL / (n + 1));
                fProxy scale = (fProxy)1 + math.abs((fProxy)lamD);

                AssertClose(eigDense[i], (fProxy)lamD, FullSpectrumTol() * scale);
                AssertClose(eigBsm[i], (fProxy)lamD, FullSpectrumTol() * scale);

                // Dense lanczos vs dense eigenvaluesSymmetric: both within FullSpectrumTol of the
                // closed form, so their mutual difference is bounded by 2x that.
                AssertClose(eigDense[i], eigRef[i], (fProxy)2 * FullSpectrumTol() * scale);
            }

            arena.Dispose();
        }

        // (b) PARTIAL-SPECTRUM EXTREMAL CONVERGENCE. Same Laplacian, but steps ~= n/2 < n. No
        // breakdown occurs for a generic seed on the Laplacian, so produced == steps. Only the
        // EXTREMAL Ritz values are expected to have converged: eigenvalues[0] (largest, closed-form
        // k = n) and eigenvalues[produced-1] (smallest, closed-form k = 1). Interior Ritz values are
        // NOT asserted. Uses the looser PartialExtremalTol (fewer steps -> coarser floor than the
        // full-spectrum case).
        void LanczosPartialExtremal()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16;
            int steps = n / 2;   // 8
            var A = arena.fProxyLaplacian1D(n);

            var eig = Eigen.lanczos(ref arena, in A, steps, out int produced, out bool converged);
            AssertTrue(converged, (fProxy)1);
            AssertTrue(produced == steps, (fProxy)2);

            // Largest Ritz value (index 0) vs closed-form k = n.
            double lamMaxD = 2.0 - 2.0 * math.cos(n * math.PI_DBL / (n + 1));
            fProxy scaleMax = (fProxy)1 + math.abs((fProxy)lamMaxD);
            AssertClose(eig[0], (fProxy)lamMaxD, PartialExtremalTol() * scaleMax);

            // Smallest Ritz value (index produced-1) vs closed-form k = 1.
            double lamMinD = 2.0 - 2.0 * math.cos(1.0 * math.PI_DBL / (n + 1));
            fProxy scaleMin = (fProxy)1 + math.abs((fProxy)lamMinD);
            AssertClose(eig[produced - 1], (fProxy)lamMinD, PartialExtremalTol() * scaleMin);

            arena.Dispose();
        }

        // (c) DENSE-vs-BSM AGREEMENT on the partial-spectrum run. The dense and BSM forms encode the
        // SAME numeric operator, run the SAME Lanczos loop from the SAME deterministic zero-seed, so
        // they differ only by spMV-vs-dense-matvec floating-point reassociation. Every Ritz value
        // (extremal AND interior) must therefore agree closely -- a stronger cross-check than (b),
        // matching the DenseVsSparseCrossCheck philosophy. Uses LooseTol (iterative-vs-iterative
        // cross-check band).
        void LanczosDenseVsBSM()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16;
            int steps = n / 2;   // 8
            var Adense = arena.fProxyLaplacian1D(n);
            var bsm    = DenseToBSM1x1(ref arena, in Adense, 3 * n);

            var eigDense = Eigen.lanczos(ref arena, in Adense, steps, out int producedDense, out bool convDense);
            AssertTrue(convDense, (fProxy)1);

            var eigBsm = Eigen.lanczos(ref arena, in bsm, steps, out int producedBsm, out bool convBsm);
            AssertTrue(convBsm, (fProxy)2);

            AssertTrue(producedDense == producedBsm, (fProxy)3);

            for (int i = 0; i < producedDense; i++)
            {
                fProxy scale = (fProxy)1 + math.abs(eigDense[i]);
                AssertClose(eigDense[i], eigBsm[i], LooseTol() * scale);
            }

            arena.Dispose();
        }
    }

    // ---- correctness entry points (Burst job + Fail-array surfacing) ----------------------

    void RunCase(SparseEigenTestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new SparseEigenTestJob { Type = type, Fail = fail }.Run();
            // A failed in-job Assert aborts the Burst job WITHOUT throwing to the caller; surface
            // the recorded diagnostics here (same convention as fProxyEigenTests).
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/code {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/code {fail[3]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }

    [Test]
    public void DenseVsSparseCrossCheckTest()
        => RunCase(SparseEigenTestJob.TestType.DenseVsSparseCrossCheck);

    [Test]
    public void LaplacianKnownSpectrumTest()
        => RunCase(SparseEigenTestJob.TestType.LaplacianKnownSpectrum);

    [Test]
    public void InverseLaplacianCrossCheckTest()
        => RunCase(SparseEigenTestJob.TestType.InverseLaplacianCrossCheck);

    [Test]
    public void InverseVsEigenvaluesSymmetricTest()
        => RunCase(SparseEigenTestJob.TestType.InverseVsEigenvaluesSymmetric);

    [Test]
    public void LanczosFullSpectrumTest()
        => RunCase(SparseEigenTestJob.TestType.LanczosFullSpectrum);

    [Test]
    public void LanczosPartialExtremalTest()
        => RunCase(SparseEigenTestJob.TestType.LanczosPartialExtremal);

    [Test]
    public void LanczosDenseVsBSMTest()
        => RunCase(SparseEigenTestJob.TestType.LanczosDenseVsBSM);

    // ---- guard / exception cases (managed thread; Assert.Throws can't run inside Burst) ----
    //
    // The BSM overloads forward into the same generic powerIteration<TOp> core, whose argument
    // guards throw ArgumentException on: A.Rows != A.Cols, v.N != A.Rows, w.N != A.Rows, v/w
    // aliasing, and maxIter < 1. Not exhaustive -- these just prove each guard fires on the BSM
    // entry point (matching fProxyEigenTests' Power* throw tests, but via fProxyBSM).

    // A square 4x4 (two 2x2 diagonal blocks) BSM -- both diagonal blocks present, well-formed.
    static fProxyBSM BuildSquareBSM(ref Arena arena)
    {
        const int BR = 2, BC = 2;
        var builder = arena.fProxyBSMBuilder(2, 2, BR, BC, 2);
        builder.AddBlock(0, 0, arena.fProxyRandomMat(BR, BC, -1f, 1f, 71001));
        builder.AddBlock(1, 1, arena.fProxyRandomMat(BR, BC, -1f, 1f, 71002));
        return builder.ToBSM(ref arena);
    }

    [Test]
    public void Power_NonSquareBSM_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            // 2x3 block grid of 2x2 blocks -> 4x6 (Rows != Cols). One block suffices; the
            // Rows != Cols guard fires before v/w are examined.
            const int BR = 2, BC = 2;
            var builder = arena.fProxyBSMBuilder(2, 3, BR, BC, 1);
            builder.AddBlock(0, 0, arena.fProxyRandomMat(BR, BC, -1f, 1f, 71101));
            var A = builder.ToBSM(ref arena);

            var v = arena.fProxyVec(A.M_Rows);
            var w = arena.fProxyVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, Consts.fProxyZeroThreshold, 1000));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Power_WrongVLength_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);   // 4x4
            var v = arena.fProxyVec(A.M_Rows - 1); // wrong length
            var w = arena.fProxyVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, Consts.fProxyZeroThreshold, 1000));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Power_AliasingVAndW_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);   // 4x4
            var v = arena.fProxyVec(A.M_Rows);
            var wAlias = v; // w aliases v (struct copy shares Data.Ptr) -> guard must fire
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref wAlias, out fProxy lambda, Consts.fProxyZeroThreshold, 1000));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Power_BadMaxIter_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);   // 4x4
            var v = arena.fProxyVec(A.M_Rows);
            var w = arena.fProxyVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, Consts.fProxyZeroThreshold, 0));
        }
        finally { arena.Dispose(); }
    }

    // ---- inversePowerIteration guard / exception cases (managed thread) -------------------
    //
    // The BSM overloads forward into the same generic inversePowerIteration<TOp> core, whose
    // argument guards throw ArgumentException on: A.Rows != A.Cols, v/y/r/p/Ap length mismatch,
    // v/y/r/p/Ap aliasing, and maxIter < 1. Not exhaustive -- these just prove each guard fires
    // on the BSM entry point (mirrors the Power_*_Throws tests above).

    [Test]
    public void InversePower_NonSquareBSM_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            // 2x3 block grid of 2x2 blocks -> 4x6 (Rows != Cols). One block suffices; the
            // Rows != Cols guard fires before v is examined.
            const int BR = 2, BC = 2;
            var builder = arena.fProxyBSMBuilder(2, 3, BR, BC, 1);
            builder.AddBlock(0, 0, arena.fProxyRandomMat(BR, BC, -1f, 1f, 71201));
            var A = builder.ToBSM(ref arena);

            var v = arena.fProxyVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.inversePowerIteration(in A, ref v, out fProxy lambda));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void InversePower_WrongVLength_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);      // 4x4
            var v = arena.fProxyVec(A.M_Rows - 1);  // wrong length
            Assert.Throws<ArgumentException>(() =>
                Eigen.inversePowerIteration(in A, ref v, out fProxy lambda));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void InversePower_AliasingVAndY_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);   // 4x4
            var v  = arena.fProxyVec(A.M_Rows);
            var r  = arena.fProxyVec(A.M_Rows);
            var p  = arena.fProxyVec(A.M_Rows);
            var Ap = arena.fProxyVec(A.M_Rows);
            var yAlias = v; // y aliases v (struct copy shares Data.Ptr) -> guard must fire
            Assert.Throws<ArgumentException>(() =>
                Eigen.inversePowerIteration(in A, ref v, ref yAlias, ref r, ref p, ref Ap, out fProxy lambda,
                    Consts.fProxyZeroThreshold, 1000, A.M_Rows, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void InversePower_BadMaxIter_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);   // 4x4
            var v = arena.fProxyVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.inversePowerIteration(in A, ref v, out fProxy lambda,
                    Consts.fProxyZeroThreshold, 0, A.M_Rows, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    // ---- lanczos guard / exception cases (managed thread; Assert.Throws can't run in Burst) ----
    //
    // lanczos<TOp>'s argument guards throw ArgumentException, checked in this order (see the core):
    // A.Rows != A.Cols  ->  steps not in [1, A.Rows]  ->  breakdownTol < 0  ->  workspace shape  ->
    // eigenvalues.N != steps  ->  vCur/w aliasing. These tests fire the first four via both the
    // dense (in fProxyMxN) and BSM (in fProxyBSM) entry points where practical, using the
    // ref-workspace overload with a valid workspace so the intended guard (not workspace shape)
    // is the one that trips.

    [Test]
    public void Lanczos_NonSquareDense_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(3, 4);                 // Rows != Cols
            var ws = arena.fProxyLanczos_WS(A.M_Rows, 1);  // sized for n = 3, steps = 1
            var eig = arena.fProxyVec(1);
            // Square guard fires before the workspace/eigenvalues shape is examined.
            Assert.Throws<ArgumentException>(() =>
                Eigen.lanczos(in A, ref ws, ref eig, out int produced, 1));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Lanczos_NonSquareBSM_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            // 2x3 block grid of 2x2 blocks -> 4x6 (Rows != Cols). One block suffices.
            const int BR = 2, BC = 2;
            var builder = arena.fProxyBSMBuilder(2, 3, BR, BC, 1);
            builder.AddBlock(0, 0, arena.fProxyRandomMat(BR, BC, -1f, 1f, 73001));
            var A = builder.ToBSM(ref arena);

            var ws = arena.fProxyLanczos_WS(A.M_Rows, 1);  // n = 4, steps = 1
            var eig = arena.fProxyVec(1);
            Assert.Throws<ArgumentException>(() =>
                Eigen.lanczos(in A, ref ws, ref eig, out int produced, 1));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Lanczos_StepsTooSmall_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyLaplacian1D(4);            // square
            var ws = arena.fProxyLanczos_WS(4, 1);
            var eig = arena.fProxyVec(1);
            // steps = 0 < 1: the [1, A.Rows] guard fires before workspace/eigenvalues are checked.
            Assert.Throws<ArgumentException>(() =>
                Eigen.lanczos(in A, ref ws, ref eig, out int produced, 0));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Lanczos_StepsTooLarge_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyLaplacian1D(4);            // square, n = 4
            var ws = arena.fProxyLanczos_WS(4, 5);         // workspace validly sized for steps = 5
            var eig = arena.fProxyVec(5);
            // steps = 5 > A.Rows = 4: the [1, A.Rows] guard fires.
            Assert.Throws<ArgumentException>(() =>
                Eigen.lanczos(in A, ref ws, ref eig, out int produced, 5));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Lanczos_NegativeBreakdownTol_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyLaplacian1D(4);            // square
            var ws = arena.fProxyLanczos_WS(4, 2);
            var eig = arena.fProxyVec(2);
            // breakdownTol < 0: guard fires (this is the breakdownTol-taking overload).
            Assert.Throws<ArgumentException>(() =>
                Eigen.lanczos(in A, ref ws, ref eig, out int produced, 2, (fProxy)(-1)));
        }
        finally { arena.Dispose(); }
    }
}
