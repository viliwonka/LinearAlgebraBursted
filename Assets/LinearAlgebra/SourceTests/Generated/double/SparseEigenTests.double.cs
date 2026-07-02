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
// the new powerIteration(in doubleBSM, ...) overloads forward through doubleBSMOperator into the
// same generic core the dense powerIteration(in doubleMxN, ...) path uses. Every sparse result is
// cross-checked against the pre-existing dense path (same recipe as doubleSparseSolverTests: build
// the SAME operator in both forms and compare), plus one literature known-spectrum case.
//
// Correctness cases run inside a [BurstCompile] IJob (matches doubleSparseSolverTests /
// doubleEigenTests) and use doubleEigenTests' Fail-NativeArray diagnostic convention: a failed
// Assert inside a Burst job aborts silently without surfacing to the runner, so every check first
// records [0]=flag, [1]=got, [2]=expected/limit, [3]=diff into Fail. Guard/exception cases run on
// the managed test thread with Assert.Throws, since NUnit's Assert.Throws cannot execute in Burst.
public class doubleSparseEigenTests
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
            LanczosEarlyBreakdownPadding,
            PowerNegativeDominant,
            LanczosVectorsResidualAndOrthonormal,
            LanczosVectorsDenseVsBSM,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/code
        public NativeArray<double> Fail;

        // Two INDEPENDENTLY-converged iterative eigenpairs (dense manual-matvec vs BSM spMV) are
        // compared, so a machine-epsilon threshold is inappropriate: mirror doubleSparseSolverTests'
        // choose-marker tolerance for iterative-vs-iterative cross-checks (looser on float).
        static double LooseTol() => 1e-5;

        // Full-spectrum Lanczos (steps == n, full reorthogonalization TWICE) makes the tridiagonal
        // T orthogonally similar to A, so its Ritz values are as accurate as running
        // eigenvaluesSymmetric on A directly -- i.e. bounded by the QL eigensolver's own floor plus
        // the reorthogonalization roundoff. This is tighter than the iterative LooseTol but a touch
        // looser than EvSymLaplacian's 1000*ZeroThreshold, to absorb that extra roundoff on the
        // larger n=16 tridiagonal used here. Applied as a (1+|lambda|)-scaled absolute tolerance.
        static double FullSpectrumTol() => 1e-9;

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
        static double PartialExtremalTol() => 5e-3;

        // Breakdown-detection threshold for the early-breakdown test's diagonal operator (||A|| < 1).
        // Must sit ABOVE the reorthogonalization roundoff floor (~eps*||A|| ~ 1e-7 float / 1e-16
        // double) so the grade-2 residual is recognized as zero, yet FAR BELOW the real off-diagonal
        // beta_1 (~0.25 here) so the break fires at the true grade, not one step early.
        static double BreakdownTol() => 1e-9;

        // Accuracy band for the two grade-exact Ritz values of that diagonal operator. eigenvaluesSymmetric
        // on the resulting 2x2 tridiagonal is essentially exact; the only error is the ~eps*||A|| floor
        // in the Lanczos-produced alpha/beta, so this is tight but not machine-eps.
        static double BreakdownRitzTol() => 1e-9;

        // Ritz VECTOR accuracy (residual ‖Av-λv‖, unit norm, pairwise orthogonality) at full
        // steps == n: T is orthogonally similar to A, so the Ritz vectors are A's eigenvectors to
        // eigenSymmetric's floor plus reorthogonalization roundoff -- comparable to FullSpectrumTol.
        static double VecTol() => 1e-8;

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
                case TestType.LanczosEarlyBreakdownPadding: LanczosEarlyBreakdownPadding(); break;
                case TestType.PowerNegativeDominant: PowerNegativeDominant(); break;
                case TestType.LanczosVectorsResidualAndOrthonormal: LanczosVectorsResidualAndOrthonormal(); break;
                case TestType.LanczosVectorsDenseVsBSM: LanczosVectorsDenseVsBSM(); break;
                case TestType.LanczosDenseVsBSM: LanczosDenseVsBSM(); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        // Same recipe as doubleSparseSolverTests.BuildDenseSPD: A = M^T M + dim*I -> strictly SPD /
        // diagonally dominant, so it has a single clearly dominant (positive) eigenvalue that power
        // iteration converges to unambiguously.
        static doubleMxN BuildDenseSPD(ref Arena arena, int dim, uint seed)
        {
            var M = arena.doubleRandomMat(dim, dim, -1f, 1f, seed);
            var A = Linear_OP.dot(M, M, true);
            for (int d = 0; d < dim; d++)
                A[d, d] += dim;
            return A;
        }

        // 1x1-block BSM built from a dense matrix's nonzero entries via AddValue (identical to
        // doubleSparseSolverTests.DenseToBSM1x1). nnzHint bounds the known nonzero pattern purely as
        // a perf hint; growth past it is safe (the builder's triplet state lives behind a shared
        // heap pointer). Encodes the SAME numeric operator as the dense form, so spMV(bsm,.) and the
        // dense matvec agree up to floating-point reassociation only.
        static doubleBSM DenseToBSM1x1(ref Arena arena, in doubleMxN A, int nnzHint)
        {
            var builder = arena.doubleBSMBuilder(A.M_Rows, A.N_Cols, 1, 1, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (double)0)
                        builder.AddValue(r, c, A[r, c]);
            return builder.ToBSM(ref arena);
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        void AssertClose(double a, double b, double precision)
        {
            double diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        // Boolean-return guard: record a distinguishing `code` in [1] so a silent Burst abort is
        // still diagnosable ([2]/[3] unused). Used for the powerIteration convergence flags.
        void AssertTrue(bool cond, double code)
        {
            if (!cond && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = code;
                Fail[2] = (double)0;
                Fail[3] = (double)0;
            }
            Assert.IsTrue(cond);
        }

        // Two unit eigenvectors are equal up to an overall sign: align the sign on a's
        // largest-magnitude component (robust to a near-zero pivot), then compare elementwise.
        void AssertVecEqUpToSign(in doubleN a, in doubleN b, int n, double absTol)
        {
            int piv = 0;
            double best = (double)0;
            for (int i = 0; i < n; i++)
            {
                double m = math.abs(a[i]);
                if (m > best) { best = m; piv = i; }
            }
            double sign = (a[piv] * b[piv] >= (double)0) ? (double)1 : (double)(-1);

            double maxErr = (double)0;
            for (int i = 0; i < n; i++)
            {
                double e = math.abs(a[i] - sign * b[i]);
                if (e > maxErr) maxErr = e;
            }
            if (!(maxErr <= absTol) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = maxErr;
                Fail[2] = absTol;
                Fail[3] = maxErr - absTol;
            }
            Assert.IsTrue(maxErr <= absTol);
        }

        // Residual property ||Av - lambda*v||_inf <= limit, where Av is supplied precomputed (here
        // from Sparse_OP.spMV on the BSM) and limit scales with max(1,|lambda|). Mirrors
        // doubleEigenTests.AssertPowerResidual but takes Av directly so the BSM matvec is the thing
        // under test.
        void AssertResidual(in doubleN Av, in doubleN v, double lambda, double limitBase, int n)
        {
            double maxRes = (double)0;
            for (int i = 0; i < n; i++)
            {
                double ri = math.abs(Av[i] - lambda * v[i]);
                if (ri > maxRes) maxRes = ri;
            }
            double scale = math.abs(lambda);
            if (scale < (double)1) scale = (double)1;
            double limit = limitBase * scale;
            if (!(maxRes <= limit) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
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

            double tol = (double)10 * Consts.doubleZeroThreshold;

            // Dense reference (v starts at zero -> deterministic seeding).
            var vDense = arena.doubleVec(dim);
            var wDense = arena.doubleVec(dim);
            bool okDense = Eigen.powerIteration(in A, ref vDense, ref wDense, out double lamDense, tol, 2000);
            AssertTrue(okDense, (double)1);

            // Sparse (BSM) path from an identically zero-seeded v.
            var vSparse = arena.doubleVec(dim);
            var wSparse = arena.doubleVec(dim);
            bool okSparse = Eigen.powerIteration(in bsm, ref vSparse, ref wSparse, out double lamSparse, tol, 2000);
            AssertTrue(okSparse, (double)2);

            // Eigenvalues agree (magnitude up to ~dim+order-of-M; scale the loose tolerance).
            double scale = (double)1 + math.abs(lamDense);
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
        // then cast, mirroring doubleEigenTests.EvSymLaplacian), and the residual A*v ~= lambda*v
        // using Sparse_OP.spMV on the BSM itself.
        void LaplacianKnownSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16;
            var A = arena.doubleLaplacian1D(n);
            var bsm = DenseToBSM1x1(ref arena, in A, 3 * n);

            double tol = (double)10 * Consts.doubleZeroThreshold;

            var v = arena.doubleVec(n);   // zero -> deterministic seeding
            var w = arena.doubleVec(n);
            bool ok = Eigen.powerIteration(in bsm, ref v, ref w, out double lambda, tol, 4000);
            AssertTrue(ok, (double)1);

            // Closed-form dominant eigenvalue (k = n), computed in double precision then cast.
            double lamD = 2.0 - 2.0 * math.cos(n * math.PI_DBL / (n + 1));
            double scale = (double)1 + math.abs((double)lamD);
            AssertClose(lambda, (double)lamD, (double)1000 * Consts.doubleZeroThreshold * scale);

            // Residual property on the BSM operator: A*v ~= lambda*v (A*v via spMV on the BSM).
            var Av = Sparse_OP.spMV(in bsm, in v);
            AssertResidual(in Av, in v, lambda, (double)100 * Consts.doubleZeroThreshold, n);

            arena.Dispose();
        }

        // ---- Milestone C2: Eigen.inversePowerIteration<TOp> (smallest eigenpair, generic over
        // IdoubleLinearOperator, inner solve via Solvers.cg<TOp>) -----------------------------
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
            var Adense = arena.doubleLaplacian1D(n);
            var bsm = DenseToBSM1x1(ref arena, in Adense, 3 * n);

            // tol is a multiple of cgTol (not the much tighter Consts.doubleZeroThreshold): the
            // outer convergence checks compare consecutive eigenpair estimates, each from its own
            // fresh CG solve accurate only to ~cgTol, so an outer tolerance tighter than that noise
            // floor could spin to maxIter without ever detecting convergence (see
            // Eigen.inversePowerIteration's no-scratch convenience overload doc comment).
            double cgTol = Consts.doubleSqrtEps;
            double tol = (double)10 * cgTol;

            // Closed-form smallest eigenvalue (k = 1), computed in double precision then cast.
            double lamD = 2.0 - 2.0 * math.cos(1.0 * math.PI_DBL / (n + 1));
            double scale = (double)1 + math.abs((double)lamD);

            // Dense inverse power iteration (v starts at zero -> deterministic seeding).
            var vDense = arena.doubleVec(n);
            bool okDense = Eigen.inversePowerIteration(in Adense, ref vDense, out double lamDense, tol, 200, n, cgTol);
            AssertTrue(okDense, (double)1);
            AssertClose(lamDense, (double)lamD, LooseTol() * scale);

            // Sparse (BSM) inverse power iteration, from an identically zero-seeded v.
            var vSparse = arena.doubleVec(n);
            bool okSparse = Eigen.inversePowerIteration(in bsm, ref vSparse, out double lamSparse, tol, 200, n, cgTol);
            AssertTrue(okSparse, (double)2);
            AssertClose(lamSparse, (double)lamD, LooseTol() * scale);

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
        // copy of the Laplacian -- doubleLaplacian1D is a pure generator, so calling it twice
        // yields two separate doubleMxN instances encoding the identical numeric operator.
        void InverseVsEigenvaluesSymmetric()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 10;
            var A = arena.doubleLaplacian1D(n);
            var ARef = arena.doubleLaplacian1D(n);   // independent copy; destroyed below

            // tol is a multiple of cgTol (not the much tighter Consts.doubleZeroThreshold): the
            // outer convergence checks compare consecutive eigenpair estimates, each from its own
            // fresh CG solve accurate only to ~cgTol, so an outer tolerance tighter than that noise
            // floor could spin to maxIter without ever detecting convergence (see
            // Eigen.inversePowerIteration's no-scratch convenience overload doc comment).
            double cgTol = Consts.doubleSqrtEps;
            double tol = (double)10 * cgTol;

            var v = arena.doubleVec(n);   // zero -> deterministic seeding
            bool ok = Eigen.inversePowerIteration(in A, ref v, out double lambda, tol, 200, n, cgTol);
            AssertTrue(ok, (double)1);

            var eigenvalues = arena.doubleVec(n);
            bool okEig = Eigen.eigenvaluesSymmetric(ref ARef, ref eigenvalues);
            AssertTrue(okEig, (double)2);

            // eigenvaluesSymmetric sorts DESCENDING -> the smallest eigenvalue is the last entry.
            double smallestRef = eigenvalues[n - 1];

            double scale = (double)1 + math.abs(smallestRef);
            AssertClose(lambda, smallestRef, LooseTol() * scale);

            arena.Dispose();
        }

        // ---- Milestone C3: Eigen.lanczos (symmetric Lanczos tridiagonalization + Ritz values via
        // eigenvaluesSymmetric on the small tridiagonal T), generic over IdoubleLinearOperator with
        // dense (doubleMxN) and BSM (doubleBSM) forwarders. Same fixture philosophy as the
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
        // Laplacian (eigenvaluesSymmetric destroys its input; doubleLaplacian1D is a pure generator,
        // so a second call yields a separate instance encoding the identical operator).
        void LanczosFullSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16;
            var Adense = arena.doubleLaplacian1D(n);
            var bsm    = DenseToBSM1x1(ref arena, in Adense, 3 * n);
            var ARef   = arena.doubleLaplacian1D(n);   // independent copy; destroyed by eigenvaluesSymmetric below

            // Full spectrum: steps == n.
            var eigDense = Eigen.lanczos(ref arena, in Adense, n, out int producedDense, out bool convDense);
            AssertTrue(convDense, (double)1);
            AssertTrue(producedDense == n, (double)2);

            var eigBsm = Eigen.lanczos(ref arena, in bsm, n, out int producedBsm, out bool convBsm);
            AssertTrue(convBsm, (double)3);
            AssertTrue(producedBsm == n, (double)4);

            // Trusted dense reference spectrum on the independent copy.
            var eigRef = arena.doubleVec(n);
            bool okEig = Eigen.eigenvaluesSymmetric(ref ARef, ref eigRef);
            AssertTrue(okEig, (double)5);

            for (int i = 0; i < n; i++)
            {
                // Descending output -> index i corresponds to k = n - i.
                int k = n - i;
                double lamD = 2.0 - 2.0 * math.cos(k * math.PI_DBL / (n + 1));
                double scale = (double)1 + math.abs((double)lamD);

                AssertClose(eigDense[i], (double)lamD, FullSpectrumTol() * scale);
                AssertClose(eigBsm[i], (double)lamD, FullSpectrumTol() * scale);

                // Dense lanczos vs dense eigenvaluesSymmetric: both within FullSpectrumTol of the
                // closed form, so their mutual difference is bounded by 2x that.
                AssertClose(eigDense[i], eigRef[i], (double)2 * FullSpectrumTol() * scale);
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
            var A = arena.doubleLaplacian1D(n);

            var eig = Eigen.lanczos(ref arena, in A, steps, out int produced, out bool converged);
            AssertTrue(converged, (double)1);
            AssertTrue(produced == steps, (double)2);

            // Largest Ritz value (index 0) vs closed-form k = n.
            double lamMaxD = 2.0 - 2.0 * math.cos(n * math.PI_DBL / (n + 1));
            double scaleMax = (double)1 + math.abs((double)lamMaxD);
            AssertClose(eig[0], (double)lamMaxD, PartialExtremalTol() * scaleMax);

            // Smallest Ritz value (index produced-1) vs closed-form k = 1.
            double lamMinD = 2.0 - 2.0 * math.cos(1.0 * math.PI_DBL / (n + 1));
            double scaleMin = (double)1 + math.abs((double)lamMinD);
            AssertClose(eig[produced - 1], (double)lamMinD, PartialExtremalTol() * scaleMin);

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
            var Adense = arena.doubleLaplacian1D(n);
            var bsm    = DenseToBSM1x1(ref arena, in Adense, 3 * n);

            var eigDense = Eigen.lanczos(ref arena, in Adense, steps, out int producedDense, out bool convDense);
            AssertTrue(convDense, (double)1);

            var eigBsm = Eigen.lanczos(ref arena, in bsm, steps, out int producedBsm, out bool convBsm);
            AssertTrue(convBsm, (double)2);

            AssertTrue(producedDense == producedBsm, (double)3);

            for (int i = 0; i < producedDense; i++)
            {
                double scale = (double)1 + math.abs(eigDense[i]);
                AssertClose(eigDense[i], eigBsm[i], LooseTol() * scale);
            }

            arena.Dispose();
        }

        // (d) EARLY-BREAKDOWN + GERSHGORIN-PADDING PATH. The Laplacian tests always run to
        // produced == steps and so never exercise lanczos's `produced < steps` branch (the most
        // numerically subtle new code: it pads T with a decoupled junk block below a Gershgorin
        // bound so sorting can't mix padding into the real Ritz values). Force it here with a
        // DIAGONAL operator whose spectrum has only TWO distinct values (0.2 x3, 0.7 x3). Lanczos on
        // a diagonal matrix has Krylov grade == the count of distinct eigenvalues the seed touches;
        // the deterministic seed (1,2,3,4,...) is nonzero on both eigenspaces, so grade == 2. With
        // steps == 4 the process breaks down at produced == 2 and pads slots [2,4). Assert (i) the
        // break is detected at exactly the grade, (ii) the 2 real Ritz values reproduce the distinct
        // eigenvalues, and (iii) the padding sorts STRICTLY AFTER every real value and never leaks
        // into [0, produced) -- the sort-order guarantee the padding construction rests on. Runs on
        // BOTH the dense and 1x1-BSM encodings so the sparse eigensolver path is covered too.
        void LanczosEarlyBreakdownPadding()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            int steps = 4;

            var A = arena.doubleMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (double)0;
            for (int i = 0; i < n; i++)
                A[i, i] = i < 3 ? (double)0.2 : (double)0.7;

            double bTol = BreakdownTol();

            // --- dense path ---
            var eig = Eigen.lanczos(ref arena, in A, steps, out int produced, out bool converged, bTol);

            AssertTrue(produced == 2, (double)1);                 // breakdown detected at the true grade
            AssertClose(eig[0], (double)0.7, BreakdownRitzTol()); // real Ritz values reproduce the
            AssertClose(eig[1], (double)0.2, BreakdownRitzTol()); // two distinct eigenvalues, descending

            // Padding sort-order guarantee. Real values are both >= 0.2 > 0; padding is
            // -bound - k*max(1,bound) with bound >= 0.7 (Gershgorin bounds the max real Ritz value),
            // so every padded slot is <= -1.7. The smallest real value must NOT be displaced, the
            // first padding value must sort after it (below -bound), and padding slots stay ordered.
            AssertTrue(eig[1] > (double)0.1, (double)2);          // real min not overwritten by padding
            AssertTrue(eig[2] < (double)(-0.7), (double)3);       // first padding sorts after reals, below -bound
            AssertTrue(eig[3] < eig[2], (double)4);               // padding slots strictly, distinctly ordered

            // --- sparse (1x1-BSM) path: same operator, same breakdown, same real Ritz values ---
            var bsm = DenseToBSM1x1(ref arena, in A, n);
            var eigB = Eigen.lanczos(ref arena, in bsm, steps, out int producedB, out bool convB, bTol);

            AssertTrue(producedB == 2, (double)5);
            AssertClose(eigB[0], (double)0.7, BreakdownRitzTol());
            AssertClose(eigB[1], (double)0.2, BreakdownRitzTol());

            arena.Dispose();
        }

        // (e) NEGATIVE-DOMINANT POWER ITERATION. Every other power-iteration fixture is SPD
        // (positive spectrum), so powerIteration's sign-alternation branch -- where the
        // largest-MAGNITUDE eigenvalue is negative and the iterate flips sign each step -- is never
        // exercised. Negate a well-separated SPD matrix: A = -(M^T M + dim*I) is symmetric with a
        // single dominant-magnitude eigenvalue that is NEGATIVE. Assert powerIteration converges,
        // returns lambda < 0, and matches the most-negative eigenvalue from a trusted
        // eigenvaluesSymmetric run on an independent copy (descending sort -> index dim-1).
        void PowerNegativeDominant()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 10;
            var A = BuildDenseSPD(ref arena, dim, 7714);   // dominant eigenvalue lamMax > 0
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    A[i, j] = -A[i, j];                    // now dominant-magnitude eigenvalue is -lamMax < 0

            // Independent copy for the trusted reference spectrum (eigenvaluesSymmetric destroys it).
            var ARef = BuildDenseSPD(ref arena, dim, 7714);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    ARef[i, j] = -ARef[i, j];

            var v = arena.doubleVec(dim);   // zero -> internal deterministic seeding
            var w = arena.doubleVec(dim);
            double tol = (double)10 * Consts.doubleZeroThreshold;
            bool ok = Eigen.powerIteration(in A, ref v, ref w, out double lambda, tol, 4000);

            AssertTrue(ok, (double)1);
            AssertTrue(lambda < (double)0, (double)2);   // sign branch: dominant eigenvalue is negative

            var eigRef = arena.doubleVec(dim);
            bool okEig = Eigen.eigenvaluesSymmetric(ref ARef, ref eigRef);
            AssertTrue(okEig, (double)3);

            // Descending sort -> the most-negative (largest-magnitude) eigenvalue is at index dim-1.
            double lamRef = eigRef[dim - 1];
            double scale = (double)1 + math.abs(lamRef);
            AssertClose(lambda, lamRef, LooseTol() * scale);

            arena.Dispose();
        }

        // ---- Ritz VECTORS (lanczosVectors): approximate eigenvectors ---------------------
        //
        // Full-spectrum Lanczos on the Laplacian returns approximate eigenVECTORS. Each must satisfy
        // the eigenpair residual ‖A v_i - lambda_i v_i‖ ≈ 0, be unit-norm, and be mutually orthogonal
        // -- the sign-free, closed-form-free correctness criteria for eigenvectors (the Ritz VALUES
        // are already pinned by LanczosFullSpectrum). Exercises the shared lanczosTridiag +
        // eigenSymmetric + Ritz-combination path.
        void LanczosVectorsResidualAndOrthonormal()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 12;
            var A = arena.doubleLaplacian1D(n);

            var eig = Eigen.lanczosVectors(ref arena, in A, n, out var ritz, out int produced, out bool conv);
            AssertTrue(conv, (double)1);
            AssertTrue(produced == n, (double)2);

            var v = arena.doubleVec(n);
            for (int i = 0; i < produced; i++)
            {
                for (int c = 0; c < n; c++) v[c] = ritz[i, c];

                // unit norm
                double nrmSq = (double)0;
                for (int c = 0; c < n; c++) nrmSq += v[c] * v[c];
                AssertClose(math.sqrt(nrmSq), (double)1, VecTol());

                // eigenpair residual ‖A v - lambda v‖_inf, scaled by max(1,|lambda|)
                var Av = Linear_OP.dot(A, v);
                double maxRes = (double)0;
                for (int c = 0; c < n; c++)
                {
                    double ri = math.abs(Av[c] - eig[i] * v[c]);
                    if (ri > maxRes) maxRes = ri;
                }
                double scale = math.abs(eig[i]);
                if (scale < (double)1) scale = (double)1;
                AssertClose(maxRes, (double)0, VecTol() * scale);
            }

            // pairwise orthogonality of the Ritz vectors
            for (int i = 0; i < produced; i++)
                for (int j = i + 1; j < produced; j++)
                {
                    double d = (double)0;
                    for (int c = 0; c < n; c++) d += ritz[i, c] * ritz[j, c];
                    AssertClose(d, (double)0, VecTol());
                }

            arena.Dispose();
        }

        // Dense and 1x1-BSM encodings of the same Laplacian must yield the SAME Ritz values and
        // (up to a per-vector sign) the SAME Ritz vectors -- confirms the BSM operator feeds
        // lanczosVectors identically to the dense path.
        void LanczosVectorsDenseVsBSM()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 12;
            var Adense = arena.doubleLaplacian1D(n);
            var bsm    = DenseToBSM1x1(ref arena, in Adense, 3 * n);

            var eigD = Eigen.lanczosVectors(ref arena, in Adense, n, out var ritzD, out int prodD, out bool convD);
            AssertTrue(convD, (double)1);
            var eigB = Eigen.lanczosVectors(ref arena, in bsm, n, out var ritzB, out int prodB, out bool convB);
            AssertTrue(convB, (double)2);
            AssertTrue(prodD == prodB, (double)3);

            var vD = arena.doubleVec(n);
            var vB = arena.doubleVec(n);
            for (int i = 0; i < prodD; i++)
            {
                double scale = (double)1 + math.abs(eigD[i]);
                AssertClose(eigD[i], eigB[i], LooseTol() * scale);

                for (int c = 0; c < n; c++) { vD[c] = ritzD[i, c]; vB[c] = ritzB[i, c]; }
                AssertVecEqUpToSign(in vD, in vB, n, LooseTol());
            }

            arena.Dispose();
        }
    }

    // ---- correctness entry points (Burst job + Fail-array surfacing) ----------------------

    void RunCase(SparseEigenTestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new SparseEigenTestJob { Type = type, Fail = fail }.Run();
            // A failed in-job Assert aborts the Burst job WITHOUT throwing to the caller; surface
            // the recorded diagnostics here (same convention as doubleEigenTests).
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/code {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (double)0)
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

    [Test]
    public void LanczosVectorsResidualAndOrthonormalTest()
        => RunCase(SparseEigenTestJob.TestType.LanczosVectorsResidualAndOrthonormal);

    [Test]
    public void LanczosVectorsDenseVsBSMTest()
        => RunCase(SparseEigenTestJob.TestType.LanczosVectorsDenseVsBSM);

    [Test]
    public void LanczosEarlyBreakdownPaddingTest()
        => RunCase(SparseEigenTestJob.TestType.LanczosEarlyBreakdownPadding);

    [Test]
    public void PowerNegativeDominantTest()
        => RunCase(SparseEigenTestJob.TestType.PowerNegativeDominant);

    // ---- guard / exception cases (managed thread; Assert.Throws can't run inside Burst) ----
    //
    // The BSM overloads forward into the same generic powerIteration<TOp> core, whose argument
    // guards throw ArgumentException on: A.Rows != A.Cols, v.N != A.Rows, w.N != A.Rows, v/w
    // aliasing, and maxIter < 1. Not exhaustive -- these just prove each guard fires on the BSM
    // entry point (matching doubleEigenTests' Power* throw tests, but via doubleBSM).

    // A square 4x4 (two 2x2 diagonal blocks) BSM -- both diagonal blocks present, well-formed.
    static doubleBSM BuildSquareBSM(ref Arena arena)
    {
        const int BR = 2, BC = 2;
        var builder = arena.doubleBSMBuilder(2, 2, BR, BC, 2);
        builder.AddBlock(0, 0, arena.doubleRandomMat(BR, BC, -1f, 1f, 71001));
        builder.AddBlock(1, 1, arena.doubleRandomMat(BR, BC, -1f, 1f, 71002));
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
            var builder = arena.doubleBSMBuilder(2, 3, BR, BC, 1);
            builder.AddBlock(0, 0, arena.doubleRandomMat(BR, BC, -1f, 1f, 71101));
            var A = builder.ToBSM(ref arena);

            var v = arena.doubleVec(A.M_Rows);
            var w = arena.doubleVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out double lambda, Consts.doubleZeroThreshold, 1000));
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
            var v = arena.doubleVec(A.M_Rows - 1); // wrong length
            var w = arena.doubleVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out double lambda, Consts.doubleZeroThreshold, 1000));
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
            var v = arena.doubleVec(A.M_Rows);
            var wAlias = v; // w aliases v (struct copy shares Data.Ptr) -> guard must fire
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref wAlias, out double lambda, Consts.doubleZeroThreshold, 1000));
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
            var v = arena.doubleVec(A.M_Rows);
            var w = arena.doubleVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out double lambda, Consts.doubleZeroThreshold, 0));
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
            var builder = arena.doubleBSMBuilder(2, 3, BR, BC, 1);
            builder.AddBlock(0, 0, arena.doubleRandomMat(BR, BC, -1f, 1f, 71201));
            var A = builder.ToBSM(ref arena);

            var v = arena.doubleVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.inversePowerIteration(in A, ref v, out double lambda));
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
            var v = arena.doubleVec(A.M_Rows - 1);  // wrong length
            Assert.Throws<ArgumentException>(() =>
                Eigen.inversePowerIteration(in A, ref v, out double lambda));
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
            var v  = arena.doubleVec(A.M_Rows);
            var r  = arena.doubleVec(A.M_Rows);
            var p  = arena.doubleVec(A.M_Rows);
            var Ap = arena.doubleVec(A.M_Rows);
            var yAlias = v; // y aliases v (struct copy shares Data.Ptr) -> guard must fire
            Assert.Throws<ArgumentException>(() =>
                Eigen.inversePowerIteration(in A, ref v, ref yAlias, ref r, ref p, ref Ap, out double lambda,
                    Consts.doubleZeroThreshold, 1000, A.M_Rows, Consts.doubleSqrtEps));
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
            var v = arena.doubleVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.inversePowerIteration(in A, ref v, out double lambda,
                    Consts.doubleZeroThreshold, 0, A.M_Rows, Consts.doubleSqrtEps));
        }
        finally { arena.Dispose(); }
    }

    // ---- lanczos guard / exception cases (managed thread; Assert.Throws can't run in Burst) ----
    //
    // lanczos<TOp>'s argument guards throw ArgumentException, checked in this order (see the core):
    // A.Rows != A.Cols  ->  steps not in [1, A.Rows]  ->  breakdownTol < 0  ->  workspace shape  ->
    // eigenvalues.N != steps  ->  vCur/w aliasing. These tests fire the first four via both the
    // dense (in doubleMxN) and BSM (in doubleBSM) entry points where practical, using the
    // ref-workspace overload with a valid workspace so the intended guard (not workspace shape)
    // is the one that trips.

    [Test]
    public void Lanczos_NonSquareDense_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.doubleMat(3, 4);                 // Rows != Cols
            var ws = arena.doubleLanczos_WS(A.M_Rows, 1);  // sized for n = 3, steps = 1
            var eig = arena.doubleVec(1);
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
            var builder = arena.doubleBSMBuilder(2, 3, BR, BC, 1);
            builder.AddBlock(0, 0, arena.doubleRandomMat(BR, BC, -1f, 1f, 73001));
            var A = builder.ToBSM(ref arena);

            var ws = arena.doubleLanczos_WS(A.M_Rows, 1);  // n = 4, steps = 1
            var eig = arena.doubleVec(1);
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
            var A = arena.doubleLaplacian1D(4);            // square
            var ws = arena.doubleLanczos_WS(4, 1);
            var eig = arena.doubleVec(1);
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
            var A = arena.doubleLaplacian1D(4);            // square, n = 4
            var ws = arena.doubleLanczos_WS(4, 5);         // workspace validly sized for steps = 5
            var eig = arena.doubleVec(5);
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
            var A = arena.doubleLaplacian1D(4);            // square
            var ws = arena.doubleLanczos_WS(4, 2);
            var eig = arena.doubleVec(2);
            // breakdownTol < 0: guard fires (this is the breakdownTol-taking overload).
            Assert.Throws<ArgumentException>(() =>
                Eigen.lanczos(in A, ref ws, ref eig, out int produced, 2, (double)(-1)));
        }
        finally { arena.Dispose(); }
    }
}
