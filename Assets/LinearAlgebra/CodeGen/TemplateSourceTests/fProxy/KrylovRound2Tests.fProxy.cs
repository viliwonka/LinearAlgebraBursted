using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Krylov Round-2 new surfaces:
//   (a) IfProxyLinearOperator.ApplyDot on every shipped operator -- each composes (Apply, then
//       Blas.dot(x,y)), so ApplyDot MUST equal Apply-then-dot BIT-EXACTLY (same matVec/spMV
//       kernel both times, same 2-arg vecDot fold): asserted with double-cast AreEqual.
//   (b) fProxyBlockJacobi.Apply's blockJacobiApplyB{1,2,3,4,6} unrolls vs the general runtime-BR
//       loop -- documented BIT-IDENTICAL (same left-to-right fold): asserted EXACT against an
//       in-test replica of the general loop reading DInv, PLUS an independent D_i*z_i==r_i check.
//       BR=5 and BR=7 exercise the general-loop fallback (correctness only).
//   (c) Accumulator-paired square-block spMV kernels bsrMatVec[Sym/T]B{2,3,4,6} -- rounding-only
//       vs the dense expansion: multi-block rows with BOTH even and odd stored-block counts
//       stress the pair loop and its scalar tail. (SparseUnrollTests already sweeps these vs the
//       dense oracle at <=2 blocks/row; this adds the >2-blocks/row tail path explicitly.)
//   (d) cg/pcg on a small BSR SPD system agree with a dense LU oracle -- pins the ApplyDot
//       routing end-to-end (cg/pcg's pAp = ApplyDot(p, Ap)).
//
// Value cases run inside a [BurstCompile] IJob (matches every other sparse suite). The rectangular
// ApplyDot dimension-mismatch throw (e) is a managed [Test] with Assert.Catch (Burst cannot
// surface an assertable managed exception).
public class fProxyKrylovRound2Tests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct KrylovRound2TestJob : IJob
    {
        public enum TestType
        {
            // (a) ApplyDot == Apply-then-Blas.dot, EXACT.
            ApplyDotDenseExact,
            ApplyDotBSRFullExact,
            ApplyDotBSRSymExact,
            ApplyDotIdentityExact,
            ApplyDotColScaledSquareExact,
            ApplyDotNormalOperatorExact,

            // (b) blockJacobiApply unrolls (EXACT vs general fold) + fallback (correctness).
            BlockJacobiB1, BlockJacobiB2, BlockJacobiB3, BlockJacobiB4, BlockJacobiB6,
            BlockJacobiB5Fallback, BlockJacobiB7Fallback,

            // (c) paired spMV kernels, >2-blocks/row tail stress, full / transposed / symmetric.
            PairedSpMVFull, PairedSpMVT, PairedSpMVSym,

            // (d) cg / pcg over BSR match a dense LU oracle.
            CgBsrMatchesLUOracle,
            PcgBsrMatchesLUOracle,
        }

        public TestType Type;

        // Paired-kernel block sizes. Declared static readonly (not an inline `int[]` in a method
        // body): Burst rejects constructing a managed array inside a job (BC1028), but a
        // statically-initialized readonly reference compiles -- same idiom as the R1
        // fProxyKrylovFusedKernelTests.Sizes field.
        static readonly int[] PairedBs = { 2, 3, 4, 6 };

        // spMV / dense-oracle agreement: values in [-1,1], a handful of products per output.
        static fProxy SpTol() => /*+choose[1e-4f|1e-11]*/1e-4f/*-choose*/;
        // iterative-solve vs direct-oracle agreement (matches fProxySparseSolverTests.Tol).
        static fProxy SolveTol() => /*+choose[1e-3f|1e-7]*/1e-3f/*-choose*/;
        // block-Jacobi inverse-apply residual D_i*z_i==r_i (an explicit small inverse solve).
        static fProxy BjTol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ApplyDotDenseExact: ApplyDotDenseExact(); break;
                case TestType.ApplyDotBSRFullExact: ApplyDotBSRFullExact(); break;
                case TestType.ApplyDotBSRSymExact: ApplyDotBSRSymExact(); break;
                case TestType.ApplyDotIdentityExact: ApplyDotIdentityExact(); break;
                case TestType.ApplyDotColScaledSquareExact: ApplyDotColScaledSquareExact(); break;
                case TestType.ApplyDotNormalOperatorExact: ApplyDotNormalOperatorExact(); break;

                case TestType.BlockJacobiB1: CheckBlockJacobi(1, 81000u, true); break;
                case TestType.BlockJacobiB2: CheckBlockJacobi(2, 82000u, true); break;
                case TestType.BlockJacobiB3: CheckBlockJacobi(3, 83000u, true); break;
                case TestType.BlockJacobiB4: CheckBlockJacobi(4, 84000u, true); break;
                case TestType.BlockJacobiB6: CheckBlockJacobi(6, 86000u, true); break;
                case TestType.BlockJacobiB5Fallback: CheckBlockJacobi(5, 85000u, false); break;
                case TestType.BlockJacobiB7Fallback: CheckBlockJacobi(7, 87000u, false); break;

                case TestType.PairedSpMVFull: PairedSpMVFull(); break;
                case TestType.PairedSpMVT: PairedSpMVT(); break;
                case TestType.PairedSpMVSym: PairedSpMVSym(); break;

                case TestType.CgBsrMatchesLUOracle: CgBsrMatchesLUOracle(); break;
                case TestType.PcgBsrMatchesLUOracle: PcgBsrMatchesLUOracle(); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        static void AssertExactVec(in fProxyN got, in fProxyN expected)
        {
            Assert.AreEqual(expected.N, got.N);
            for (int i = 0; i < got.N; i++)
                Assert.AreEqual((double)expected[i], (double)got[i]);   // bit-exact
        }

        static void AssertClose(fProxy got, fProxy expected, fProxy tol)
            => Assert.IsTrue(math.abs(got - expected) <= tol * ((fProxy)1 + math.abs(expected)));

        static void AssertVecClose(in fProxyN got, in fProxyN expected, fProxy tol)
        {
            Assert.AreEqual(expected.N, got.N);
            for (int i = 0; i < got.N; i++) AssertClose(got[i], expected[i], tol);
        }

        // Reference y = A^T x from the dense expansion (independent of Blas.trans), mirroring
        // fProxySparseUnrollTests.DenseTransMatVec.
        static void DenseTransMatVec(in fProxyMxN dense, in fProxyN x, ref fProxyN y)
        {
            for (int j = 0; j < dense.N_Cols; j++)
            {
                fProxy s = 0;
                for (int i = 0; i < dense.M_Rows; i++) s += dense[i, j] * x[i];
                y[j] = s;
            }
        }

        // SPD b x b block D = M^T M + b*I: symmetric, well-conditioned -> LU-invertible, and
        // D_i z_i == r_i holds tightly.
        static fProxyMxN SpdBlock(ref Arena arena, int b, uint seed)
        {
            var M = arena.fProxyRandomMat(b, b, -1f, 1f, seed);
            var D = Blas.dot(M, M, true);   // M^T M (symmetric, PSD)
            for (int d = 0; d < b; d++) D[d, d] += (fProxy)b;
            return D;
        }

        // Same SPD recipe as fProxySparseSolverTests.BuildDenseSPD.
        static fProxyMxN BuildDenseSPD(ref Arena arena, int dim, uint seed)
        {
            var M = arena.fProxyRandomMat(dim, dim, -1f, 1f, seed);
            var A = Blas.dot(M, M, true);
            for (int d = 0; d < dim; d++) A[d, d] += dim;
            return A;
        }

        static fProxyBSR DenseToBSR1x1(ref Arena arena, in fProxyMxN A, int nnzHint)
        {
            var builder = arena.fProxyBSRBuilder(A.M_Rows, A.N_Cols, 1, 1, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (fProxy)0) builder.AddValue(r, c, A[r, c]);
            return builder.ToBSR(ref arena);
        }

        // ==============================================================================
        // (a) ApplyDot == Apply, then Blas.dot(x, y) -- BIT-EXACT for every operator.
        //     ApplyDot fills y and returns d; the reference re-runs Apply into a fresh y2 and
        //     takes Blas.dot(x, y2). Same kernel + same 2-arg dot fold => exact.
        // ==============================================================================

        static void CheckApplyDotExact<TOp>(in TOp op, in fProxyN x, ref Arena arena)
            where TOp : struct, IfProxyLinearOperator
        {
            int rows = op.Rows;
            var y1 = arena.fProxyVec(rows);
            fProxy d1 = op.ApplyDot(in x, ref y1);

            var y2 = arena.fProxyVec(rows);
            op.Apply(in x, ref y2);
            fProxy d2 = Blas.dot(x, y2);

            AssertExactVec(in y1, in y2);
            Assert.AreEqual((double)d2, (double)d1);   // bit-exact scalar
        }

        void ApplyDotDenseExact()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 11;
            var A = BuildDenseSPD(ref arena, n, 70001);      // square so dot(x, A x) is well-formed
            var x = arena.fProxyRandomVec(n, -1f, 1f, 70002);
            CheckApplyDotExact(new fProxyDenseOperator(in A), in x, ref arena);
            arena.Dispose();
        }

        void ApplyDotBSRFullExact()
        {
            var arena = new Arena(Allocator.Persistent);
            // Square 3x3-block BSR (full storage), several scattered blocks.
            const int b = 3, nb = 4;
            int dim = b * nb;
            var builder = arena.fProxyBSRBuilder(nb, nb, b, b, 8);
            builder.AddBlock(0, 0, arena.fProxyRandomMat(b, b, -1f, 1f, 70101));
            builder.AddBlock(0, 2, arena.fProxyRandomMat(b, b, -1f, 1f, 70102));
            builder.AddBlock(1, 1, arena.fProxyRandomMat(b, b, -1f, 1f, 70103));
            builder.AddBlock(1, 3, arena.fProxyRandomMat(b, b, -1f, 1f, 70104));
            builder.AddBlock(2, 0, arena.fProxyRandomMat(b, b, -1f, 1f, 70105));
            builder.AddBlock(3, 3, arena.fProxyRandomMat(b, b, -1f, 1f, 70106));
            var A = builder.ToBSR(ref arena);
            var x = arena.fProxyRandomVec(dim, -1f, 1f, 70110);
            CheckApplyDotExact(new fProxyBSROperator(in A), in x, ref arena);
            arena.Dispose();
        }

        void ApplyDotBSRSymExact()
        {
            var arena = new Arena(Allocator.Persistent);
            const int b = 3, nb = 4;
            int dim = b * nb;
            var builder = arena.fProxyBSRBuilder(nb, nb, b, b, 8);
            builder.AddBlock(0, 0, SpdBlock(ref arena, b, 70201));   // symmetric diagonal blocks
            builder.AddBlock(1, 1, SpdBlock(ref arena, b, 70202));
            builder.AddBlock(2, 2, SpdBlock(ref arena, b, 70203));
            builder.AddBlock(3, 3, SpdBlock(ref arena, b, 70204));
            builder.AddBlock(0, 1, arena.fProxyRandomMat(b, b, -1f, 1f, 70205));  // upper off-diagonals
            builder.AddBlock(1, 3, arena.fProxyRandomMat(b, b, -1f, 1f, 70206));
            var A = builder.ToBSRSymmetric(ref arena);
            var x = arena.fProxyRandomVec(dim, -1f, 1f, 70210);
            CheckApplyDotExact(new fProxyBSROperator(in A), in x, ref arena);
            arena.Dispose();
        }

        void ApplyDotIdentityExact()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 9;
            var x = arena.fProxyRandomVec(n, -1f, 1f, 70301);
            CheckApplyDotExact(new fProxyIdentityOperator(n), in x, ref arena);
            arena.Dispose();
        }

        void ApplyDotColScaledSquareExact()
        {
            var arena = new Arena(Allocator.Persistent);
            // SQUARE inner so x and y share a length and dot(x, y) is well-formed (the rectangular
            // case's dimension throw is the managed test below).
            int n = 8;
            var A = BuildDenseSPD(ref arena, n, 70401);
            var d = arena.fProxyRandomVec(n, (fProxy)0.5f, (fProxy)2f, 70402);   // nonzero scale
            var scratch = arena.fProxyVec(n);
            var op = new fProxyColScaledOperator<fProxyDenseOperator>(new fProxyDenseOperator(in A), d, scratch);
            var x = arena.fProxyRandomVec(n, -1f, 1f, 70403);
            CheckApplyDotExact(in op, in x, ref arena);
            arena.Dispose();
        }

        void ApplyDotNormalOperatorExact()
        {
            var arena = new Arena(Allocator.Persistent);
            // LP normal operator M = As D As^T + reg*I over a rectangular BSR inner (m x n).
            int m = 6, n = 4;
            var Adense = arena.fProxyRandomMat(m, n, -1f, 1f, 70501);
            var bsm = DenseToBSR1x1(ref arena, in Adense, m * n);
            var inner = new fProxyBSROperator(in bsm);
            var d = arena.fProxyRandomVec(n, (fProxy)0.25f, (fProxy)1.5f, 70502);  // length As.Cols
            var scratch = arena.fProxyVec(n);
            var op = new fProxyNormalOperator<fProxyBSROperator>(in inner, in d, in scratch, (fProxy)0.3f);
            var x = arena.fProxyRandomVec(m, -1f, 1f, 70503);   // M is m x m -> x length m
            CheckApplyDotExact(in op, in x, ref arena);
            arena.Dispose();
        }

        // ==============================================================================
        // (b) fProxyBlockJacobi.Apply: unrolled kernels (b in {1,2,3,4,6}) vs the general loop,
        //     BIT-IDENTICAL; b=5/7 fall through to the general loop (correctness only).
        // ==============================================================================

        // Block-diagonal BSR (only diagonal blocks -> ToDense is exactly blockdiag(D_i)).
        static fProxyBSR BuildBlockDiagBSR(ref Arena arena, int nb, int b, uint seed)
        {
            var builder = arena.fProxyBSRBuilder(nb, nb, b, b, nb);
            for (int i = 0; i < nb; i++)
                builder.AddBlock(i, i, SpdBlock(ref arena, b, seed + (uint)i + 1u));
            return builder.ToBSR(ref arena);
        }

        void CheckBlockJacobi(int b, uint seed, bool checkFold)
        {
            var arena = new Arena(Allocator.Persistent);

            int nb = 4;
            int dim = nb * b;
            var A = BuildBlockDiagBSR(ref arena, nb, b, seed);
            var M = arena.fProxyBlockJacobi(in A);

            var r = arena.fProxyRandomVec(dim, -1f, 1f, seed + 500u);
            var z = arena.fProxyVec(dim);
            M.Apply(in r, ref z);                       // dispatches to unroll (b in {1,2,3,4,6}) or general loop

            if (checkFold)
            {
                // Replica of the general runtime-BR loop (UnsafeOP.blockJacobiApply general fallback
                // in fProxyBlockJacobi.Apply), reading the SAME DInv the unroll reads. Same
                // left-to-right fold order => the unrolled result must be bit-identical.
                var dinv = M.DInv;
                int blockLen = b * b;
                var zRef = arena.fProxyVec(dim);
                for (int i = 0; i < nb; i++)
                {
                    int rowBase = i * b;
                    int blockOff = i * blockLen;
                    for (int lr = 0; lr < b; lr++)
                    {
                        fProxy sum = 0;
                        for (int lc = 0; lc < b; lc++)
                            sum += dinv[blockOff + lr * b + lc] * r[rowBase + lc];
                        zRef[rowBase + lr] = sum;
                    }
                }
                AssertExactVec(in z, in zRef);
            }

            // Independent correctness (also the "hand-computed dense reference" for the b=5/7
            // fallback): z = M^-1 r <=> D_i z_i == r_i for every diagonal block. D_i is read from
            // the dense expansion, NOT from DInv -> genuinely independent of the applied inverse.
            var dense = A.ToDense(ref arena);
            for (int i = 0; i < nb; i++)
            {
                int rowBase = i * b;
                for (int lr = 0; lr < b; lr++)
                {
                    fProxy prod = 0;
                    for (int lc = 0; lc < b; lc++)
                        prod += dense[rowBase + lr, rowBase + lc] * z[rowBase + lc];
                    AssertClose(prod, r[rowBase + lr], BjTol());
                }
            }

            arena.Dispose();
        }

        // ==============================================================================
        // (c) Paired square-block spMV kernels vs the dense expansion. Rows carry BOTH even and
        //     odd stored-block counts (>2 per row) so the pair loop runs multiple iterations and
        //     the odd tail (the `if (k < rowEnd)` scalar remainder) fires. Swept over b in
        //     {2,3,4,6} (the paired sizes). B1 is intentionally excluded (not accumulator-paired).
        // ==============================================================================

        // Non-symmetric 5x5-block grid: rows with 4, 3, 2, 1, 0 stored blocks.
        static fProxyBSR BuildFullMultiBlock(ref Arena arena, int b, uint seed)
        {
            var builder = arena.fProxyBSRBuilder(5, 5, b, b, 16);
            builder.AddBlock(0, 0, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 1u));   // row0: 4 (even)
            builder.AddBlock(0, 1, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 2u));
            builder.AddBlock(0, 2, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 3u));
            builder.AddBlock(0, 3, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 4u));
            builder.AddBlock(1, 0, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 5u));   // row1: 3 (odd)
            builder.AddBlock(1, 2, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 6u));
            builder.AddBlock(1, 4, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 7u));
            builder.AddBlock(2, 1, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 8u));   // row2: 2 (even)
            builder.AddBlock(2, 3, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 9u));
            builder.AddBlock(3, 3, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 10u));  // row3: 1 (odd)
            return builder.ToBSR(ref arena);                                          // row4: empty
        }

        // Symmetric (upper-triangle) 5x5-block grid: SPD diagonal at every block-row plus upper
        // off-diagonals giving rows with 4, 3, 2 stored blocks.
        static fProxyBSR BuildSymMultiBlock(ref Arena arena, int b, uint seed)
        {
            var builder = arena.fProxyBSRBuilder(5, 5, b, b, 16);
            for (int i = 0; i < 5; i++)
                builder.AddBlock(i, i, SpdBlock(ref arena, b, seed + (uint)i + 1u));
            builder.AddBlock(0, 1, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 11u));  // row0: diag + 3
            builder.AddBlock(0, 2, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 12u));
            builder.AddBlock(0, 4, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 13u));
            builder.AddBlock(1, 3, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 14u));  // row1: diag + 2
            builder.AddBlock(1, 4, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 15u));
            builder.AddBlock(2, 3, arena.fProxyRandomMat(b, b, -1f, 1f, seed + 16u));  // row2: diag + 1
            return builder.ToBSRSymmetric(ref arena);
        }

        void PairedSpMVFull()
        {
            var arena = new Arena(Allocator.Persistent);
            for (int t = 0; t < PairedBs.Length; t++)
            {
                int b = PairedBs[t];
                var A = BuildFullMultiBlock(ref arena, b, (uint)(91000 + b * 100));
                var dense = A.ToDense(ref arena);
                var x = arena.fProxyRandomVec(A.N_Cols, -1f, 1f, (uint)(91500 + b));
                var y = arena.fProxyVec(A.M_Rows);
                BSR.spMV(in A, in x, ref y);
                AssertVecClose(in y, Blas.dot(dense, x), SpTol());
            }
            arena.Dispose();
        }

        void PairedSpMVT()
        {
            var arena = new Arena(Allocator.Persistent);
            for (int t = 0; t < PairedBs.Length; t++)
            {
                int b = PairedBs[t];
                var A = BuildFullMultiBlock(ref arena, b, (uint)(92000 + b * 100));
                var dense = A.ToDense(ref arena);
                var xt = arena.fProxyRandomVec(A.M_Rows, -1f, 1f, (uint)(92500 + b));
                var yt = arena.fProxyVec(A.N_Cols);
                BSR.spMVT(in A, in xt, ref yt);
                var ytRef = arena.fProxyVec(A.N_Cols);
                DenseTransMatVec(in dense, in xt, ref ytRef);
                AssertVecClose(in yt, in ytRef, SpTol());
            }
            arena.Dispose();
        }

        void PairedSpMVSym()
        {
            var arena = new Arena(Allocator.Persistent);
            for (int t = 0; t < PairedBs.Length; t++)
            {
                int b = PairedBs[t];
                var A = BuildSymMultiBlock(ref arena, b, (uint)(93000 + b * 100));
                var dense = A.ToDense(ref arena);
                var x = arena.fProxyRandomVec(A.N_Cols, -1f, 1f, (uint)(93500 + b));

                var y = arena.fProxyVec(A.M_Rows);
                BSR.spMV(in A, in x, ref y);
                AssertVecClose(in y, Blas.dot(dense, x), SpTol());

                // A == A^T: spMVT compared to an independent dense transpose-matvec.
                var ytRef = arena.fProxyVec(A.N_Cols);
                DenseTransMatVec(in dense, in x, ref ytRef);
                var yt = arena.fProxyVec(A.N_Cols);
                BSR.spMVT(in A, in x, ref yt);
                AssertVecClose(in yt, in ytRef, SpTol());
            }
            arena.Dispose();
        }

        // ==============================================================================
        // (d) cg / pcg over a BSR SPD system match a dense LU oracle -- exercises ApplyDot end to
        //     end (cg/pcg compute pAp = op.ApplyDot(p, Ap) internally).
        // ==============================================================================

        void CgBsrMatchesLUOracle()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 12;
            var A = BuildDenseSPD(ref arena, dim, 94001);
            var bsm = DenseToBSR1x1(ref arena, in A, dim * dim);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 94002);

            // Dense LU oracle on COPIES (decompInPlace/decompSolve are destructive).
            var LUcopy = A.Copy();
            var pivot = new Pivot(dim, Allocator.Temp);
            bool okLU = LU.decompInPlace(ref LUcopy, ref pivot);
            Assert.IsTrue(okLU);
            var xLU = b.Copy();
            LU.decompSolve(ref LUcopy, in pivot, ref xLU);
            pivot.Dispose();

            var xCg = arena.fProxyVec(dim);
            bool okCg = Krylov.cg(in bsm, in b, ref xCg, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okCg);
            AssertVecClose(in xCg, in xLU, SolveTol());

            var Ax = BSR.spMV(in bsm, in xCg);
            AssertVecClose(in Ax, in b, SolveTol());

            arena.Dispose();
        }

        void PcgBsrMatchesLUOracle()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 12;
            var A = BuildDenseSPD(ref arena, dim, 95001);
            var bsm = DenseToBSR1x1(ref arena, in A, dim * dim);
            var M = arena.fProxyBlockJacobi(in bsm);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 95002);

            var LUcopy = A.Copy();
            var pivot = new Pivot(dim, Allocator.Temp);
            bool okLU = LU.decompInPlace(ref LUcopy, ref pivot);
            Assert.IsTrue(okLU);
            var xLU = b.Copy();
            LU.decompSolve(ref LUcopy, in pivot, ref xLU);
            pivot.Dispose();

            var xPcg = arena.fProxyVec(dim);
            bool okPcg = Krylov.pcg(in bsm, in M, in b, ref xPcg, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okPcg);
            AssertVecClose(in xPcg, in xLU, SolveTol());

            var Ax = BSR.spMV(in bsm, in xPcg);
            AssertVecClose(in Ax, in b, SolveTol());

            arena.Dispose();
        }
    }

    // ---- (a) ApplyDot exact ----
    [Test] public void ApplyDotDenseExactTest()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.ApplyDotDenseExact }.Run();
    [Test] public void ApplyDotBSRFullExactTest()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.ApplyDotBSRFullExact }.Run();
    [Test] public void ApplyDotBSRSymExactTest()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.ApplyDotBSRSymExact }.Run();
    [Test] public void ApplyDotIdentityExactTest()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.ApplyDotIdentityExact }.Run();
    [Test] public void ApplyDotColScaledSquareExactTest()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.ApplyDotColScaledSquareExact }.Run();
    [Test] public void ApplyDotNormalOperatorExactTest()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.ApplyDotNormalOperatorExact }.Run();

    // ---- (b) block-Jacobi unrolls + fallback ----
    [Test] public void BlockJacobiB1Test()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.BlockJacobiB1 }.Run();
    [Test] public void BlockJacobiB2Test()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.BlockJacobiB2 }.Run();
    [Test] public void BlockJacobiB3Test()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.BlockJacobiB3 }.Run();
    [Test] public void BlockJacobiB4Test()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.BlockJacobiB4 }.Run();
    [Test] public void BlockJacobiB6Test()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.BlockJacobiB6 }.Run();
    [Test] public void BlockJacobiB5FallbackTest()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.BlockJacobiB5Fallback }.Run();
    [Test] public void BlockJacobiB7FallbackTest()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.BlockJacobiB7Fallback }.Run();

    // ---- (c) paired spMV tail stress ----
    [Test] public void PairedSpMVFullTest()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.PairedSpMVFull }.Run();
    [Test] public void PairedSpMVTTest()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.PairedSpMVT }.Run();
    [Test] public void PairedSpMVSymTest()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.PairedSpMVSym }.Run();

    // ---- (d) cg/pcg BSR vs LU oracle ----
    [Test] public void CgBsrMatchesLUOracleTest()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.CgBsrMatchesLUOracle }.Run();
    [Test] public void PcgBsrMatchesLUOracleTest()
        => new KrylovRound2TestJob { Type = KrylovRound2TestJob.TestType.PcgBsrMatchesLUOracle }.Run();

    // ==============================================================================
    // (e) Rectangular fProxyColScaledOperator.ApplyDot: Apply is fine (x length Cols, y length
    //     Rows) but the following Blas.dot(x, y) sees mismatched lengths (Cols != Rows) and throws
    //     ArgumentException -- the same dimension guard Blas.dot itself enforces. Managed [Test] +
    //     Assert.Catch (a Burst job cannot surface an assertable managed exception).
    // ==============================================================================
    [Test]
    public void ColScaledRectangularApplyDotThrowsOnDimensionMismatch()
    {
        var arena = new Arena(Allocator.Persistent);

        int m = 7, n = 4;                                   // rectangular inner -> Rows != Cols
        var A = arena.fProxyRandomMat(m, n, -1f, 1f, 96001);
        var d = arena.fProxyRandomVec(n, (fProxy)0.5f, (fProxy)2f, 96002);
        var scratch = arena.fProxyVec(n);
        var op = new fProxyColScaledOperator<fProxyDenseOperator>(new fProxyDenseOperator(in A), d, scratch);

        var x = arena.fProxyRandomVec(n, -1f, 1f, 96003);   // length Cols = n
        var y = arena.fProxyVec(m);                         // length Rows = m

        Assert.Catch<ArgumentException>(() => op.ApplyDot(in x, ref y));

        arena.Dispose();
    }
}
