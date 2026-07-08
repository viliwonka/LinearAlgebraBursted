using System;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

using LinearAlgebra;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    /// <summary>
    /// Symmetric normal-equations operator M = Aₛ · diag(D) · Aₛᵀ, presented matrix-free over any inner
    /// <typeparamref name="TInner"/> operator Aₛ -- never forms M. This is the SPD system an interior-
    /// point LP solves each iteration (D = Z S⁻¹, the primal/dual diagonal, changes every iteration), so
    /// it lets <see cref="Krylov.pcg{TOp,TPre}"/> solve the normal equations directly. Also a general
    /// "AᵀDA / A D Aᵀ" operator usable beyond LP.
    ///
    /// <c>Rows == Cols == Aₛ.Rows</c> (M is Aₛ.Rows square). Apply(v) = Aₛ (D ∘ (Aₛᵀ v)). Symmetric, so
    /// <c>ApplyT == Apply</c>. Holds the inner operator, the diagonal <c>D</c> (length Aₛ.Cols -- a
    /// handle the caller rewrites in place each iteration), and one owned <c>Scratch</c> of length
    /// Aₛ.Cols. Readonly, same shape as <see cref="doubleColScaledOperator{TInner}"/>.
    /// </summary>
    public readonly struct doubleNormalOperator<TInner> : IdoubleLinearOperator
        where TInner : struct, IdoubleLinearOperator
    {
        public readonly TInner As;
        public readonly doubleN D;        // length As.Cols; the (mutable) diagonal
        public readonly doubleN Scratch;  // length As.Cols
        public readonly double Reg;       // primal-dual regularization: M := Aₛ D Aₛᵀ + Reg·I

        public doubleNormalOperator(in TInner aS, in doubleN d, in doubleN scratch, double reg)
        {
            if (d.N != aS.Cols)
                throw new ArgumentException("doubleNormalOperator: D.N must equal As.Cols");
            if (scratch.N != aS.Cols)
                throw new ArgumentException("doubleNormalOperator: scratch.N must equal As.Cols");
            As = aS; D = d; Scratch = scratch; Reg = reg;
        }

        public int Rows => As.Rows;
        public int Cols => As.Rows;   // M is m×m

        public void Apply(in doubleN v, ref doubleN y)
        {
            var w = Scratch;                          // handle copy -> writable through the interface
            As.ApplyT(in v, ref w);                   // w = Aₛᵀ v            (length Cols(Aₛ))
            for (int j = 0; j < w.N; j++) w[j] *= D[j];
            As.Apply(in w, ref y);                    // y = Aₛ (D ∘ Aₛᵀ v)   (length Rows(Aₛ))
            if (Reg != (double)0) for (int i = 0; i < y.N; i++) y[i] += Reg * v[i];   // + Reg·I
        }

        public void ApplyT(in doubleN v, ref doubleN y) => Apply(in v, ref y);   // symmetric

        // Per-row apply (M is symmetric m×m); never the hot path for LP -- present only to satisfy the
        // interface. Two bounded Temp scratch vectors per call.
        public void ApplyBlock(in doubleMxN Vrows, ref doubleMxN AVrows, int rows)
        {
            int cols = Vrows.N_Cols;
            var rin = new doubleN(cols, Allocator.Temp, false);
            var rout = new doubleN(cols, Allocator.Temp, false);
            for (int i = 0; i < rows; i++)
            {
                for (int c = 0; c < cols; c++) rin[c] = Vrows[i, c];
                Apply(in rin, ref rout);
                for (int c = 0; c < cols; c++) AVrows[i, c] = rout[c];
            }
            rout.Dispose();
            rin.Dispose();
        }
    }

    /// <summary>
    /// Diagonal Jacobi preconditioner z = diag(M)⁻¹ r for the interior-point normal equations
    /// M = Aₛ D Aₛᵀ. diag(M) is computed matrix-free once per iteration (see
    /// <c>doubleLadOperator.NormalDiagonal</c> / <see cref="Sparse.BSR.rowSquaredWeighted"/>) and this
    /// preconditioner just stores its reciprocal. Cheap, and the natural first preconditioner for the
    /// PCG inner solve (M ill-conditions as the interior point approaches the boundary). Holds the
    /// <c>InvDiag</c> handle (length m); the caller rewrites its contents each iteration.
    /// </summary>
    public readonly struct doubleNormalJacobi : IdoublePreconditioner
    {
        public readonly doubleN InvDiag;   // length m; 1 / diag(M)

        public doubleNormalJacobi(in doubleN invDiag) { InvDiag = invDiag; }

        public void Apply(in doubleN r, ref doubleN z)
        {
            for (int i = 0; i < r.N; i++) z[i] = r[i] * InvDiag[i];
        }
    }
}

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Matrix-free constraint operator for the least-absolute-deviation LP standard form. Given a sparse
    /// design matrix A (m×n, BSR), presents Aₛ = [A | −A | −I | I] (m × (2n+2m)) over the split variables
    /// z = [x⁺(n) | x⁻(n) | u(m) | v(m)]:
    ///   Apply(z)  = A x⁺ − A x⁻ − u + v                 (length m)
    ///   ApplyT(r) = [Aᵀr ; −Aᵀr ; −r ; r]               (length 2n+2m)
    /// Fully matrix-free over A (two spMV / one spMVT plus copies). Holds A and three owned scratch
    /// vectors (Sp length n for the x⁺/x⁻ slices, Tm length m, Atr length n for Aᵀr).
    ///
    /// Also computes the interior-point Jacobi diagonal diag(Aₛ D Aₛᵀ) via <see cref="NormalDiagonal"/>.
    /// </summary>
    public readonly struct doubleLadOperator : IdoubleLinearOperator
    {
        public readonly doubleBSR A;
        public readonly int M, N;
        public readonly doubleN Sp;    // length N
        public readonly doubleN Tm;    // length M
        public readonly doubleN Atr;   // length N

        public doubleLadOperator(in doubleBSR a, in doubleN sp, in doubleN tm, in doubleN atr)
        {
            A = a; M = a.M_Rows; N = a.N_Cols;
            if (sp.N != N) throw new ArgumentException("doubleLadOperator: sp.N must equal A.N_Cols");
            if (tm.N != M) throw new ArgumentException("doubleLadOperator: tm.N must equal A.M_Rows");
            if (atr.N != N) throw new ArgumentException("doubleLadOperator: atr.N must equal A.N_Cols");
            Sp = sp; Tm = tm; Atr = atr;
        }

        public int Rows => M;
        public int Cols => 2 * N + 2 * M;

        public void Apply(in doubleN z, ref doubleN y)
        {
            var sp = Sp; var tm = Tm;
            for (int j = 0; j < N; j++) sp[j] = z[j];              // x⁺
            BSR.spMV(in A, in sp, ref y);                          // y = A x⁺
            for (int j = 0; j < N; j++) sp[j] = z[N + j];          // x⁻
            BSR.spMV(in A, in sp, ref tm);                         // tm = A x⁻
            for (int i = 0; i < M; i++) y[i] = y[i] - tm[i] - z[2 * N + i] + z[2 * N + M + i];
        }

        public void ApplyT(in doubleN r, ref doubleN outv)
        {
            var atr = Atr;
            BSR.spMVT(in A, in r, ref atr);                        // atr = Aᵀ r   (length N)
            for (int j = 0; j < N; j++) { outv[j] = atr[j]; outv[N + j] = -atr[j]; }
            for (int i = 0; i < M; i++) { outv[2 * N + i] = -r[i]; outv[2 * N + M + i] = r[i]; }
        }

        // diag(Aₛ diag(D) Aₛᵀ)_i, with D = [d⁺(n) | d⁻(n) | dᵤ(m) | dᵥ(m)]:
        //   Σ_j A[i,j]²·(d⁺[j] + d⁻[j])  +  dᵤ[i]  +  dᵥ[i]
        // (the −A block contributes (−A[i,j])² = A[i,j]², the ±I blocks each contribute 1²).
        public void NormalDiagonal(in doubleN D, ref doubleN diag)
        {
            var sp = Sp;
            for (int j = 0; j < N; j++) sp[j] = D[j] + D[N + j];   // w = d⁺ + d⁻
            BSR.rowSquaredWeighted(in A, in sp, ref diag);        // diag[i] = Σ_j A[i,j]² w[j]
            for (int i = 0; i < M; i++) diag[i] += D[2 * N + i] + D[2 * N + M + i];
        }

        public void ApplyBlock(in doubleMxN Vrows, ref doubleMxN AVrows, int rows)
        {
            var rin = new doubleN(Cols, Allocator.Temp, false);
            var rout = new doubleN(Rows, Allocator.Temp, false);
            for (int i = 0; i < rows; i++)
            {
                for (int c = 0; c < Cols; c++) rin[c] = Vrows[i, c];
                Apply(in rin, ref rout);
                for (int c = 0; c < Rows; c++) AVrows[i, c] = rout[c];
            }
            rout.Dispose();
            rin.Dispose();
        }
    }

    public static partial class BSR
    {
        /// <summary>
        /// Row-wise squared, column-weighted reduction: out[i] = Σ_j A[i,j]² · w[j], computed directly
        /// from the stored blocks in one O(nnz) pass (no AᵀA, no transpose-matvecs). The row dual of
        /// <see cref="columnNormsSquared"/> (which gives Σ_i A[i,j]²). Feeds the interior-point normal
        /// Jacobi diagonal diag(A diag(w) Aᵀ). Written into the caller's out (length A.M_Rows), no alloc.
        /// NOT supported for Symmetric (upper-block-only) storage -- the implicit lower blocks would be
        /// under-counted.
        /// </summary>
        public static void rowSquaredWeighted(in doubleBSR A, in doubleN w, ref doubleN outv)
        {
            if (w.N != A.N_Cols)
                throw new ArgumentException("rowSquaredWeighted: w.N must equal A.N_Cols");
            if (outv.N != A.M_Rows)
                throw new ArgumentException("rowSquaredWeighted: outv.N must equal A.M_Rows");
            if (A.Symmetric)
                throw new ArgumentException("rowSquaredWeighted: not supported for Symmetric (upper-block-only) storage");

            int BR = A.BR, BC = A.BC, blockSize = BR * BC;

            unsafe
            {
                int* rowPtr = A.RowPtr.Ptr;
                int* colInd = A.ColInd.Ptr;
                double* values = A.Values.Ptr;
                double* wPtr = w.Data.Ptr;
                double* oPtr = outv.Data.Ptr;

                UnsafeUtility.MemClear(oPtr, (long)outv.Data.Length * UnsafeUtility.SizeOf<double>());

                for (int bi = 0; bi < A.BlockRows; bi++)
                {
                    int rowBase = bi * BR;
                    for (int k = rowPtr[bi]; k < rowPtr[bi + 1]; k++)
                    {
                        int colBase = colInd[k] * BC;
                        double* block = values + (long)k * blockSize;
                        for (int r = 0; r < BR; r++)
                            for (int c = 0; c < BC; c++)
                            {
                                double v = block[r * BC + c];
                                oPtr[rowBase + r] += v * v * wPtr[colBase + c];
                            }
                    }
                }
            }
        }
    }
}
