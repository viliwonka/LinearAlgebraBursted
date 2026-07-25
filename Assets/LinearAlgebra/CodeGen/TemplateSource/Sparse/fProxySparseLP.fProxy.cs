using System;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

using LinearAlgebra;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    /// <summary>
    /// A standard-form LP constraint operator Aₛ (the interior point works on min cᵀz s.t. Aₛ z = b,
    /// z ≥ 0) that, on top of the usual <see cref="IfProxyLinearOperator"/> Apply/ApplyT, can report the
    /// diagonal of its interior-point normal matrix diag(Aₛ diag(D) Aₛᵀ) matrix-free -- the hook a
    /// diagonally-preconditioned normal-equations solve needs beyond a plain linear operator.
    /// <see cref="Sparse.fProxyLadOperator"/> (Aₛ = [A | -A | -I | I]) is the standard-form encoding of
    /// an L1 regression. NOTE no solver in the library currently consumes this: LP.lad over a sparse
    /// design goes to LP.ladFN, which works on the ORIGINAL m x n matrix instead of reformulating.
    /// </summary>
    public interface IfProxyStandardFormOperator : IfProxyLinearOperator
    {
        /// <summary>diag(Aₛ diag(D) Aₛᵀ) written into <paramref name="diag"/> (length Aₛ.Rows), computed
        /// matrix-free from the sparse entries. D has length Aₛ.Cols.</summary>
        void NormalDiagonal(in fProxyN D, ref fProxyN diag);
    }

    /// <summary>
    /// Symmetric normal-equations operator M = Aₛ · diag(D) · Aₛᵀ, presented matrix-free over any inner
    /// <typeparamref name="TInner"/> operator Aₛ -- never forms M. This is the SPD system an interior-
    /// point LP solves each iteration (D = Z S⁻¹, the primal/dual diagonal, changes every iteration), so
    /// it lets <see cref="Krylov.cg{TOp,TPre}"/> solve the normal equations directly. Also a general
    /// "AᵀDA / A D Aᵀ" operator usable beyond LP.
    ///
    /// <c>Rows == Cols == Aₛ.Rows</c> (M is Aₛ.Rows square). Apply(v) = Aₛ (D ∘ (Aₛᵀ v)). Symmetric, so
    /// <c>ApplyT == Apply</c>. Holds the inner operator, the diagonal <c>D</c> (length Aₛ.Cols -- a
    /// handle the caller rewrites in place each iteration), and one owned <c>Scratch</c> of length
    /// Aₛ.Cols. Readonly, same shape as <see cref="fProxyColScaledOperator{TInner}"/>.
    /// </summary>
    public readonly struct fProxyNormalOperator<TInner> : IfProxyLinearOperator
        where TInner : struct, IfProxyLinearOperator
    {
        public readonly TInner As;
        public readonly fProxyN D;        // length As.Cols; the (mutable) diagonal
        public readonly fProxyN Scratch;  // length As.Cols
        public readonly fProxy Reg;       // primal-dual regularization: M := Aₛ D Aₛᵀ + Reg·I

        public fProxyNormalOperator(in TInner aS, in fProxyN d, in fProxyN scratch, fProxy reg)
        {
            if (d.N != aS.Cols)
                throw new ArgumentException("fProxyNormalOperator: D.N must equal As.Cols");
            if (scratch.N != aS.Cols)
                throw new ArgumentException("fProxyNormalOperator: scratch.N must equal As.Cols");
            As = aS; D = d; Scratch = scratch; Reg = reg;
        }

        public int Rows => As.Rows;
        public int Cols => As.Rows;   // M is m×m

        public void Apply(in fProxyN v, ref fProxyN y)
        {
            var w = Scratch;                          // handle copy -> writable through the interface
            As.ApplyT(in v, ref w);                   // w = Aₛᵀ v            (length Cols(Aₛ))
            for (int j = 0; j < w.N; j++) w[j] *= D[j];
            As.Apply(in w, ref y);                    // y = Aₛ (D ∘ Aₛᵀ v)   (length Rows(Aₛ))
            if (Reg != (fProxy)0) for (int i = 0; i < y.N; i++) y[i] += Reg * v[i];   // + Reg·I
        }

        public void ApplyT(in fProxyN v, ref fProxyN y) => Apply(in v, ref y);   // symmetric

        // Composes Apply + dot; no fused kernel here.
        public fProxy ApplyDot(in fProxyN v, ref fProxyN y)
        {
            Apply(in v, ref y);
            return Blas.dot(v, y);
        }

        // Per-row apply (M is symmetric m×m); never the hot path for LP -- present only to satisfy the
        // interface. Two bounded Temp scratch vectors per call.
        public void ApplyBlock(in fProxyMxN Vrows, ref fProxyMxN AVrows, int rows)
        {
            int cols = Vrows.N_Cols;
            var rin = new fProxyN(cols, Allocator.Temp, false);
            var rout = new fProxyN(cols, Allocator.Temp, false);
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
    public readonly struct fProxyLadOperator : IfProxyStandardFormOperator
    {
        public readonly fProxyBSR A;
        public readonly int M, N;
        public readonly fProxyN Sp;    // length N
        public readonly fProxyN Tm;    // length M
        public readonly fProxyN Atr;   // length N

        public fProxyLadOperator(in fProxyBSR a, in fProxyN sp, in fProxyN tm, in fProxyN atr)
        {
            A = a; M = a.M_Rows; N = a.N_Cols;
            if (sp.N != N) throw new ArgumentException("fProxyLadOperator: sp.N must equal A.N_Cols");
            if (tm.N != M) throw new ArgumentException("fProxyLadOperator: tm.N must equal A.M_Rows");
            if (atr.N != N) throw new ArgumentException("fProxyLadOperator: atr.N must equal A.N_Cols");
            Sp = sp; Tm = tm; Atr = atr;
        }

        public int Rows => M;
        public int Cols => 2 * N + 2 * M;

        public void Apply(in fProxyN z, ref fProxyN y)
        {
            var sp = Sp; var tm = Tm;
            for (int j = 0; j < N; j++) sp[j] = z[j];              // x⁺
            BSR.spMV(in A, in sp, ref y);                          // y = A x⁺
            for (int j = 0; j < N; j++) sp[j] = z[N + j];          // x⁻
            BSR.spMV(in A, in sp, ref tm);                         // tm = A x⁻
            for (int i = 0; i < M; i++) y[i] = y[i] - tm[i] - z[2 * N + i] + z[2 * N + M + i];
        }

        public void ApplyT(in fProxyN r, ref fProxyN outv)
        {
            var atr = Atr;
            BSR.spMVT(in A, in r, ref atr);                        // atr = Aᵀ r   (length N)
            for (int j = 0; j < N; j++) { outv[j] = atr[j]; outv[N + j] = -atr[j]; }
            for (int i = 0; i < M; i++) { outv[2 * N + i] = -r[i]; outv[2 * N + M + i] = r[i]; }
        }

        // Rectangular (Rows = M, Cols = 2N+2M): satisfies the interface, but no solver calls
        // ApplyDot on this operator directly (only fProxyNormalOperator wraps it for PCG, and
        // that wrapper has its own ApplyDot). Composes: Apply, then a plain dot pass.
        public fProxy ApplyDot(in fProxyN z, ref fProxyN y)
        {
            Apply(in z, ref y);
            return Blas.dot(z, y);
        }

        // diag(Aₛ diag(D) Aₛᵀ)_i, with D = [d⁺(n) | d⁻(n) | dᵤ(m) | dᵥ(m)]:
        //   Σ_j A[i,j]²·(d⁺[j] + d⁻[j])  +  dᵤ[i]  +  dᵥ[i]
        // (the −A block contributes (−A[i,j])² = A[i,j]², the ±I blocks each contribute 1²).
        public void NormalDiagonal(in fProxyN D, ref fProxyN diag)
        {
            var sp = Sp;
            for (int j = 0; j < N; j++) sp[j] = D[j] + D[N + j];   // w = d⁺ + d⁻
            BSR.rowSquaredWeighted(in A, in sp, ref diag);        // diag[i] = Σ_j A[i,j]² w[j]
            for (int i = 0; i < M; i++) diag[i] += D[2 * N + i] + D[2 * N + M + i];
        }

        public void ApplyBlock(in fProxyMxN Vrows, ref fProxyMxN AVrows, int rows)
        {
            var rin = new fProxyN(Cols, Allocator.Temp, false);
            var rout = new fProxyN(Rows, Allocator.Temp, false);
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
        /// NOT supported for Symmetric (lower-block-only) storage -- the implicit upper blocks would be
        /// under-counted.
        /// </summary>
        public static void rowSquaredWeighted(in fProxyBSR A, in fProxyN w, ref fProxyN outv)
        {
            if (w.N != A.N_Cols)
                throw new ArgumentException("rowSquaredWeighted: w.N must equal A.N_Cols");
            if (outv.N != A.M_Rows)
                throw new ArgumentException("rowSquaredWeighted: outv.N must equal A.M_Rows");
            if (A.Symmetric)
                throw new ArgumentException("rowSquaredWeighted: not supported for Symmetric (lower-block-only) storage");

            int BR = A.BR, BC = A.BC, blockSize = BR * BC;

            unsafe
            {
                int* rowPtr = A.RowPtr.Ptr;
                int* colInd = A.ColInd.Ptr;
                fProxy* values = A.Values.Ptr;
                fProxy* wPtr = w.Data.Ptr;
                fProxy* oPtr = outv.Data.Ptr;

                UnsafeUtility.MemClear(oPtr, (long)outv.Data.Length * UnsafeUtility.SizeOf<fProxy>());

                for (int bi = 0; bi < A.BlockRows; bi++)
                {
                    int rowBase = bi * BR;
                    for (int k = rowPtr[bi]; k < rowPtr[bi + 1]; k++)
                    {
                        int colBase = colInd[k] * BC;
                        fProxy* block = values + (long)k * blockSize;
                        for (int r = 0; r < BR; r++)
                            for (int c = 0; c < BC; c++)
                            {
                                fProxy v = block[r * BC + c];
                                oPtr[rowBase + r] += v * v * wPtr[colBase + c];
                            }
                    }
                }
            }
        }
    }
}
