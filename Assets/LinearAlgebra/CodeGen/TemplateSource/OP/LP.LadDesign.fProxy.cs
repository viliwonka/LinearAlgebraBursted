using System;

using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

using LinearAlgebra.Internal;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    /// <summary>
    /// The design matrix A of an L1 / quantile regression, behind the three operations the
    /// Frisch-Newton core needs: A x, Aᵀ y, and the weighted Gram Aᵀ diag(q) A. Solvers are generic
    /// over <c>TDesign : struct, IfProxyLadDesign</c>, so each call compiles to a direct call and one
    /// core body serves both dense and block-sparse designs.
    ///
    /// Deliberately NOT <see cref="IfProxyLinearOperator"/>: that interface also requires ApplyDot and
    /// ApplyBlock, which a regression design has no use for, and the weighted Gram it does need has no
    /// place there. Same rationale as
    /// <see cref="Sparse.IfProxyStandardFormOperator"/> being its own interface.
    /// </summary>
    public interface IfProxyLadDesign
    {
        /// <summary>m -- observations.</summary>
        int Rows { get; }
        /// <summary>n -- coefficients. The weighted Gram is n x n and DENSE, so n drives cost.</summary>
        int Cols { get; }

        /// <summary>y = A x. x has length Cols, y length Rows. y must not alias x.</summary>
        void Apply(in fProxyN x, ref fProxyN y);

        /// <summary>x = Aᵀ y. y has length Rows, x length Cols. x must not alias y.</summary>
        void ApplyT(in fProxyN y, ref fProxyN x);

        /// <summary>
        /// M (Cols x Cols) = Aᵀ diag(q) A, fully written (both triangles), q of length Rows. M is
        /// overwritten, not accumulated. No regularization is applied -- the caller adds it, so it can
        /// be made relative to M's own scale.
        /// </summary>
        void WeightedGram(in fProxyN q, ref fProxyMxN M);
    }

    /// <summary>Dense <see cref="fProxyMxN"/> design.</summary>
    public readonly struct fProxyDenseLadDesign : IfProxyLadDesign
    {
        readonly fProxyMxN A;

        public fProxyDenseLadDesign(in fProxyMxN a) { A = a; }

        public int Rows => A.M_Rows;
        public int Cols => A.N_Cols;

        public unsafe void Apply(in fProxyN x, ref fProxyN y)
        {
            int m = A.M_Rows, n = A.N_Cols;
            fProxy* Ap = A.Data.Ptr; fProxy* xp = x.Data.Ptr; fProxy* yp = y.Data.Ptr;
            UnsafeUtility.MemClear(yp, (long)m * UnsafeUtility.SizeOf<fProxy>());
            for (int i = 0; i < m; i++) yp[i] = UnsafeOP.vecDot(Ap + (long)i * n, xp, n);
        }

        public unsafe void ApplyT(in fProxyN y, ref fProxyN x)
        {
            int m = A.M_Rows, n = A.N_Cols;
            fProxy* Ap = A.Data.Ptr; fProxy* yp = y.Data.Ptr; fProxy* xp = x.Data.Ptr;
            UnsafeUtility.MemClear(xp, (long)n * UnsafeUtility.SizeOf<fProxy>());
            for (int i = 0; i < m; i++)
            {
                fProxy v = yp[i];
                if (v == (fProxy)0) continue;
                UnsafeOP.axpy(xp, Ap + (long)i * n, v, n);
            }
        }

        // One cache-friendly pass over A's ROWS: row i contributes q_i · A[i,:] ⊗ A[i,:] to the upper
        // triangle, then mirrored. Row-major storage makes A[i,:] unit-stride, which a column-contracted
        // order would not be; the inner sweep is an AXPY routed through UnsafeOP.axpy.
        public unsafe void WeightedGram(in fProxyN q, ref fProxyMxN M)
        {
            int m = A.M_Rows, n = A.N_Cols;
            for (int r = 0; r < n; r++)
                for (int c = r; c < n; c++)
                    M[r, c] = (fProxy)0;

            fProxy* Ap = A.Data.Ptr;
            fProxy* Mp = M.Data.Ptr;
            for (int i = 0; i < m; i++)
            {
                fProxy qi = q[i];
                fProxy* Arow = Ap + (long)i * n;
                for (int r = 0; r < n; r++)
                {
                    fProxy v = qi * Arow[r];
                    if (v == (fProxy)0) continue;
                    UnsafeOP.axpy(Mp + (long)r * n + r, Arow + r, v, n - r);
                }
            }

            for (int r = 0; r < n; r++)
                for (int c = r + 1; c < n; c++)
                    M[c, r] = M[r, c];
        }
    }

    /// <summary>
    /// Block-sparse <see cref="fProxyBSR"/> design. A x and Aᵀ y go through the library's block spMV /
    /// spMVT; the weighted Gram streams the block rows directly. Cost is O(sum of row_nnz²) per build,
    /// so it scales with the SPARSITY of A and with n, never with m² -- the reason a tall sparse design
    /// is cheap here.
    /// </summary>
    public readonly struct fProxyBsrLadDesign : IfProxyLadDesign
    {
        readonly fProxyBSR A;

        public fProxyBsrLadDesign(in fProxyBSR a) { A = a; }

        public int Rows => A.M_Rows;
        public int Cols => A.N_Cols;

        public void Apply(in fProxyN x, ref fProxyN y) => BSR.spMV(in A, in x, ref y);
        public void ApplyT(in fProxyN y, ref fProxyN x) => BSR.spMVT(in A, in y, ref x);

        // For each scalar row i, its nonzeros are spread across every block in i's block row, so the
        // outer product is formed by walking that block row twice. Pairs are filtered by j2 >= j1
        // rather than by starting the inner walk at the outer block: ColInd order within a block row is
        // not part of the BSR contract, and skipping ahead would drop entries if it were unsorted.
        public unsafe void WeightedGram(in fProxyN q, ref fProxyMxN M)
        {
            int n = A.N_Cols;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    M[r, c] = (fProxy)0;

            int* rowPtr = A.RowPtr.Ptr;
            int* colInd = A.ColInd.Ptr;
            fProxy* vals = A.Values.Ptr;
            int BR = A.BR, BC = A.BC, blockSize = BR * BC;

            for (int br = 0; br < A.BlockRows; br++)
            {
                int k0 = rowPtr[br], k1 = rowPtr[br + 1];
                if (k0 == k1) continue;

                for (int r = 0; r < BR; r++)
                {
                    fProxy qi = q[br * BR + r];
                    if (qi == (fProxy)0) continue;

                    for (int ka = k0; ka < k1; ka++)
                    {
                        fProxy* rowA = vals + (long)ka * blockSize + (long)r * BC;
                        int jaBase = colInd[ka] * BC;
                        for (int ca = 0; ca < BC; ca++)
                        {
                            fProxy v = qi * rowA[ca];
                            if (v == (fProxy)0) continue;
                            int j1 = jaBase + ca;

                            for (int kb = k0; kb < k1; kb++)
                            {
                                fProxy* rowB = vals + (long)kb * blockSize + (long)r * BC;
                                int jbBase = colInd[kb] * BC;
                                for (int cb = 0; cb < BC; cb++)
                                {
                                    int j2 = jbBase + cb;
                                    if (j2 < j1) continue;
                                    M[j1, j2] += v * rowB[cb];
                                }
                            }
                        }
                    }
                }
            }

            for (int r = 0; r < n; r++)
                for (int c = r + 1; c < n; c++)
                    M[c, r] = M[r, c];
        }
    }
}
