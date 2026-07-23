using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra;
using LinearAlgebra.Internal;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Block-Jacobi preconditioner for a square BSR (<c>BlockRows==BlockCols</c>, <c>BR==BC</c>):
    /// z = M⁻¹ r where M = blockdiag(A_00, A_11, ..., A_{nb-1,nb-1}), i.e. each diagonal block
    /// inverted independently and applied block-wise. The <c>BR==1</c> case degenerates to
    /// point-Jacobi (z_i = r_i / A_ii) automatically.
    ///
    /// Built ONCE from a compressed <see cref="fProxyBSR"/> via LU decomposition on each diagonal
    /// block (<see cref="LU.decompInPlace(ref fProxyMxN, ref Pivot)"/> /
    /// <see cref="LU.decompSolve(ref fProxyMxN, in Pivot, ref fProxyN)"/>); <see cref="Apply"/> is
    /// then a zero-alloc block-diagonal matvec every PCG iteration. Readonly: nothing mutates
    /// after the constructor fills DInv.
    /// </summary>
    public readonly partial struct fProxyBlockJacobi : IfProxyPreconditioner, IDisposable
    {
        public readonly int BlockRows;  // nb: number of diagonal blocks (== BlockCols of the source BSR)
        public readonly int BR;         // block dimension (== BC of the source BSR)

        public int Rows => BlockRows * BR;

        private readonly UnsafeList<fProxy> _dInv;

        /// <summary>Inverted diagonal blocks, flat row-major per block: DInv[i*BR*BR + r*BR + c]
        /// holds (A_ii⁻¹)[r,c]. Length nb*BR*BR.</summary>
        public unsafe UnsafeList<fProxy> DInv => _dInv;

        /// <summary>
        /// Builds the preconditioner from A's diagonal blocks. A must be square
        /// (BlockRows==BlockCols, BR==BC). Throws ArgumentException if a diagonal block is
        /// missing from the stored pattern or is singular. Use the out-info overload to receive
        /// the singular-block outcome as a <see cref="PreconditionerInfo"/> instead of an exception.
        /// </summary>
        public unsafe fProxyBlockJacobi(in fProxyBSR A, Allocator allocator)
        {
            this = new fProxyBlockJacobi(in A, allocator, out PreconditionerInfo info);
            if (!info.Solved)
                throw new ArgumentException("fProxyBlockJacobi: diagonal block is singular");
        }

        /// <summary>
        /// Non-throwing build: info carries the per-block LU's own status — Success, or the
        /// failing block's status (Singular) when a diagonal block cannot be inverted (the
        /// preconditioner is then unusable — do not Apply). shift is always 0 and attempts 1
        /// (BlockJacobi has no shift retry). Caller-contract violations (non-square, missing
        /// diagonal block) still throw.
        /// </summary>
        public unsafe fProxyBlockJacobi(in fProxyBSR A, Allocator allocator, out PreconditionerInfo info)
        {
            _dInv = default;

            if (A.BlockRows != A.BlockCols || A.BR != A.BC)
                throw new ArgumentException("fProxyBlockJacobi: A must be square (BlockRows==BlockCols, BR==BC)");

            BlockRows = A.BlockRows;
            BR = A.BR;

            int blockLen = BR * BR;
            var dinv = new UnsafeList<fProxy>(BlockRows * blockLen, allocator, NativeArrayOptions.ClearMemory);
            dinv.Resize(BlockRows * blockLen, NativeArrayOptions.ClearMemory);

            // Stack scratch for the BR <= 16 fast path (Gauss-Jordan needs the working block +
            // the growing inverse). stackalloc'd ONCE, never inside the loop; 2 x 2 KB at the
            // BR = 16 double worst case.
            fProxy* Mwork = stackalloc fProxy[16 * 16];
            fProxy* Inv = stackalloc fProxy[16 * 16];

            for (int i = 0; i < BlockRows; i++)
            {
                // Blocks within a block-row are stored in ascending ColInd (BSR invariant) --
                // scan forward and stop as soon as we pass column i.
                int s = A.RowPtr[i], e = A.RowPtr[i + 1];
                int found = -1;
                for (int k = s; k < e; k++)
                {
                    int blockCol = A.ColInd[k];
                    if (blockCol == i) { found = k; break; }
                    if (blockCol > i) break;
                }

                if (found < 0)
                {
                    dinv.Dispose();
                    throw new ArgumentException("fProxyBlockJacobi: missing diagonal block in A");
                }

                int srcOff = found * blockLen;
                int dstOff = i * blockLen;

                if (BR <= 16)
                {
                    // Fast path: zero-alloc Gauss-Jordan inversion on the stack scratch.
                    for (int t = 0; t < blockLen; t++)
                        Mwork[t] = A.Values[srcOff + t];

                    if (!InvertBlock(Mwork, Inv, BR))
                    {
                        dinv.Dispose();
                        info = new PreconditionerInfo { status = DirectSolveStatus.Singular, shift = 0, attempts = 1 };
                        return;
                    }

                    for (int t = 0; t < blockLen; t++)
                        dinv[dstOff + t] = Inv[t];
                    continue;
                }

                // General path (BR > 16): LU on Temp scratch, unit-vector column solves.
                var Dcopy = new fProxyMxN(BR, BR, Allocator.Temp, true);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Dcopy[r, c] = A.Values[srcOff + r * BR + c];

                var P = new Pivot(BR, Allocator.Temp);
                var luInfo = LU.decompInPlace(ref Dcopy, ref P);

                if (!luInfo)
                {
                    P.Dispose();
                    Dcopy.Dispose();
                    dinv.Dispose();
                    // propagate the failing block's LU status
                    info = new PreconditionerInfo { status = luInfo.status, shift = 0, attempts = 1 };
                    return;
                }

                // Column-by-column solve against unit vectors -> the explicit BR x BR inverse.
                var col = new fProxyN(BR, Allocator.Temp, true);
                for (int c = 0; c < BR; c++)
                {
                    for (int r = 0; r < BR; r++)
                        col[r] = (r == c) ? (fProxy)1 : (fProxy)0;

                    LU.decompSolve(ref Dcopy, in P, ref col);

                    for (int r = 0; r < BR; r++)
                        dinv[dstOff + r * BR + c] = col[r];
                }

                col.Dispose();
                P.Dispose();
                Dcopy.Dispose();
            }

            _dInv = dinv;
            info = new PreconditionerInfo { status = DirectSolveStatus.Success, shift = 0, attempts = 1 };
        }

        // In-place Gauss-Jordan inverse with partial pivoting: M (n x n row-major, DESTROYED) ->
        // Inv. Returns false when a pivot falls at or below a diagonal-scaled floor (16*eps*diagMax,
        // matching fProxyILU0) so a denormal pivot reports singular instead of inverting to Inf.
        // NaN-safe, the library's usual pivot idiom. M and Inv are distinct stack buffers.
        static unsafe bool InvertBlock(fProxy* M, fProxy* Inv, int n)
        {
            fProxy diagMax = 0;
            for (int r = 0; r < n; r++)
            {
                fProxy av = math.abs(M[r * n + r]);
                if (av > diagMax) diagMax = av;
            }
            if (diagMax <= (fProxy)0) diagMax = (fProxy)1;
            fProxy pivotFloor = (fProxy)16 * Consts.fProxyEpsilon * diagMax;

            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Inv[r * n + c] = (r == c) ? (fProxy)1 : (fProxy)0;

            for (int c = 0; c < n; c++)
            {
                int piv = c;
                fProxy best = math.abs(M[c * n + c]);
                for (int r = c + 1; r < n; r++)
                {
                    fProxy av = math.abs(M[r * n + c]);
                    if (av > best) { best = av; piv = r; }
                }
                if (!(best > pivotFloor))
                    return false;

                if (piv != c)
                {
                    for (int t = 0; t < n; t++)
                    {
                        fProxy tm = M[piv * n + t]; M[piv * n + t] = M[c * n + t]; M[c * n + t] = tm;
                        fProxy ti = Inv[piv * n + t]; Inv[piv * n + t] = Inv[c * n + t]; Inv[c * n + t] = ti;
                    }
                }

                fProxy invD = (fProxy)1 / M[c * n + c];
                for (int t = 0; t < n; t++)
                {
                    M[c * n + t] *= invD;
                    Inv[c * n + t] *= invD;
                }

                for (int r = 0; r < n; r++)
                {
                    if (r == c) continue;
                    fProxy f = M[r * n + c];
                    if (f == (fProxy)0) continue;
                    for (int t = 0; t < n; t++)
                    {
                        M[r * n + t] -= f * M[c * n + t];
                        Inv[r * n + t] -= f * Inv[c * n + t];
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// z = M⁻¹ r, applied block-wise: z_i = A_ii⁻¹ · r_i. z must not alias r (each z_i read
        /// draws on the full r_i block; overwriting r in place mid-block would corrupt later
        /// rows of the same block's product). For BR in {1,2,3,4,6}, dispatches to a fully
        /// unrolled dense b x b matvec (<see cref="UnsafeOP.blockJacobiApplyB1"/>..B6), bit-identical
        /// to the general loop below. Any other BR falls through to the general runtime-BR loop.
        /// </summary>
        public bool IsIdentity => false;
        public bool IsSpd => true;
        public bool IsConstant => true;

        public unsafe void Apply(in fProxyN r, ref fProxyN z)
        {
            int n = Rows;

            if (r.N != n)
                throw new ArgumentException("fProxyBlockJacobi.Apply: r.N must equal Rows");
            if (z.N != n)
                throw new ArgumentException("fProxyBlockJacobi.Apply: z.N must equal Rows");

            if (z.Data.Ptr == r.Data.Ptr)
                throw new ArgumentException("fProxyBlockJacobi.Apply: z must not alias r");

            fProxy* rp = r.Data.Ptr;
            fProxy* zp = z.Data.Ptr;
            fProxy* dp = DInv.Ptr;

            switch (BR)
            {
                case 1: UnsafeOP.blockJacobiApplyB1(dp, rp, zp, BlockRows); return;
                case 2: UnsafeOP.blockJacobiApplyB2(dp, rp, zp, BlockRows); return;
                case 3: UnsafeOP.blockJacobiApplyB3(dp, rp, zp, BlockRows); return;
                case 4: UnsafeOP.blockJacobiApplyB4(dp, rp, zp, BlockRows); return;
                case 6: UnsafeOP.blockJacobiApplyB6(dp, rp, zp, BlockRows); return;
            }

            int blockLen = BR * BR;

            for (int i = 0; i < BlockRows; i++)
            {
                int rowBase = i * BR;
                int blockOff = i * blockLen;

                for (int lr = 0; lr < BR; lr++)
                {
                    fProxy sum = 0;
                    for (int lc = 0; lc < BR; lc++)
                        sum += dp[blockOff + lr * BR + lc] * rp[rowBase + lc];
                    zp[rowBase + lr] = sum;
                }
            }
        }

        /// <summary>Disposes the DInv buffer.</summary>
        public unsafe void Dispose()
        {
            _dInv.Dispose();
        }
    }
}
