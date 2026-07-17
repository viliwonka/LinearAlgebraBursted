using System;
using Unity.Mathematics;
using Unity.Burst;
using System.Runtime.CompilerServices;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    // Public surface. The vector/flat norms accept either a vector (fProxyN) or a matrix
    // (fProxyMxN, treated as a flat array). Each is a thin, inlined forwarder to the generic
    // fProxyNormsCore body, so the class merges to a bare `Norms` with no prefix while the body
    // stays single-source and Burst emits identical code to the old generic call.
    public static partial class Norms {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy L2(in fProxyN   a) => fProxyNormsCore.L2(a);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy L2(in fProxyMxN a) => fProxyNormsCore.L2(a);

        // Standard L1 norm: the sum of absolute values, Σ|xᵢ| (NOT averaged by length).
        // Naïve accumulation (no Kahan/pairwise compensation): accurate at moderate sizes; very
        // long float vectors may lose precision. The same caveat applies to matrixL1 / matrixLInf.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy L1(in fProxyN   a) => fProxyNormsCore.L1(a);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy L1(in fProxyMxN a) => fProxyNormsCore.L1(a);

        // L-infinity (max-abs) norm: the largest absolute element, max_i |xᵢ|.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy LInf(in fProxyN   a) => fProxyNormsCore.LInf(a);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy LInf(in fProxyMxN a) => fProxyNormsCore.LInf(a);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy L2Range(in fProxyN a, int start, int end)
        {
            if (start >= end)
                throw new ArgumentException("Norms.L2Range: start must be less than end");

            if (start < 0 || end > a.Data.Length)
                throw new ArgumentOutOfRangeException("Norms.L2Range: start and end must be within bounds of vector");

            unsafe
            {
                return math.sqrt(UnsafeOP.vecDotRange(a.Data.Ptr, a.Data.Ptr, start, end));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void normalizeL2(in fProxyN   x) => fProxyNormsCore.NormalizeL2(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void normalizeL2(in fProxyMxN x) => fProxyNormsCore.NormalizeL2(x);

        // returns length before normalization
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeL2(in fProxyN   x, int start, int end) => fProxyNormsCore.NormalizeL2(x, start, end);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeL2(in fProxyMxN x, int start, int end) => fProxyNormsCore.NormalizeL2(x, start, end);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeL1(in fProxyN   x) => fProxyNormsCore.NormalizeL1(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeL1(in fProxyMxN x) => fProxyNormsCore.NormalizeL1(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeL1(in fProxyN   x, int start, int end) => fProxyNormsCore.NormalizeL1(x, start, end);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeL1(in fProxyMxN x, int start, int end) => fProxyNormsCore.NormalizeL1(x, start, end);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeLMax(in fProxyN   x) => fProxyNormsCore.NormalizeLMax(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeLMax(in fProxyMxN x) => fProxyNormsCore.NormalizeLMax(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeLMax(in fProxyN   x, int start, int end) => fProxyNormsCore.NormalizeLMax(x, start, end);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeLMax(in fProxyMxN x, int start, int end) => fProxyNormsCore.NormalizeLMax(x, start, end);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeLP(in fProxyN   x, fProxy p) => fProxyNormsCore.NormalizeLP(x, p);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeLP(in fProxyMxN x, fProxy p) => fProxyNormsCore.NormalizeLP(x, p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeLP(in fProxyN   x, int start, int end, fProxy p) => fProxyNormsCore.NormalizeLP(x, start, end, p);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy normalizeLP(in fProxyMxN x, int start, int end, fProxy p) => fProxyNormsCore.NormalizeLP(x, start, end, p);

        // ---- Enum-dispatch normalize ----

        /// <summary>Normalize x to unit norm in-place, using the specified <paramref name="n"/> (L1/L2/Linf).
        /// Delegates to the corresponding <c>normalizeL1</c>/<c>normalizeL2</c>/<c>normalizeLMax</c> kernel.</summary>
        /// <remarks>Flat form — treats the input as one 1-D array. For a matrix this is the
        /// <b>whole-matrix</b> scope (all elements as a single distribution); use
        /// <see cref="normalizeRows"/> or <see cref="normalizeColumns"/> for per-axis normalization.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void normalize(in fProxyN   x, Norm n) => fProxyNormsCore.Normalize(x, n);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void normalize(in fProxyMxN x, Norm n) => fProxyNormsCore.Normalize(x, n);

        // Zero-norm row → left at 0 (NaN-safe !(norm > 0) guard). No allocation.
        public static void normalizeRows(ref fProxyMxN A, Norm n)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;

            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    fProxy rowNorm;
                    switch (n)
                    {
                        case Norm.L1:   rowNorm = UnsafeOP.sumAbs(row, nc); break;
                        case Norm.L2:   rowNorm = math.sqrt(UnsafeOP.vecDot(row, row, nc)); break;
                        default:        rowNorm = UnsafeOP.maxAbs(row, nc); break; // Linf
                    }

                    if (!(rowNorm > 0f)) continue; // zero-norm row → leave unchanged

                    fProxy inv = (fProxy)1f / rowNorm;
                    for (int c = 0; c < nc; c++) row[c] *= inv;
                }
            }
        }

        // Zero-norm column → left at 0 (NaN-safe !(norm > 0) guard). No allocation.
        public static void normalizeColumns(ref fProxyMxN A, Norm n)
        {
            int nr = A.M_Rows, nc = A.N_Cols;
            if (nr == 0 || nc == 0) return;

            // Per-column norms in one row-major pass (unit-stride inner loop vectorises; each column
            // still accumulates its rows in ascending order → bit-identical to the strided per-column
            // loops), then a branch-free row-major scale. One length-N_Cols Temp holds norms then inv.
            fProxyN inv = A.fProxyTempVec(nc);
            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                fProxy* ip = inv.Data.Ptr;
                for (int c = 0; c < nc; c++) ip[c] = 0f;

                switch (n)
                {
                    case Norm.L1:
                        for (int r = 0; r < nr; r++) { fProxy* row = ap + (long)r * nc; for (int c = 0; c < nc; c++) ip[c] += math.abs(row[c]); }
                        break;
                    case Norm.L2:
                        for (int r = 0; r < nr; r++) { fProxy* row = ap + (long)r * nc; for (int c = 0; c < nc; c++) ip[c] += row[c] * row[c]; }
                        for (int c = 0; c < nc; c++) ip[c] = math.sqrt(ip[c]);
                        break;
                    default: // Linf
                        for (int r = 0; r < nr; r++) { fProxy* row = ap + (long)r * nc; for (int c = 0; c < nc; c++) ip[c] = math.max(ip[c], math.abs(row[c])); }
                        break;
                }

                // norm → reciprocal; zero-norm (or NaN) columns get factor 1 so `*= 1` leaves them
                // bit-identically unchanged (x*1 == x for every value, matching the old skip).
                for (int c = 0; c < nc; c++) ip[c] = (ip[c] > 0f) ? (fProxy)1f / ip[c] : (fProxy)1f;

                for (int r = 0; r < nr; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++) row[c] *= ip[c];
                }
            }
        }

        // ---- Induced (operator) matrix norms ----

        // Induced 1-norm ‖A‖₁: the maximum absolute column sum, max_j Σ_i |A[i,j]|. Allocation-free.
        public static fProxy matrixL1(in fProxyMxN A)
        {
            int nr = A.M_Rows, nc = A.N_Cols;
            if (nr == 0 || nc == 0)
                return (fProxy)0;

            // Max absolute column sum via a row-major per-column accumulate: the inner loop is
            // unit-stride (vectorises) and each column still sums its rows in ascending order, so the
            // result is bit-identical to the strided form. One length-N_Cols Temp accumulator.
            fProxyN acc = A.fProxyTempVec(nc);
            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                fProxy* cp = acc.Data.Ptr;
                for (int j = 0; j < nc; j++) cp[j] = (fProxy)0;
                for (int i = 0; i < nr; i++)
                {
                    fProxy* row = ap + (long)i * nc;
                    for (int j = 0; j < nc; j++) cp[j] += math.abs(row[j]);
                }
            }

            fProxy best = (fProxy)0;
            for (int j = 0; j < nc; j++)
                if (acc[j] > best) best = acc[j];
            return best;
        }

        // Induced ∞-norm ‖A‖∞: the maximum absolute row sum, max_i Σ_j |A[i,j]|. Allocation-free.
        public static fProxy matrixLInf(in fProxyMxN A)
        {
            fProxy best = (fProxy)0;
            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                int nc = A.N_Cols;
                for (int i = 0; i < A.M_Rows; i++)
                {
                    fProxy rowSum = UnsafeOP.sumAbs(ap + (long)i * nc, nc);
                    if (rowSum > best)
                        best = rowSum;
                }
            }
            return best;
        }

        // Induced 2-norm (spectral norm) ‖A‖₂ = σ_max(A), the largest singular value. Runs a
        // values-only SVD on a copy (A is not modified); allocates SVD scratch from A's arena.
        // Returns NaN when the SVD fails to converge.
        public static fProxy matrixL2(in fProxyMxN A)
        {
            int k = math.min(A.M_Rows, A.N_Cols);
            if (k == 0)
                return (fProxy)0;

            fProxyN S = A.fProxyTempVec(k);
            if (!SVD.singularValues(in A, ref S))
                return fProxy.NaN;   // bidiagonal QR did not converge; S is unwritten
            return S[0];   // singular values are sorted descending -> σ_max
        }
    }

    // Internal generic bodies (one per norm kernel), written once. `floatNormsCore` and
    // `doubleNormsCore` are distinct types, so the type-identical generic signatures never
    // collide (unlike a merged `Norms`); the public forwarders above inline straight through.
    internal static class fProxyNormsCore
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy L2<T>(in T a) where T : unmanaged, IUnsafefProxyArray {
            unsafe { return math.sqrt(UnsafeOP.vecDot(a.Data.Ptr, a.Data.Ptr, a.Data.Length)); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy L1<T>(in T a) where T : unmanaged, IUnsafefProxyArray {
            unsafe { return UnsafeOP.sumAbs(a.Data.Ptr, a.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy LInf<T>(in T a) where T : unmanaged, IUnsafefProxyArray {
            unsafe { return UnsafeOP.maxAbs(a.Data.Ptr, a.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void NormalizeL2<T>(in T x) where T : unmanaged, IUnsafefProxyArray
        {
            unsafe { UnsafeOP.normalizeL2InPlace(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeL2<T>(in T x, int start, int end) where T : unmanaged, IUnsafefProxyArray
        {
            if (start >= end)
                throw new ArgumentException("NormalizeL2: start must be less than end");

            if (start < 0 || end > x.Data.Length)
                throw new ArgumentOutOfRangeException("NormalizeL2: start and end must be within bounds of vector");

            unsafe { return UnsafeOP.normalizeL2InPlace(x.Data.Ptr, start, end); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeL1<T>(in T x) where T : unmanaged, IUnsafefProxyArray
        {
            unsafe { return UnsafeOP.normalizeL1(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeL1<T>(in T x, int start, int end) where T : unmanaged, IUnsafefProxyArray
        {
            if (start >= end)
                throw new ArgumentException("NormalizeL1: start must be less than end");

            if (start < 0 || end > x.Data.Length)
                throw new ArgumentOutOfRangeException("NormalizeL1: start and end must be within bounds of vector");

            unsafe { return UnsafeOP.normalizeL1(x.Data.Ptr, start, end); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeLMax<T>(in T x) where T : unmanaged, IUnsafefProxyArray
        {
            unsafe { return UnsafeOP.normalizeLMax(x.Data.Ptr, x.Data.Length); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeLMax<T>(in T x, int start, int end) where T : unmanaged, IUnsafefProxyArray
        {
            if (start >= end)
                throw new ArgumentException("NormalizeLMax: start must be less than end");

            if (start < 0 || end > x.Data.Length)
                throw new ArgumentOutOfRangeException("NormalizeLMax: start and end must be within bounds of vector");

            unsafe { return UnsafeOP.normalizeLMax(x.Data.Ptr, start, end); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeLP<T>(in T x, fProxy p) where T : unmanaged, IUnsafefProxyArray
        {
            unsafe { return UnsafeOP.normalizeLP(x.Data.Ptr, x.Data.Length, p); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy NormalizeLP<T>(in T x, int start, int end, fProxy p) where T : unmanaged, IUnsafefProxyArray
        {
            if (start >= end)
                throw new ArgumentException("NormalizeLP: start must be less than end");

            if (start < 0 || end > x.Data.Length)
                throw new ArgumentOutOfRangeException("NormalizeLP: start and end must be within bounds of vector");

            unsafe { return UnsafeOP.normalizeLP(x.Data.Ptr, start, end, p); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Normalize<T>(in T x, Norm n) where T : unmanaged, IUnsafefProxyArray
        {
            switch (n)
            {
                case Norm.L1:   NormalizeL1(in x);   break;
                case Norm.L2:   NormalizeL2(in x);   break;
                default:        NormalizeLMax(in x);  break; // Linf
            }
        }
    }
}
