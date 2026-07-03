using System.Collections;
using System.Collections.Generic;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.CompilerServices;

using UnityEngine;

namespace LinearAlgebra {

    /// <summary>
    /// Pivot is a more efficient replacement for permutation matrix.
    /// Has a single vector of indices, that can be used to swap vectors elements, or rows/columns of a matrix.
    /// Kind of like "swizzle".
    /// </summary>
    public partial struct Pivot {

        private UnsafeList<int> indices;

        /// <summary>
        /// Tracks the number of effective swaps (calls to Swap where i != j) since construction or Reset.
        /// </summary>
        private int swapCount;

        public int N => indices.Length;

        /// <summary>
        /// Returns +1 if the number of effective swaps is even, -1 if odd.
        /// Reflects the parity of the permutation.
        /// </summary>
        public int Sign => (swapCount & 1) == 0 ? 1 : -1;

        public Pivot(int size, Allocator allocator = Allocator.Temp) {
            indices = new UnsafeList<int>(size, allocator);
            indices.Resize(size);
            swapCount = 0;
            Reset();
        }

        public int this[int i] {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get {
                if (i < 0 || i >= indices.Length)
                    throw new System.ArgumentOutOfRangeException("i", "Pivot index out of range");
                return indices[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Swap(int i, int j) {

            if (i < 0 || i >= indices.Length)
                throw new System.ArgumentOutOfRangeException("i", "Pivot index out of range");
            if (j < 0 || j >= indices.Length)
                throw new System.ArgumentOutOfRangeException("j", "Pivot index out of range");

            if (i == j)
                return;

            swapCount++;

            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        public void Reset() {
            for (int i = 0; i < indices.Length; i++)
                indices[i] = i;
            swapCount = 0;
        }

        public void Dispose() {
            indices.Dispose();
        }

        public Pivot Copy() {

            var copy = new Pivot(indices.Length, Allocator.Temp);

            copy.indices.CopyFrom(indices);
            copy.swapCount = swapCount;

            return copy;
        }

        public Pivot InverseCopy() {

            var copy = new Pivot(indices.Length, Allocator.Temp);

            for (int i = 0; i < indices.Length; i++)
                copy.indices[this[i]] = i;

            // A permutation and its inverse have equal parity
            copy.swapCount = swapCount;

            return copy;
        }

        // Inverse operation inplace
        public void InverseInpl() {

            var tempPivot = new Pivot(indices.Length, Allocator.Temp);

            // copy original into tempPivot
            for (int i = 0; i < indices.Length; i++)
                tempPivot.indices[i] = indices[i];

            for (int i = 0; i < indices.Length; i++)
                indices[tempPivot[i]] = i;

            // A permutation and its inverse have equal parity; swapCount unchanged
            tempPivot.Dispose();
        }

        public void Print() {

            FixedString4096Bytes toPrint = new FixedString4096Bytes();

            for (int i = 0; i < indices.Length; i++)
                toPrint.Append($"{indices[i]}");

            Debug.Log(toPrint);
        }

        /// <summary>
        /// Burst-safe compact summary, e.g. <c>Pivot[N=5, sign=+1]: (2 0 1 4 3)</c>. Caps gracefully
        /// (appends "..." and stops) for very large N rather than overflowing the FixedString.
        /// Never allocates managed memory.
        /// </summary>
        public FixedString4096Bytes ToFixedString() {

            int n = indices.Length;
            FixedString4096Bytes str = $"Pivot[N={n}, sign=";

            if (Sign > 0) {
                FixedString32Bytes signStr = "+1";
                str.Append(signStr);
            }
            else {
                FixedString32Bytes signStr = "-1";
                str.Append(signStr);
            }

            FixedString32Bytes open = "]: (";
            str.Append(open);

            bool truncated = false;
            for (int i = 0; i < n; i++) {

                if (i > 0)
                    str.Append(' ');

                FixedString32Bytes elementStr = $"{indices[i]}";
                str.Append(elementStr);

                if (str.Length > 3500) {
                    truncated = true;
                    break;
                }
            }

            if (truncated) {
                FixedString32Bytes ellipsis = " ...)";
                str.Append(ellipsis);
            }
            else {
                str.Append(')');
            }

            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }

}
