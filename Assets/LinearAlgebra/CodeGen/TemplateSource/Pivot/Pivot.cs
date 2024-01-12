using System.Collections;
using System.Collections.Generic;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.CompilerServices;

using UnityEngine;

namespace LinearAlgebra {
    
    // TODO: Finish the struct, that can be used for pivoting in algorithms like LU decomposition
    // Arena dependency?
    public partial struct Pivot {

        private UnsafeList<int> indices;

        public int N => indices.Length;

        public Pivot(int size, Allocator allocator = Allocator.Temp) {
            indices = new UnsafeList<int>(size, allocator);
            indices.Resize(size);
            Reset();
        }

        public int this[int i] {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => indices[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Swap(int i, int j) {

            if (i == j)
                return;

            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        public void Reset() {
            for (int i = 0; i < indices.Length; i++)
                indices[i] = i;
        }

        public void Dispose() {
            indices.Dispose();
        }

        public Pivot Copy() {

            var copy = new Pivot(indices.Length, Allocator.Temp);

            copy.indices.CopyFrom(indices);

            return copy;
        }

        public Pivot InverseCopy() {

            var copy = new Pivot(indices.Length, Allocator.Temp);

            for (int i = 0; i < indices.Length; i++)
                copy.indices[this[i]] = i;

            return copy;
        }

        // Inverse operation inplace
        public void InverseInplace() {

            var tempPivot = new Pivot(indices.Length, Allocator.Temp);

            // copy original into tempPivot
            for (int i = 0; i < indices.Length; i++)
                tempPivot.indices[i] = indices[i];

            for (int i = 0; i < indices.Length; i++)
                indices[tempPivot[i]] = i;

            tempPivot.Dispose();
        }

        public void Print() {

            FixedString4096Bytes toPrint = new FixedString4096Bytes();

            for (int i = 0; i < indices.Length; i++)
                toPrint.Append($"{indices[i]}");

            Debug.Log(toPrint);
        }
    }

}