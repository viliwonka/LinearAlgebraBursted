using System.Collections;
using System.Collections.Generic;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

using UnityEngine;

namespace LinearAlgebra {
    
    // TODO: Finish the struct, that can be used for pivoting in algorithms like LU decomposition
    public struct Pivot {

        public UnsafeList<int> indices;
        
        public Pivot(int size, Allocator allocator = Allocator.Temp) {
            indices = new UnsafeList<int>(size, allocator);
            indices.Resize(size);
            Reset(0, size);
        }

        public int this[int i] {
            get => indices[i];
            set => indices[i] = value;
        }

        public void Swap(int i, int j) {

            if (i == j)
                return;

            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        public void Reset(int start, int end) {
            for (int i = start; i < end; i++)
                indices[i] = i;
        }

        public void Dispose() {
            indices.Dispose();
        }
    }

}