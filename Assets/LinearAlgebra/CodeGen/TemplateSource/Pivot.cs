using System.Collections;
using System.Collections.Generic;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.CompilerServices;

using UnityEngine;

namespace LinearAlgebra {
    
    // TODO: Finish the struct, that can be used for pivoting in algorithms like LU decomposition
    public struct Pivot {

        public UnsafeList<int> indices;
        
        public Pivot(int size, Allocator allocator = Allocator.Temp) {
            indices = new UnsafeList<int>(size, allocator);
            indices.Resize(size);
            Reset();
        }

        public int this[int i] {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => indices[i];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => indices[i] = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Swap(int from, int to) {

            if (from == to)
                return;

            int temp = indices[from];
            indices[from] = indices[to];
            indices[to] = temp;
        }

        public void Reset() {
            for (int i = 0; i < indices.Length; i++)
                indices[i] = i;
        }

        public void Dispose() {
            indices.Dispose();
        }
    }

}