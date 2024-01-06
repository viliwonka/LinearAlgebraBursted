using System.Collections;
using System.Collections.Generic;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.CompilerServices;

using UnityEngine;

namespace LinearAlgebra {
    
    // TODO: Finish the struct, that can be used for pivoting in algorithms like LU decomposition
    public struct Pivot {

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

        public void ApplyRow<M, T>(ref M A) where M : unmanaged, IMatrix<T> where T : unmanaged {

            Pivot tempPivot = Copy();

            tempPivot.Print();

            for (int fromR = 0; fromR < tempPivot.indices.Length; fromR++) {

                int toR = tempPivot.indices[fromR];

                while(toR != fromR) {

                    for (int k = 0; k < A.N_Cols; k++) {
                        T tempElement = A[toR, k];
                        A[toR, k] = A[fromR, k];
                        A[fromR, k] = tempElement;
                    }

                    tempPivot.Swap(fromR, toR);

                    toR = tempPivot.indices[fromR];
                }

                tempPivot.Print(); 
            }

            tempPivot.Dispose();
        }

        public void ApplyInverseRow<M, T>(ref M A) where M : unmanaged, IMatrix<T> where T : unmanaged {

            Pivot tempPivot = Copy();
            Pivot tempInversePivot = new Pivot(tempPivot.N, Allocator.Temp);
            tempPivot.Print();

            for (int fromR = 0; fromR < tempPivot.indices.Length; fromR++) {
                int toR = tempPivot.indices[fromR];
                while (toR != fromR) {
                    tempPivot.Swap(fromR, toR);
                    tempInversePivot.Swap(fromR, toR);
                    toR = tempPivot.indices[fromR];
                }

                tempPivot.Print();
            }
            tempPivot.Dispose();

            tempInversePivot.ApplyRow<M, T>(ref A);

            tempInversePivot.Dispose();
        }


        public Pivot Copy() {

            var copy = new Pivot(indices.Length, Allocator.Temp);

            for (int i = 0; i < indices.Length; i++)
                copy[i] = indices[i];

            return copy;
        }

        public void Print() {

            FixedString4096Bytes toPrint = new FixedString4096Bytes();

            for (int i = 0; i < indices.Length; i++)
                toPrint.Append($"{indices[i]}");

            Debug.Log(toPrint);
        }
    }

}