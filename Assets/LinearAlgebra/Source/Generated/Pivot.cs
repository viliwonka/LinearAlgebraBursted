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

        // Applies pivot to matrix A inplace
        // Note: it is still acting as a permutation matrix operation P, so applying twice will not yield original result!
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

            }

            tempPivot.Dispose();
        }

        // Applies operation inplace
        // Inverse operation of 'permutation matrix', so apply ApplyRow and ApplyInverseRow to matrix A will yield original result
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

            }
            tempPivot.Dispose();

            tempInversePivot.ApplyRow<M, T>(ref A);

            tempInversePivot.Dispose();
        }


        public Pivot Copy() {

            var copy = new Pivot(indices.Length, Allocator.Temp);

            for (int i = 0; i < indices.Length; i++)
                copy.indices[i] = indices[i];

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