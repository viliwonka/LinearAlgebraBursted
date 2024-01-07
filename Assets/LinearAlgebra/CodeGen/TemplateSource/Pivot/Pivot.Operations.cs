using System.Collections;
using System.Collections.Generic;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.CompilerServices;

using UnityEngine;

namespace LinearAlgebra {
    
    // TODO: Arena dependency?
    // might be smart before writing more tests

    // Pivot is a more efficient replacement for permutation matrix,
    // for easier use in algorithms like LU decomposition
    public partial struct Pivot {

        //+copyReplace

        // Applies pivot to vector v inplace, modifying pivot (resets it)
        public static void ApplyVecInpl(ref fProxyN v, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    fProxy tempElement = v[toR];
                    v[toR] = v[fromR];
                    v[fromR] = tempElement;

                    pivot.Swap(fromR, toR);

                    toR = pivot.indices[fromR];
                }
            }
        }

        // Applies pivot to rows of matrix A inplace, modifying pivot (resets it)
        public static void ApplyRowInpl(ref fProxyMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    for (int k = 0; k < A.N_Cols; k++) {
                        fProxy tempElement = A[toR, k];
                        A[toR, k] = A[fromR, k];
                        A[fromR, k] = tempElement;
                    }

                    pivot.Swap(fromR, toR);

                    toR = pivot.indices[fromR];
                }
            }
        }

        // Applies pivot to columns of matrix A inplace, modifying pivot (resets it)
        public static void ApplyColumnInpl(ref fProxyMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    for (int k = 0; k < A.M_Rows; k++) {
                        fProxy tempElement = A[k, toR];
                        A[k, toR] = A[k, fromR];
                        A[k, fromR] = tempElement;
                    }

                    pivot.Swap(fromR, toR);

                    toR = pivot.indices[fromR];
                }

            }
        }

        // Applies permutation operation on rows of matrix
        public void ApplyRow(ref fProxyMxN A) {

            if (A.M_Rows != this.N)
                throw new System.ArgumentException("Matrix rows and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyRowInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }

        // Applies permutation operation on columns of matrix
        public void ApplyColumn(ref fProxyMxN A) {
            
            if (A.N_Cols != this.N)
                throw new System.ArgumentException("Matrix columns and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyColumnInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }

        // Applies permutation operation on vector
        public void ApplyVec(ref fProxyN v) {

            if(v.N != this.N)
                throw new System.ArgumentException("Vector and pivot must have same dimension");

            Pivot tempPivot = Copy();

            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        // Applies inverse permutation operation on vector
        public void ApplyInverseVec(ref fProxyN v) {

            Pivot tempPivot = InverseCopy();
            
            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        // Applies inverse permutation operation on rows of matrix
        public void ApplyInverseRow(ref fProxyMxN A) {

            Pivot tempPivot = InverseCopy();

            ApplyRowInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }

        // Applies inverse permutation operation on columns of matrix
        public void ApplyInverseColumn(ref fProxyMxN A) {

            Pivot tempPivot = InverseCopy();
            
            ApplyColumnInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }
        //-copyReplace

    }

}