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

        

        // Applies pivot to vector v inplace, modifying pivot (resets it)
        public static void ApplyVecInpl(ref floatN v, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    float tempElement = v[toR];
                    v[toR] = v[fromR];
                    v[fromR] = tempElement;

                    pivot.Swap(fromR, toR);

                    toR = pivot.indices[fromR];
                }
            }
        }

        // Applies pivot to rows of matrix A inplace, modifying pivot (resets it)
        private static void ApplyRowInpl(ref floatMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        UnsafeOP.swapRows(A.Data.Ptr, fromR, toR, A.N_Cols);
                    }
                    
                    toR = pivot.indices[fromR];
                }
            }
        }

        // Applies pivot to columns of matrix A inplace, modifying pivot (resets it)
        private static void ApplyColumnInpl(ref floatMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        UnsafeOP.swapColumns(A.Data.Ptr, fromR, toR, A.M_Rows, A.N_Cols);
                    }

                    toR = pivot.indices[fromR];
                }

            }
        }

        // Applies permutation operation on rows of matrix
        public void ApplyRow(ref floatMxN A) {

            if (A.M_Rows != this.N)
                throw new System.ArgumentException("Matrix rows and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyRowInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }

        // Applies permutation operation on columns of matrix
        public void ApplyColumn(ref floatMxN A) {
            
            if (A.N_Cols != this.N)
                throw new System.ArgumentException("Matrix columns and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyColumnInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }

        // Applies permutation operation on vector
        public void ApplyVec(ref floatN v) {

            if(v.N != this.N)
                throw new System.ArgumentException("Vector and pivot must have same dimension");

            Pivot tempPivot = Copy();

            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        // Applies inverse permutation operation on vector
        public void ApplyInverseVec(ref floatN v) {

            Pivot tempPivot = InverseCopy();
            
            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        // Applies inverse permutation operation on rows of matrix
        public void ApplyInverseRow(ref floatMxN A) {

            Pivot tempPivot = InverseCopy();

            ApplyRowInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }

        // Applies inverse permutation operation on columns of matrix
        public void ApplyInverseColumn(ref floatMxN A) {

            Pivot tempPivot = InverseCopy();
            
            ApplyColumnInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }
        

        // Applies pivot to vector v inplace, modifying pivot (resets it)
        public static void ApplyVecInpl(ref doubleN v, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    double tempElement = v[toR];
                    v[toR] = v[fromR];
                    v[fromR] = tempElement;

                    pivot.Swap(fromR, toR);

                    toR = pivot.indices[fromR];
                }
            }
        }

        // Applies pivot to rows of matrix A inplace, modifying pivot (resets it)
        private static void ApplyRowInpl(ref doubleMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        UnsafeOP.swapRows(A.Data.Ptr, fromR, toR, A.N_Cols);
                    }
                    
                    toR = pivot.indices[fromR];
                }
            }
        }

        // Applies pivot to columns of matrix A inplace, modifying pivot (resets it)
        private static void ApplyColumnInpl(ref doubleMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        UnsafeOP.swapColumns(A.Data.Ptr, fromR, toR, A.M_Rows, A.N_Cols);
                    }

                    toR = pivot.indices[fromR];
                }

            }
        }

        // Applies permutation operation on rows of matrix
        public void ApplyRow(ref doubleMxN A) {

            if (A.M_Rows != this.N)
                throw new System.ArgumentException("Matrix rows and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyRowInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }

        // Applies permutation operation on columns of matrix
        public void ApplyColumn(ref doubleMxN A) {
            
            if (A.N_Cols != this.N)
                throw new System.ArgumentException("Matrix columns and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyColumnInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }

        // Applies permutation operation on vector
        public void ApplyVec(ref doubleN v) {

            if(v.N != this.N)
                throw new System.ArgumentException("Vector and pivot must have same dimension");

            Pivot tempPivot = Copy();

            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        // Applies inverse permutation operation on vector
        public void ApplyInverseVec(ref doubleN v) {

            Pivot tempPivot = InverseCopy();
            
            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        // Applies inverse permutation operation on rows of matrix
        public void ApplyInverseRow(ref doubleMxN A) {

            Pivot tempPivot = InverseCopy();

            ApplyRowInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }

        // Applies inverse permutation operation on columns of matrix
        public void ApplyInverseColumn(ref doubleMxN A) {

            Pivot tempPivot = InverseCopy();
            
            ApplyColumnInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }
        

    }

}