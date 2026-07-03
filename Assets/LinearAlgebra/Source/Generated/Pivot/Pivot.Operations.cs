using System.Collections;
using System.Collections.Generic;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.CompilerServices;

using UnityEngine;
using LinearAlgebra.Internal;

namespace LinearAlgebra {

    // Pivot is a more efficient replacement for permutation matrix,
    // for easier use in algorithms like LU decomposition
    public partial struct Pivot {

        

        /// <summary>Applies pivot to vector v inplace; resets pivot to [0, 1, 2, ...].</summary>
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

        /// <summary>Applies pivot to rows of matrix A inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyRowInpl(ref floatMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        Unsafe_OP.swapRows(A.Data.Ptr, fromR, toR, A.N_Cols);
                    }
                    
                    toR = pivot.indices[fromR];
                }
            }
        }

        /// <summary>Applies pivot to columns of matrix A inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyColumnInpl(ref floatMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        Unsafe_OP.swapColumns(A.Data.Ptr, fromR, toR, A.M_Rows, A.N_Cols);
                    }

                    toR = pivot.indices[fromR];
                }

            }
        }

        /// <summary>Applies pivot to rows of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyRow(ref floatMxN A) {

            if (A.M_Rows != this.N)
                throw new System.ArgumentException("Matrix rows and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyRowInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }


        /// <summary>Applies pivot to columns of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyColumn(ref floatMxN A) {
            
            if (A.N_Cols != this.N)
                throw new System.ArgumentException("Matrix columns and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyColumnInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies pivot to vector v; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyVec(ref floatN v) {

            if(v.N != this.N)
                throw new System.ArgumentException("Vector and pivot must have same dimension");

            Pivot tempPivot = Copy();

            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to vector v; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseVec(ref floatN v) {

            Pivot tempPivot = InverseCopy();
            
            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to rows of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseRow(ref floatMxN A) {

            Pivot tempPivot = InverseCopy();

            ApplyRowInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to columns of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseColumn(ref floatMxN A) {

            Pivot tempPivot = InverseCopy();
            
            ApplyColumnInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }
        

        /// <summary>Applies pivot to vector v inplace; resets pivot to [0, 1, 2, ...].</summary>
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

        /// <summary>Applies pivot to rows of matrix A inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyRowInpl(ref doubleMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        Unsafe_OP.swapRows(A.Data.Ptr, fromR, toR, A.N_Cols);
                    }
                    
                    toR = pivot.indices[fromR];
                }
            }
        }

        /// <summary>Applies pivot to columns of matrix A inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyColumnInpl(ref doubleMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        Unsafe_OP.swapColumns(A.Data.Ptr, fromR, toR, A.M_Rows, A.N_Cols);
                    }

                    toR = pivot.indices[fromR];
                }

            }
        }

        /// <summary>Applies pivot to rows of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyRow(ref doubleMxN A) {

            if (A.M_Rows != this.N)
                throw new System.ArgumentException("Matrix rows and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyRowInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }


        /// <summary>Applies pivot to columns of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyColumn(ref doubleMxN A) {
            
            if (A.N_Cols != this.N)
                throw new System.ArgumentException("Matrix columns and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyColumnInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies pivot to vector v; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyVec(ref doubleN v) {

            if(v.N != this.N)
                throw new System.ArgumentException("Vector and pivot must have same dimension");

            Pivot tempPivot = Copy();

            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to vector v; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseVec(ref doubleN v) {

            Pivot tempPivot = InverseCopy();
            
            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to rows of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseRow(ref doubleMxN A) {

            Pivot tempPivot = InverseCopy();

            ApplyRowInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to columns of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseColumn(ref doubleMxN A) {

            Pivot tempPivot = InverseCopy();
            
            ApplyColumnInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }
        

        /// <summary>Applies pivot to vector v inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyVecInpl(ref intN v, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    int tempElement = v[toR];
                    v[toR] = v[fromR];
                    v[fromR] = tempElement;

                    pivot.Swap(fromR, toR);

                    toR = pivot.indices[fromR];
                }
            }
        }

        /// <summary>Applies pivot to rows of matrix A inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyRowInpl(ref intMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        Unsafe_OP.swapRows(A.Data.Ptr, fromR, toR, A.N_Cols);
                    }
                    
                    toR = pivot.indices[fromR];
                }
            }
        }

        /// <summary>Applies pivot to columns of matrix A inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyColumnInpl(ref intMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        Unsafe_OP.swapColumns(A.Data.Ptr, fromR, toR, A.M_Rows, A.N_Cols);
                    }

                    toR = pivot.indices[fromR];
                }

            }
        }

        /// <summary>Applies pivot to rows of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyRow(ref intMxN A) {

            if (A.M_Rows != this.N)
                throw new System.ArgumentException("Matrix rows and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyRowInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }


        /// <summary>Applies pivot to columns of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyColumn(ref intMxN A) {
            
            if (A.N_Cols != this.N)
                throw new System.ArgumentException("Matrix columns and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyColumnInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies pivot to vector v; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyVec(ref intN v) {

            if(v.N != this.N)
                throw new System.ArgumentException("Vector and pivot must have same dimension");

            Pivot tempPivot = Copy();

            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to vector v; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseVec(ref intN v) {

            Pivot tempPivot = InverseCopy();
            
            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to rows of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseRow(ref intMxN A) {

            Pivot tempPivot = InverseCopy();

            ApplyRowInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to columns of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseColumn(ref intMxN A) {

            Pivot tempPivot = InverseCopy();
            
            ApplyColumnInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }
        

        /// <summary>Applies pivot to vector v inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyVecInpl(ref shortN v, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    short tempElement = v[toR];
                    v[toR] = v[fromR];
                    v[fromR] = tempElement;

                    pivot.Swap(fromR, toR);

                    toR = pivot.indices[fromR];
                }
            }
        }

        /// <summary>Applies pivot to rows of matrix A inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyRowInpl(ref shortMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        Unsafe_OP.swapRows(A.Data.Ptr, fromR, toR, A.N_Cols);
                    }
                    
                    toR = pivot.indices[fromR];
                }
            }
        }

        /// <summary>Applies pivot to columns of matrix A inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyColumnInpl(ref shortMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        Unsafe_OP.swapColumns(A.Data.Ptr, fromR, toR, A.M_Rows, A.N_Cols);
                    }

                    toR = pivot.indices[fromR];
                }

            }
        }

        /// <summary>Applies pivot to rows of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyRow(ref shortMxN A) {

            if (A.M_Rows != this.N)
                throw new System.ArgumentException("Matrix rows and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyRowInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }


        /// <summary>Applies pivot to columns of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyColumn(ref shortMxN A) {
            
            if (A.N_Cols != this.N)
                throw new System.ArgumentException("Matrix columns and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyColumnInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies pivot to vector v; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyVec(ref shortN v) {

            if(v.N != this.N)
                throw new System.ArgumentException("Vector and pivot must have same dimension");

            Pivot tempPivot = Copy();

            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to vector v; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseVec(ref shortN v) {

            Pivot tempPivot = InverseCopy();
            
            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to rows of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseRow(ref shortMxN A) {

            Pivot tempPivot = InverseCopy();

            ApplyRowInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to columns of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseColumn(ref shortMxN A) {

            Pivot tempPivot = InverseCopy();
            
            ApplyColumnInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }
        

        /// <summary>Applies pivot to vector v inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyVecInpl(ref longN v, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    long tempElement = v[toR];
                    v[toR] = v[fromR];
                    v[fromR] = tempElement;

                    pivot.Swap(fromR, toR);

                    toR = pivot.indices[fromR];
                }
            }
        }

        /// <summary>Applies pivot to rows of matrix A inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyRowInpl(ref longMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        Unsafe_OP.swapRows(A.Data.Ptr, fromR, toR, A.N_Cols);
                    }
                    
                    toR = pivot.indices[fromR];
                }
            }
        }

        /// <summary>Applies pivot to columns of matrix A inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyColumnInpl(ref longMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        Unsafe_OP.swapColumns(A.Data.Ptr, fromR, toR, A.M_Rows, A.N_Cols);
                    }

                    toR = pivot.indices[fromR];
                }

            }
        }

        /// <summary>Applies pivot to rows of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyRow(ref longMxN A) {

            if (A.M_Rows != this.N)
                throw new System.ArgumentException("Matrix rows and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyRowInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }


        /// <summary>Applies pivot to columns of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyColumn(ref longMxN A) {
            
            if (A.N_Cols != this.N)
                throw new System.ArgumentException("Matrix columns and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyColumnInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies pivot to vector v; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyVec(ref longN v) {

            if(v.N != this.N)
                throw new System.ArgumentException("Vector and pivot must have same dimension");

            Pivot tempPivot = Copy();

            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to vector v; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseVec(ref longN v) {

            Pivot tempPivot = InverseCopy();
            
            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to rows of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseRow(ref longMxN A) {

            Pivot tempPivot = InverseCopy();

            ApplyRowInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to columns of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseColumn(ref longMxN A) {

            Pivot tempPivot = InverseCopy();
            
            ApplyColumnInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }
        

        /// <summary>Applies pivot to vector v inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyVecInpl(ref boolN v, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    bool tempElement = v[toR];
                    v[toR] = v[fromR];
                    v[fromR] = tempElement;

                    pivot.Swap(fromR, toR);

                    toR = pivot.indices[fromR];
                }
            }
        }

        /// <summary>Applies pivot to rows of matrix A inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyRowInpl(ref boolMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        Unsafe_OP.swapRows(A.Data.Ptr, fromR, toR, A.N_Cols);
                    }
                    
                    toR = pivot.indices[fromR];
                }
            }
        }

        /// <summary>Applies pivot to columns of matrix A inplace; resets pivot to [0, 1, 2, ...].</summary>
        public static void ApplyColumnInpl(ref boolMxN A, ref Pivot pivot) {

            for (int fromR = 0; fromR < pivot.N; fromR++) {

                int toR = pivot.indices[fromR];

                while (toR != fromR) {

                    pivot.Swap(fromR, toR);
                    
                    unsafe {
                        Unsafe_OP.swapColumns(A.Data.Ptr, fromR, toR, A.M_Rows, A.N_Cols);
                    }

                    toR = pivot.indices[fromR];
                }

            }
        }

        /// <summary>Applies pivot to rows of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyRow(ref boolMxN A) {

            if (A.M_Rows != this.N)
                throw new System.ArgumentException("Matrix rows and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyRowInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }


        /// <summary>Applies pivot to columns of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyColumn(ref boolMxN A) {
            
            if (A.N_Cols != this.N)
                throw new System.ArgumentException("Matrix columns and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyColumnInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies pivot to vector v; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyVec(ref boolN v) {

            if(v.N != this.N)
                throw new System.ArgumentException("Vector and pivot must have same dimension");

            Pivot tempPivot = Copy();

            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to vector v; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseVec(ref boolN v) {

            Pivot tempPivot = InverseCopy();
            
            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to rows of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseRow(ref boolMxN A) {

            Pivot tempPivot = InverseCopy();

            ApplyRowInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }

        /// <summary>Applies the inverse pivot to columns of matrix A; copies the pivot first, so the original pivot is left unchanged.</summary>
        public void ApplyInverseColumn(ref boolMxN A) {

            Pivot tempPivot = InverseCopy();
            
            ApplyColumnInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }
        

    }

}