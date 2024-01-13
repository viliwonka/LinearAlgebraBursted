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

        //+copyReplaceAll

        /// <summary>
        /// Applies pivot to vector v inplace, modifying pivot (resets it)
        /// </summary>
        /// <param name="v">Vector</param>
        /// <param name="pivot">Pivot, will be reset [0, 1, 2, .. ]</param>
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

        /// <summary>
        /// Applies pivot to rows of matrix A inplace, modifying pivot (resets it)
        /// </summary>
        /// <param name="A">Matrix being modified inplace</param>
        /// <param name="pivot">Pivot will also be reset</param>
        public static void ApplyRowInpl(ref fProxyMxN A, ref Pivot pivot) {

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

        /// <summary>
        /// Applies pivot to columns of matrix A inplace, modifying pivot (resets it)
        /// </summary>
        /// <param name="A">Matrix being modified inplace</param>
        /// <param name="pivot">Pivot will also be reset</param>
        public static void ApplyColumnInpl(ref fProxyMxN A, ref Pivot pivot) {

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

        /// <summary>
        /// Applies pivot to rows of matrix A inplace, without modifying pivot data
        /// </summary>
        /// <param name="A">Matrix being modified inplace</param>
        public void ApplyRow(ref fProxyMxN A) {

            if (A.M_Rows != this.N)
                throw new System.ArgumentException("Matrix rows and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyRowInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }


        /// <summary>
        /// Applies pivot to columns of matrix A inplace, without modifying pivot data
        /// </summary>
        /// <param name="A">Matrix being modified inplace</param>
        public void ApplyColumn(ref fProxyMxN A) {
            
            if (A.N_Cols != this.N)
                throw new System.ArgumentException("Matrix columns and pivot must have same dimension");

            Pivot tempPivot = Copy();
            
            ApplyColumnInpl(ref A, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>
        /// Applies pivot to vector v inplace, without modifying pivot data
        /// </summary>
        /// <param name="v">Vector being modified inplace</param>
        public void ApplyVec(ref fProxyN v) {

            if(v.N != this.N)
                throw new System.ArgumentException("Vector and pivot must have same dimension");

            Pivot tempPivot = Copy();

            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>
        /// Applies inverse pivot to vector v inplace, without modifying pivot data
        /// </summary>
        /// <param name="v">Vector being modified inplace</param>
        public void ApplyInverseVec(ref fProxyN v) {

            Pivot tempPivot = InverseCopy();
            
            ApplyVecInpl(ref v, ref tempPivot);

            tempPivot.Dispose();
        }

        /// <summary>
        /// Applies inverse pivot to rows of matrix A inplace, without modifying pivot data
        /// </summary>
        /// <param name="A">Matrix being modified inplace</param>
        public void ApplyInverseRow(ref fProxyMxN A) {

            Pivot tempPivot = InverseCopy();

            ApplyRowInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }

        /// <summary>
        /// Applies inverse pivot to columns of matrix A inplace, without modifying pivot data
        /// </summary>
        /// <param name="A">Matrix being modified inplace</param>
        public void ApplyInverseColumn(ref fProxyMxN A) {

            Pivot tempPivot = InverseCopy();
            
            ApplyColumnInpl(ref A, ref tempPivot);
            
            tempPivot.Dispose();
        }
        //-copyReplaceAll

    }

}