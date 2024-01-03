using System.Collections;
using System.Collections.Generic;

using UnityEditor;

using UnityEngine;

namespace LinearAlgebra {

    // A m x n matrix view, to read, write or operate on a submatrix of a matrix
    // Also useful to get vectors or columns from a matrix
    public struct viewBoolMxN {

        // could just use UnsafeList<bool> and support both boolMxN and boolN
        public unsafe boolMxN mat;

        public int sizeN;
        public int sizeM;

        public int posM;
        public int posN;

        public int strideM;
        public int strideN;

        public viewBoolMxN(ref boolMxN mat, int posM, int posN, int sizeM, int sizeN, int strideM = 1, int strideN = 1) {
            
            if (strideM < 1 || strideN < 1)
                throw new System.ArgumentOutOfRangeException("both strideM and strideN must be greater than 0");

            if (posN + sizeN * strideN > mat.N_Cols || posM + sizeM * strideM > mat.M_Rows)
                throw new System.ArgumentOutOfRangeException("strideN * sizeN or strideM * sizeM above matrix dimension");

            if (posN < 0 || posM < 0)
                throw new System.ArgumentOutOfRangeException("posN or posM must be greater than 0");

            this.sizeN = sizeN;
            this.sizeM = sizeM;

            this.posM = posM;
            this.posN = posN;

            this.strideM = strideM;
            this.strideN = strideN;
            
            this.mat = mat;
        }

        public ref bool this[int r, int c] {
            get {

                if(r < 0 || r >= sizeM || c < 0 || c >= sizeN)
                    throw new System.ArgumentOutOfRangeException("r or c out of range");

                return ref mat[posM + r * strideM, posN + c * strideN];
                
            }
        }
        // row major order of elements in matrix
        public ref bool this[int i] {
            get {

                if(i < 0 || i >= sizeM * sizeN)
                    throw new System.ArgumentOutOfRangeException("i out of range");
                
                //todo: check correctness, idk what it does
                return ref mat[posM + i / sizeN * strideM, posN + i % sizeN * strideN];
            }
        }

        // Example operators
        public static viewBoolMxN operator |(viewBoolMxN lhs, bool rhs) {

            int len = lhs.sizeM * lhs.sizeN;

            for (int i = 0; i < len; i++)
                lhs[i] = lhs[i] | rhs;
            
            return lhs;
        }

        public static viewBoolMxN operator |(viewBoolMxN lhs, viewBoolMxN rhs) {

            if(lhs.sizeM != rhs.sizeM || lhs.sizeN != rhs.sizeN)
                throw new System.ArgumentException("lhs and rhs must have same size");

            int len = lhs.sizeM * lhs.sizeN;

            for (int i = 0; i < len; i++)
                lhs[i] = lhs[i] | rhs[i];

            return lhs;
        }

        public static viewBoolMxN operator |(viewBoolMxN lhs, boolMxN rhs) {

            if (lhs.sizeM != rhs.M_Rows || lhs.sizeN != rhs.N_Cols)
                throw new System.ArgumentException("lhs and rhs must have same size");

            int len = lhs.sizeM * lhs.sizeN;

            for (int i = 0; i < len; i++)
                lhs[i] = lhs[i] | rhs[i];

            return lhs;
        }
    }

}