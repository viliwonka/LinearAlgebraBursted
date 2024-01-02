using Unity.Mathematics;

//+deleteThis
using LinearAlgebra.mathProxies;
//-deleteThis

namespace LinearAlgebra
{
    // have to test it all, have to do other conversions too
    public static partial class ArenaExtensions {

        #region CONVERSIONS_FROM_MATH
        public static fProxyN Convert(this ref Arena arena, in fProxy2 mathVec)
        {
            var vec = arena.fProxyVec(2, true);

            vec[0] = mathVec.x;
            vec[1] = mathVec.y;

            return vec;
        }

        public static fProxyN Convert(this ref Arena arena, in fProxy3 mathVec)
        {
            var vec = arena.fProxyVec(3, true);

            vec[0] = mathVec.x;
            vec[1] = mathVec.y;
            vec[2] = mathVec.z;

            return vec;
        }

        public static fProxyN Convert(this ref Arena arena, in fProxy4 mathVec)
        {
            var vec = arena.fProxyVec(4, true);

            vec[0] = mathVec.x;
            vec[1] = mathVec.y;
            vec[2] = mathVec.z;
            vec[3] = mathVec.w;

            return vec;
        }

        public static fProxyMxN Convert(this ref Arena arena, in fProxy2x2 mathMat)
        {
            var mat = arena.fProxyMat(2, 2, true);

            mat[0, 0] = mathMat.c0.x;
            mat[1, 0] = mathMat.c0.y;
            mat[0, 1] = mathMat.c1.x;
            mat[1, 1] = mathMat.c1.y;

            return mat;
        }

        public static fProxyMxN Convert(this ref Arena arena, in fProxy3x3 mathMat)
        {
            var mat = arena.fProxyMat(3, 3, true);

            mat[0, 0] = mathMat.c0.x;
            mat[1, 0] = mathMat.c0.y;
            mat[2, 0] = mathMat.c0.z;
            mat[0, 1] = mathMat.c1.x;
            mat[1, 1] = mathMat.c1.y;
            mat[2, 1] = mathMat.c1.z;
            mat[0, 2] = mathMat.c2.x;
            mat[1, 2] = mathMat.c2.y;
            mat[2, 2] = mathMat.c2.z;

            return mat;
        }

        public static fProxyMxN Convert(this ref Arena arena, in fProxy4x4 mathMat)
        {
            var mat = arena.fProxyMat(4, 4, true);

            mat[0, 0] = mathMat.c0.x;
            mat[1, 0] = mathMat.c0.y;
            mat[2, 0] = mathMat.c0.z;
            mat[3, 0] = mathMat.c0.w;
            mat[0, 1] = mathMat.c1.x;
            mat[1, 1] = mathMat.c1.y;
            mat[2, 1] = mathMat.c1.z;
            mat[3, 1] = mathMat.c1.w;
            mat[0, 2] = mathMat.c2.x;
            mat[1, 2] = mathMat.c2.y;
            mat[2, 2] = mathMat.c2.z;
            mat[3, 2] = mathMat.c2.w;
            mat[0, 3] = mathMat.c3.x;
            mat[1, 3] = mathMat.c3.y;
            mat[2, 3] = mathMat.c3.z;
            mat[3, 3] = mathMat.c3.w;

            return mat;
        }

        #endregion

        #region CONVERSIONS_TO_MATH
        public static fProxy2 Convert(this ref Arena arena, in fProxyN mathVec) {
            var vec = new fProxy2();

            vec.x = mathVec[0];
            vec.y = mathVec[1];

            return vec;
        }
        #endregion


    }
}