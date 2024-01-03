using Unity.Mathematics;



namespace LinearAlgebra
{
    // have to test it all, have to do other conversions too
    public static partial class ArenaExtensions {

        #region CONVERSIONS_FROM_MATH
        public static floatN Convert(this ref Arena arena, in float2 mathVec)
        {
            var vec = arena.floatVec(2, true);

            vec[0] = mathVec.x;
            vec[1] = mathVec.y;

            return vec;
        }

        public static floatN Convert(this ref Arena arena, in float3 mathVec)
        {
            var vec = arena.floatVec(3, true);

            vec[0] = mathVec.x;
            vec[1] = mathVec.y;
            vec[2] = mathVec.z;

            return vec;
        }

        public static floatN Convert(this ref Arena arena, in float4 mathVec)
        {
            var vec = arena.floatVec(4, true);

            vec[0] = mathVec.x;
            vec[1] = mathVec.y;
            vec[2] = mathVec.z;
            vec[3] = mathVec.w;

            return vec;
        }

        public static floatMxN Convert(this ref Arena arena, in float2x2 mathMat)
        {
            var mat = arena.floatMat(2, 2, true);

            mat[0, 0] = mathMat.c0.x;
            mat[1, 0] = mathMat.c0.y;
            mat[0, 1] = mathMat.c1.x;
            mat[1, 1] = mathMat.c1.y;

            return mat;
        }

        public static floatMxN Convert(this ref Arena arena, in float3x3 mathMat)
        {
            var mat = arena.floatMat(3, 3, true);

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

        public static floatMxN Convert(this ref Arena arena, in float4x4 mathMat)
        {
            var mat = arena.floatMat(4, 4, true);

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
        public static float2 Convert(this ref Arena arena, in floatN mathVec) {
            var vec = new float2();

            vec.x = mathVec[0];
            vec.y = mathVec[1];

            return vec;
        }
        #endregion


    }
}