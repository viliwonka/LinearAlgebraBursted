using Unity.Mathematics;
using Unity.Collections;

//+deleteThis
// TEMPLATE-ONLY alias: codegen rewrites each fProxy* token -> float*/double* (real Unity.Mathematics
// types), so the field access below (.x/.y/.z/.w, .c0/.c1/.c2/.c3) resolves natively -- no proxy-struct
// shim needed. See proxyStructs.math.cs / docs/dev/spec-alias-simd-proxies.md.
using fProxy2 = Unity.Mathematics.float2;
using fProxy3 = Unity.Mathematics.float3;
using fProxy4 = Unity.Mathematics.float4;
using fProxy2x2 = Unity.Mathematics.float2x2;
using fProxy3x3 = Unity.Mathematics.float3x3;
using fProxy4x4 = Unity.Mathematics.float4x4;
//-deleteThis

namespace LinearAlgebra
{
    // Standalone (non-arena) conversions between fixed-size Unity.Mathematics types and dynamic
    // vectors/matrices. Same semantics as ArenaConversions.fProxy.cs: forward converters allocate
    // a fresh fProxyN/fProxyMxN via allocator; reverse converters read into a fixed-size value and
    // allocate nothing.
    public static partial class ConvertOP
    {
        #region CONVERSIONS_FROM_MATH
        public static fProxyN Convert(in fProxy2 mathVec, Allocator allocator = Allocator.Temp)
        {
            var vec = new fProxyN(2, allocator, true);

            vec[0] = mathVec.x;
            vec[1] = mathVec.y;

            return vec;
        }

        public static fProxyN Convert(in fProxy3 mathVec, Allocator allocator = Allocator.Temp)
        {
            var vec = new fProxyN(3, allocator, true);

            vec[0] = mathVec.x;
            vec[1] = mathVec.y;
            vec[2] = mathVec.z;

            return vec;
        }

        public static fProxyN Convert(in fProxy4 mathVec, Allocator allocator = Allocator.Temp)
        {
            var vec = new fProxyN(4, allocator, true);

            vec[0] = mathVec.x;
            vec[1] = mathVec.y;
            vec[2] = mathVec.z;
            vec[3] = mathVec.w;

            return vec;
        }

        public static fProxyMxN Convert(in fProxy2x2 mathMat, Allocator allocator = Allocator.Temp)
        {
            var mat = new fProxyMxN(2, 2, allocator, true);

            mat[0, 0] = mathMat.c0.x;
            mat[1, 0] = mathMat.c0.y;
            mat[0, 1] = mathMat.c1.x;
            mat[1, 1] = mathMat.c1.y;

            return mat;
        }

        public static fProxyMxN Convert(in fProxy3x3 mathMat, Allocator allocator = Allocator.Temp)
        {
            var mat = new fProxyMxN(3, 3, allocator, true);

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

        public static fProxyMxN Convert(in fProxy4x4 mathMat, Allocator allocator = Allocator.Temp)
        {
            var mat = new fProxyMxN(4, 4, allocator, true);

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
        public static fProxy2 Convert(in fProxyN mathVec) {
            if (mathVec.N < 2)
                throw new System.ArgumentException("Convert(fProxyN -> fProxy2): source vector must have length >= 2");

            var vec = new fProxy2();

            vec.x = mathVec[0];
            vec.y = mathVec[1];

            return vec;
        }
        #endregion


    }
}
