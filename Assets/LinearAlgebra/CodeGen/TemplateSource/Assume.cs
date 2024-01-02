using Unity.Mathematics;
using System;

namespace LinearAlgebra
{
    internal static partial class Assume
    {
        internal static void SameDim(int a, int b)
        {
            if (a != b)
                throw new ArgumentException("Vectors or matrices must have compatible dimension");
        }

        internal static void IndexInsideBounds(in int2 dim, int2 index)
        {
            if (math.any(0 < index) && math.any(index >= dim))
                throw new ArgumentException("Index out of bounds");
        }

        internal static void SameDim(in boolN a, in boolN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("Vectors must have same dimension");
        }

        internal static void SameDim(in boolMxN a, in boolMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("Matrices must have same dimension");
        }

    }
}