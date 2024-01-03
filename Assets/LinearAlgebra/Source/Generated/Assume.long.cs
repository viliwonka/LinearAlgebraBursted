using Unity.Mathematics;
using System;

namespace LinearAlgebra
{
    internal static partial class Assume
    {
        internal static void SameDim(in longN a, in longN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("longNs must have same dimension");
            
        }

        internal static void SameDim(in longMxN a, in longMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("longMxNs must have same dimension");
            
        }

        internal static void SameDim(in longN a, in boolN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("longN and boolN must have same dimension");
        }

        internal static void SameDim(in longMxN a, in boolMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("longMxN and boolMxN must have same dimension");
        }
    }
}