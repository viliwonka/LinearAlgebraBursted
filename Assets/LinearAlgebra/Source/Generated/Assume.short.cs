using Unity.Mathematics;
using System;

namespace LinearAlgebra
{
    internal static partial class Assume
    {
        internal static void SameDim(in shortN a, in shortN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("shortNs must have same dimension");
            
        }

        internal static void SameDim(in shortMxN a, in shortMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("shortMxNs must have same dimension");
            
        }

        internal static void SameDim(in shortN a, in boolN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("shortN and boolN must have same dimension");
        }

        internal static void SameDim(in shortMxN a, in boolMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("shortMxN and boolMxN must have same dimension");
        }
    }
}