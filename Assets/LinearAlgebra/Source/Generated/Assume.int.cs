using Unity.Mathematics;
using System;

namespace LinearAlgebra
{
    internal static partial class Assume
    {
        internal static void SameDim(in intN a, in intN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("intNs must have same dimension");
            
        }

        internal static void SameDim(in intMxN a, in intMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("intMxNs must have same dimension");
            
        }

        internal static void SameDim(in intN a, in boolN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("intN and boolN must have same dimension");
        }

        internal static void SameDim(in intMxN a, in boolMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("intMxN and boolMxN must have same dimension");
        }
    }
}