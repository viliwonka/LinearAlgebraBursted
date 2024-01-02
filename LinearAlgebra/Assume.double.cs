using Unity.Mathematics;
using System;

namespace LinearAlgebra
{
    internal static partial class Assume
    {
        internal static void SameDim(in doubleN a, in doubleN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("doubleNs must have same dimension");
            
        }

        internal static void SameDim(in doubleMxN a, in doubleMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("doubleMxNs must have same dimension");
            
        }

        internal static void SameDim(in doubleN a, in boolN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("doubleN and boolN must have same dimension");
        }

        internal static void SameDim(in doubleMxN a, in boolMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("doubleMxN and boolMxN must have same dimension");
        }
    }
}