using Unity.Mathematics;
using System;

namespace LinearAlgebra
{
    internal static partial class Assume
    {
        internal static void SameDim(in floatN a, in floatN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("floatNs must have same dimension");
            
        }

        internal static void SameDim(in floatMxN a, in floatMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("floatMxNs must have same dimension");
            
        }

        internal static void SameDim(in floatN a, in boolN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("floatN and boolN must have same dimension");
        }

        internal static void SameDim(in floatMxN a, in boolMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("floatMxN and boolMxN must have same dimension");
        }
    }
}