using Unity.Mathematics;
using System;

namespace LinearAlgebra
{
    internal static partial class Assume
    {
        internal static void SameDim(in fProxyN a, in fProxyN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("fProxyNs must have same dimension");
            
        }

        internal static void SameDim(in fProxyMxN a, in fProxyMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("fProxyMxNs must have same dimension");
            
        }

        internal static void SameDim(in fProxyN a, in boolN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("fProxyN and boolN must have same dimension");
        }

        internal static void SameDim(in fProxyMxN a, in boolMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("fProxyMxN and boolMxN must have same dimension");
        }
    }
}