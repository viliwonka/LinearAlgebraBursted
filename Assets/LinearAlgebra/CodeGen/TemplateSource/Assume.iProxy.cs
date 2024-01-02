using Unity.Mathematics;
using System;

namespace LinearAlgebra
{
    internal static partial class Assume
    {
        internal static void SameDim(in iProxyN a, in iProxyN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("iProxyNs must have same dimension");
            
        }

        internal static void SameDim(in iProxyMxN a, in iProxyMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("iProxyMxNs must have same dimension");
            
        }

        internal static void SameDim(in iProxyN a, in boolN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("iProxyN and boolN must have same dimension");
        }

        internal static void SameDim(in iProxyMxN a, in boolMxN b)
        {
            if (a.M_Rows != b.M_Rows && a.N_Cols != b.N_Cols)
                throw new ArgumentException("iProxyMxN and boolMxN must have same dimension");
        }
    }
}