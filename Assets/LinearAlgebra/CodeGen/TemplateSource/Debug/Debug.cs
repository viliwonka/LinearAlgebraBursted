#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Collections;
using Unity.Mathematics;
using Unity.Burst;

namespace LinearAlgebra
{
    [BurstCompile]
    public static partial class Print {

        static readonly FixedString128Bytes startStr = "[";
        static readonly FixedString128Bytes endStr = "]";

        [BurstCompile]
        public static void Log(in boolN a, int start = 0, int end = -1)
        {
            FixedString4096Bytes str = new FixedString4096Bytes();

            int dim = a.N;

            if (end == -1)
                end = a.N;

            FixedString128Bytes dimStr = $"BoolVec Dim: {dim} \n";

            str.Append(dimStr);
            
            str.Append(startStr);
            for (int i = start; i < end; i++)
            {
                bool element = a[i];
                
                if (element)
                    str.Append('T');
                else
                    str.Append('F');
            }
            str.Append(endStr);
            str.Append('\n');


            UnityEngine.Debug.Log($"{str}");
        }

        [BurstCompile]
        public static void Log(in boolMxN m, bool showTrue)
        {
            FixedString4096Bytes str = new FixedString4096Bytes();

            int dimRows = m.M_Rows;
            int dimCols = m.N_Cols;

            FixedString128Bytes printName = $"BoolMatrix sparsity print\n";
            FixedString128Bytes dimStr = $"Dim | Rows:{dimRows} Cols:{dimCols} \n";

            str.Append(printName);
            str.Append(dimStr);

            str.Append('\n');

            for (int r = 0; r < dimRows; r++)
            {
                str.Append('[');
                for (int c = 0; c < dimCols; c++)
                {
                    if (m[r, c] == showTrue)
                        str.Append('X');
                    else
                        str.Append(' ');
                }
                str.Append(']');
                str.Append('\n');
            }


            UnityEngine.Debug.Log($"{str}");
        }
    }
}
