#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Collections;
using Unity.Mathematics;
using Unity.Burst;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    public static partial class Print {

        public static void Log(in floatN a, int start = 0, int end = -1)
        {
            FixedString4096Bytes str = new FixedString4096Bytes();

            int dim = a.N;
            
            if(end == -1)
                end = a.N;

            float normL2 = floatNormsOP.L2Range(a, start, end);

            FixedString128Bytes dimStr = $"Dim: {dim} \n";
            FixedString128Bytes normStr = $"L2: {normL2:G3} \n";

            str.Append(dimStr);
            str.Append(normStr);

            str.Append(startStr);
            for (int i = start; i < end; i++)
            {
                float element = a[i];
                FixedString128Bytes elementString;

                if (element >= 0)
                    str.Append('+');

                if (i == a.N - 1)
                    elementString = $"{element:G3}";
                else
                    elementString = $"{element:G3}, ";
                
                str.Append(elementString);
            }
            str.Append(endStr);
            str.Append('\n');


            UnityEngine.Debug.Log($"{str}");
        }

        public static void Log(in floatMxN m)
        {
            FixedString4096Bytes str = new FixedString4096Bytes();

            int dimRows = m.M_Rows;
            int dimCols = m.N_Cols;

            FixedString128Bytes dimStr = $"Dim | Rows:{dimRows} Cols:{dimCols} \n";
            //FixedString128Bytes normStr = $"L2: {normL2} \n";
            
            str.Append(dimStr);
            //str.Append(normStr);
            
            str.Append('\n');

            for (int r = 0; r < dimRows; r++)
            {
                str.Append('[');
                for (int c = 0; c < dimCols; c++)
                {
                    float element =  math.round(m[r, c]*10000f)/10000f;
                    if (element >= 0)
                        str.Append('+');
                    FixedString128Bytes elementString;

                    if (c == dimCols - 1)
                        elementString = $"{element:G5}";
                    else
                        elementString = $"{element:G5} |";
                    
                    str.Append(elementString);
                }
                str.Append(']');
                str.Append('\n');
            }


            UnityEngine.Debug.Log($"{str}");
        }

        public static void Spy(in floatMxN m) => Spy(m, 0.01f);

        public static void Spy(in floatMxN m, float absTreshold)
        {
            FixedString4096Bytes str = new FixedString4096Bytes();

            int dimRows = m.M_Rows;
            int dimCols = m.N_Cols;

            FixedString128Bytes printName = $"Matrix sparsity print\n";
            FixedString128Bytes dimStr = $"Dim | Rows:{dimRows} Cols:{dimCols} \n";
            
            str.Append(printName);
            str.Append(dimStr);
            
            str.Append('\n');

            for (int r = 0; r < dimRows; r++)
            {
                str.Append('[');
                for (int c = 0; c < dimCols; c++)
                {
                    float element = math.abs(m[r, c]);
                    if (element >= absTreshold)
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
