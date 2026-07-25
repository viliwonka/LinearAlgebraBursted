using Unity.Collections;
using Unity.Mathematics;

//alsoExpand[uint]// int-family console dumps; widening to `long` below is a lossless implicit
//conversion for uint too (as it already is for int/short), so no signed-only ops here.

namespace BULA
{
    // Integer vector / matrix console dumps, mirroring the fProxy Print.Log overloads (which only
    // covered the float types). Burst-callable, capped at a 4 KB FixedString like the rest of Print;
    // for unbounded output use the managed Print.ToText / Print.ToCsv helpers.
    public static partial class Print {

        public static void Log(in iProxyN a, int start = 0, int end = -1)
        {
            FixedString4096Bytes str = new FixedString4096Bytes();

            int dim = a.N;

            if (end == -1)
                end = a.N;

            FixedString128Bytes dimStr = $"Dim: {dim} \n";
            str.Append(dimStr);

            str.Append(startStr);
            for (int i = start; i < end; i++)
            {
                // widen short/int/long to long so the FixedString interpolation compiles for every iProxy type
                long element = a[i];
                FixedString128Bytes elementString;

                if (i == a.N - 1)
                    elementString = $"{element}";
                else
                    elementString = $"{element}, ";

                str.Append(elementString);
            }
            str.Append(endStr);
            str.Append('\n');

            UnityEngine.Debug.Log($"{str}");
        }

        public static void Log(in iProxyMxN m)
        {
            FixedString4096Bytes str = new FixedString4096Bytes();

            int dimRows = m.M_Rows;
            int dimCols = m.N_Cols;

            FixedString128Bytes dimStr = $"Dim | Rows:{dimRows} Cols:{dimCols} \n";
            str.Append(dimStr);
            str.Append('\n');

            for (int r = 0; r < dimRows; r++)
            {
                str.Append('[');
                for (int c = 0; c < dimCols; c++)
                {
                    long element = m[r, c];
                    FixedString128Bytes elementString;

                    if (c == dimCols - 1)
                        elementString = $"{element}";
                    else
                        elementString = $"{element} |";

                    str.Append(elementString);
                }
                str.Append(']');
                str.Append('\n');
            }

            UnityEngine.Debug.Log($"{str}");
        }
    }
}
