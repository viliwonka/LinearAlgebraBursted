using System.Globalization;
using System.IO;
using System.Text;

namespace LinearAlgebra
{
    // Managed (allocating, NON-Burst) text / CSV exporters for int/short/long matrices and
    // vectors, mirroring Debug/Export.fProxy.cs. Integers have no round-trip precision concerns
    // (ToString is always exact); only the cast off the long proxy struct is required, via the
    // same per-type "choose" codegen marker used in Export.fProxy.cs. Unbounded StringBuilder --
    // unlike Print.Log's 4 KB FixedString, these never truncate. Managed / editor code only, never
    // from inside a Burst job. Codegen internals: see Debug/DEVLOG.md.
    //
        public static partial class Print
    {
        public static string ToText(in longMxN m)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < m.M_Rows; r++)
            {
                for (int c = 0; c < m.N_Cols; c++)
                {
                    if (c > 0) sb.Append(' ');
                    sb.Append(((long)m[r, c]).ToString(CultureInfo.InvariantCulture));
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static string ToText(in longN v)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < v.N; i++)
            {
                if (i > 0) sb.Append('\n');
                sb.Append(((long)v[i]).ToString(CultureInfo.InvariantCulture));
            }
            return sb.ToString();
        }

        public static string ToCsv(in longMxN m)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < m.M_Rows; r++)
            {
                for (int c = 0; c < m.N_Cols; c++)
                {
                    if (c > 0) sb.Append(',');
                    sb.Append(((long)m[r, c]).ToString(CultureInfo.InvariantCulture));
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static string ToCsv(in longN v)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < v.N; i++)
            {
                sb.Append(((long)v[i]).ToString(CultureInfo.InvariantCulture));
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static void SaveCsv(in longMxN m, string path) => File.WriteAllText(path, ToCsv(in m));
        public static void SaveCsv(in longN v, string path) => File.WriteAllText(path, ToCsv(in v));
    }
}
