using System.Globalization;
using System.IO;
using System.Text;

namespace LinearAlgebra
{
    // Managed (allocating, NON-Burst) text / CSV exporters for int/short/long matrices and
    // vectors, mirroring Debug/Export.fProxy.cs. Integers have no round-trip precision concerns
    // (ToString is always exact), so there is no "G9 vs G17"-style per-type format choice needed --
    // only the cast off the int PROXY struct (which -- like fProxy, see proxyStructs.cs -- only
    // exposes a parameterless ToString()) is required, via the same per-type "choose" codegen
    // marker (see GenUtils.cs) used in Export.fProxy.cs: casts to (int) / (short) / (long)
    // at the proxy-compile stage (int defines an implicit conversion to int, so the int choice
    // is a direct identity cast there) and an identity cast after codegen substitution.
    // (Do NOT write the literal choose-marker token in this comment -- the codegen parser is
    //  content-sensitive and would try to expand it.)
    //
    // Unlike Print.Log -- which is Burst-callable but capped at a 4 KB FixedString and SILENTLY
    // TRUNCATES past it -- these build an unbounded System.Text.StringBuilder, so they never
    // truncate. Call them from managed / editor code only, NEVER from inside a Burst job.
    //
        public static partial class Print
    {
        public static string ToText(in intMxN m)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < m.M_Rows; r++)
            {
                for (int c = 0; c < m.N_Cols; c++)
                {
                    if (c > 0) sb.Append(' ');
                    sb.Append(((int)m[r, c]).ToString(CultureInfo.InvariantCulture));
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static string ToText(in intN v)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < v.N; i++)
            {
                if (i > 0) sb.Append('\n');
                sb.Append(((int)v[i]).ToString(CultureInfo.InvariantCulture));
            }
            return sb.ToString();
        }

        public static string ToCsv(in intMxN m)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < m.M_Rows; r++)
            {
                for (int c = 0; c < m.N_Cols; c++)
                {
                    if (c > 0) sb.Append(',');
                    sb.Append(((int)m[r, c]).ToString(CultureInfo.InvariantCulture));
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static string ToCsv(in intN v)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < v.N; i++)
            {
                sb.Append(((int)v[i]).ToString(CultureInfo.InvariantCulture));
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static void SaveCsv(in intMxN m, string path) => File.WriteAllText(path, ToCsv(in m));
        public static void SaveCsv(in intN v, string path) => File.WriteAllText(path, ToCsv(in v));
    }
}
