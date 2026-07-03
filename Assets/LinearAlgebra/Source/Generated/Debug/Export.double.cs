using System.Globalization;
using System.IO;
using System.Text;

namespace LinearAlgebra
{
    // Managed (allocating, NON-Burst) text / CSV exporters for float/double matrices and vectors.
    //
    // This used to be hand-authored per type (see git history) because the raw double PROXY struct
    // -- compiled directly by the TemplateSource-firstpass assembly, see CodeGen/TemplateSource/
    // proxyStructs.cs -- only exposes a parameterless ToString(); it is float/double's
    // ToString(string format, IFormatProvider) that gives the round-trip / InvariantCulture
    // guarantees these exporters need, and the proxy has no such overload. The fix: cast every
    // value to the ACTUAL generated type via the per-type "choose" codegen marker (see
    // GenUtils.cs; do NOT write the literal marker token in a comment -- the parser is
    // content-sensitive and would try to expand it) before formatting -- casts to (float) /
    // (double) at the proxy-compile
    // stage (double defines an implicit conversion to float, so the float choice is a direct
    // identity cast there) and the identity cast after codegen substitution (float -> float /
    // double -> double). Same trick selects "G9" (float) vs "G17" (double) for ToCsv -- the
    // minimal digit counts .NET documents as round-tripping each type exactly -- matching the
    // original hand-written precision choice.
    //
    // ToText uses "G7" for both types (human preview, not required to round-trip).
    //
    // Unlike Print.Log -- which is Burst-callable but capped at a 4 KB FixedString and SILENTLY
    // TRUNCATES past it -- these build an unbounded System.Text.StringBuilder, so they never
    // truncate. Call them from managed / editor code only, NEVER from inside a Burst job.
    public static partial class Print
    {
        public static string ToText(in doubleMxN m)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < m.M_Rows; r++)
            {
                for (int c = 0; c < m.N_Cols; c++)
                {
                    if (c > 0) sb.Append(' ');
                    sb.Append(((double)m[r, c]).ToString("G7", CultureInfo.InvariantCulture));
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static string ToText(in doubleN v)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < v.N; i++)
            {
                if (i > 0) sb.Append('\n');
                sb.Append(((double)v[i]).ToString("G7", CultureInfo.InvariantCulture));
            }
            return sb.ToString();
        }

        public static string ToCsv(in doubleMxN m)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < m.M_Rows; r++)
            {
                for (int c = 0; c < m.N_Cols; c++)
                {
                    if (c > 0) sb.Append(',');
                    sb.Append(((double)m[r, c]).ToString("G17", CultureInfo.InvariantCulture));
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static string ToCsv(in doubleN v)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < v.N; i++)
            {
                sb.Append(((double)v[i]).ToString("G17", CultureInfo.InvariantCulture));
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static void SaveCsv(in doubleMxN m, string path) => File.WriteAllText(path, ToCsv(in m));
        public static void SaveCsv(in doubleN v, string path) => File.WriteAllText(path, ToCsv(in v));
    }
}
