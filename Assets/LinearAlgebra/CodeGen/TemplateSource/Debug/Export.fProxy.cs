using System.Globalization;
using System.IO;
using System.Text;

namespace LinearAlgebra
{
    // Managed (allocating, NON-Burst) text / CSV exporters for float/double matrices and vectors.
    //
    // Values are cast to the generated float/double type before formatting (the raw fProxy proxy only
    // has a parameterless ToString; float/double's ToString(format, IFormatProvider) is what gives the
    // round-trip and InvariantCulture guarantees). ToCsv uses "G9"/"G17" -- the minimal digit counts
    // .NET documents as round-tripping float/double exactly; ToText uses "G7" (human preview, not
    // required to round-trip).
    //
    // Unlike Print.Log -- which is Burst-callable but capped at a 4 KB FixedString and silently
    // truncates past it -- these build an unbounded System.Text.StringBuilder, so they never truncate.
    // Call them from managed / editor code only, never from inside a Burst job.
    public static partial class Print
    {
        public static string ToText(in fProxyMxN m)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < m.M_Rows; r++)
            {
                for (int c = 0; c < m.N_Cols; c++)
                {
                    if (c > 0) sb.Append(' ');
                    sb.Append(((/*+choose[float|double]*/float/*-choose*/)m[r, c]).ToString("G7", CultureInfo.InvariantCulture));
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static string ToText(in fProxyN v)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < v.N; i++)
            {
                if (i > 0) sb.Append('\n');
                sb.Append(((/*+choose[float|double]*/float/*-choose*/)v[i]).ToString("G7", CultureInfo.InvariantCulture));
            }
            return sb.ToString();
        }

        public static string ToCsv(in fProxyMxN m)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < m.M_Rows; r++)
            {
                for (int c = 0; c < m.N_Cols; c++)
                {
                    if (c > 0) sb.Append(',');
                    sb.Append(((/*+choose[float|double]*/float/*-choose*/)m[r, c]).ToString(/*+choose["G9"|"G17"]*/"G9"/*-choose*/, CultureInfo.InvariantCulture));
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static string ToCsv(in fProxyN v)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < v.N; i++)
            {
                sb.Append(((/*+choose[float|double]*/float/*-choose*/)v[i]).ToString(/*+choose["G9"|"G17"]*/"G9"/*-choose*/, CultureInfo.InvariantCulture));
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static void SaveCsv(in fProxyMxN m, string path) => File.WriteAllText(path, ToCsv(in m));
        public static void SaveCsv(in fProxyN v, string path) => File.WriteAllText(path, ToCsv(in v));
    }
}
