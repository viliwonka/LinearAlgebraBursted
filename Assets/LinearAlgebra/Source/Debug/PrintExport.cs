using System.Globalization;
using System.IO;
using System.Text;

namespace LinearAlgebra
{
    // Managed (allocating, NON-Burst) text / CSV exporters for matrices and vectors.
    //
    // Unlike Print.Log — which is Burst-callable but capped at a 4 KB FixedString and SILENTLY
    // TRUNCATES past it — these build an unbounded System.Text.StringBuilder, so they never
    // truncate. Use them for large data and as the bridge to Python / Excel / MATLAB for real
    // plotting. They allocate managed garbage and use System.IO, so call them from managed / editor
    // code, NOT from inside a Burst job.
    //
    // This file is hand-authored (not codegen'd): formatted ToString(format, InvariantCulture)
    // requires the concrete float / double types, and the template's proxy type only exposes a
    // parameterless ToString(). float uses G9 (round-trips a float); double uses G17 (round-trips a
    // double); both pin InvariantCulture so '.' is the decimal separator on every locale.
    public static partial class Print
    {
        // ---------- float ----------

        public static string ToText(in floatMxN m)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < m.M_Rows; r++)
            {
                for (int c = 0; c < m.N_Cols; c++)
                {
                    if (c > 0) sb.Append(' ');
                    sb.Append(m[r, c].ToString("G7", CultureInfo.InvariantCulture));
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static string ToText(in floatN v)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < v.N; i++)
            {
                if (i > 0) sb.Append('\n');
                sb.Append(v[i].ToString("G7", CultureInfo.InvariantCulture));
            }
            return sb.ToString();
        }

        public static string ToCsv(in floatMxN m)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < m.M_Rows; r++)
            {
                for (int c = 0; c < m.N_Cols; c++)
                {
                    if (c > 0) sb.Append(',');
                    sb.Append(m[r, c].ToString("G9", CultureInfo.InvariantCulture));
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static string ToCsv(in floatN v)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < v.N; i++)
            {
                sb.Append(v[i].ToString("G9", CultureInfo.InvariantCulture));
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static void SaveCsv(in floatMxN m, string path) => File.WriteAllText(path, ToCsv(in m));
        public static void SaveCsv(in floatN v, string path) => File.WriteAllText(path, ToCsv(in v));

        // ---------- double ----------

        public static string ToText(in doubleMxN m)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < m.M_Rows; r++)
            {
                for (int c = 0; c < m.N_Cols; c++)
                {
                    if (c > 0) sb.Append(' ');
                    sb.Append(m[r, c].ToString("G7", CultureInfo.InvariantCulture));
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
                sb.Append(v[i].ToString("G7", CultureInfo.InvariantCulture));
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
                    sb.Append(m[r, c].ToString("G17", CultureInfo.InvariantCulture));
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
                sb.Append(v[i].ToString("G17", CultureInfo.InvariantCulture));
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static void SaveCsv(in doubleMxN m, string path) => File.WriteAllText(path, ToCsv(in m));
        public static void SaveCsv(in doubleN v, string path) => File.WriteAllText(path, ToCsv(in v));
    }
}
