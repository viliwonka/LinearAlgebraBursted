using System.Globalization;
using System.IO;
using System.Text;

namespace BULA
{
    // Managed (allocating, NON-Burst) text / CSV exporters for int/short/long matrices and
    // vectors, mirroring Debug/Export.fProxy.cs. Integers have no round-trip precision concerns
    // (ToString is always exact); only the cast off the iProxy proxy struct is required, via the
    // same per-type "choose" codegen marker used in Export.fProxy.cs. Unbounded StringBuilder --
    // unlike Print.Log's 4 KB FixedString, these never truncate. Managed / editor code only, never
    // from inside a Burst job. Codegen internals: see Debug/DEVLOG.md.
    //
    //alsoExpand[uint]// the per-type choose lists below carry a 4th (uint) value; uint's cast is
    //likewise a direct identity (iProxy defines an implicit conversion to uint there too).
    public static partial class Print
    {
        public static string ToText(in iProxyMxN m)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < m.M_Rows; r++)
            {
                for (int c = 0; c < m.N_Cols; c++)
                {
                    if (c > 0) sb.Append(' ');
                    sb.Append(((/*+choose[int|short|long|uint]*/int/*-choose*/)m[r, c]).ToString(CultureInfo.InvariantCulture));
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static string ToText(in iProxyN v)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < v.N; i++)
            {
                if (i > 0) sb.Append('\n');
                sb.Append(((/*+choose[int|short|long|uint]*/int/*-choose*/)v[i]).ToString(CultureInfo.InvariantCulture));
            }
            return sb.ToString();
        }

        public static string ToCsv(in iProxyMxN m)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < m.M_Rows; r++)
            {
                for (int c = 0; c < m.N_Cols; c++)
                {
                    if (c > 0) sb.Append(',');
                    sb.Append(((/*+choose[int|short|long|uint]*/int/*-choose*/)m[r, c]).ToString(CultureInfo.InvariantCulture));
                }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static string ToCsv(in iProxyN v)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < v.N; i++)
            {
                sb.Append(((/*+choose[int|short|long|uint]*/int/*-choose*/)v[i]).ToString(CultureInfo.InvariantCulture));
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static void SaveCsv(in iProxyMxN m, string path) => File.WriteAllText(path, ToCsv(in m));
        public static void SaveCsv(in iProxyN v, string path) => File.WriteAllText(path, ToCsv(in v));
    }
}
