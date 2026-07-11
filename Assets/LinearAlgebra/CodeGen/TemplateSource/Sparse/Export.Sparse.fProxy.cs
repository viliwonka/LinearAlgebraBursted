using System.Globalization;
using System.IO;
using System.Text;
using Unity.Collections;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    // Managed (allocating, NON-Burst) text / CSV export for the block-sparse fProxyBSR, mirroring
    // Debug/Export.fProxy.cs. Lives in namespace LinearAlgebra (not LinearAlgebra.Sparse) so it
    // merges into the same `Print` partial class as the dense exporters and the Sparse Debug
    // overloads -- fProxyBSR is brought in via the `using LinearAlgebra.Sparse;` above (same
    // pattern as Sparse/Debug.Sparse.fProxy.cs and Sparse/Arena.Sparse.fProxy.cs).
    public static partial class Print
    {
        /// <summary>
        /// Dense-ish preview: densifies via fProxyBSR.ToDense into a scratch Arena (allocated and
        /// disposed internally -- the caller does not need one of their own) and reuses the
        /// existing dense Print.ToText(in fProxyMxN). For a preview of the STORAGE itself (not the
        /// expanded dense matrix), see ToCsv(in fProxyBSR)/SaveCsv, which write a block-level
        /// coordinate/triplet list instead.
        /// </summary>
        public static string ToText(in fProxyBSR m)
        {
            var arena = new Arena(Allocator.Persistent);
            var dense = m.ToDense(ref arena);
            string text = ToText(in dense);
            arena.Dispose();
            return text;
        }

        /// <summary>
        /// Block-level coordinate/triplet CSV: one row per STORED block, "blockRow,blockCol,v0,v1,
        /// ...,v(BR*BC-1)" with the block's values flattened row-major (matching Values' own
        /// layout -- see fProxyBSR.cs). Avoids needing an Arena (unlike ToText's ToDense route) by
        /// reading RowPtr/ColInd/Values directly. For Symmetric matrices this reflects exactly
        /// what is stored (the lower block-triangle only), not a mirrored dense expansion.
        /// </summary>
        public static string ToCsv(in fProxyBSR m)
        {
            int blockLen = m.BR * m.BC;

            var sb = new StringBuilder();
            sb.Append("blockRow,blockCol");
            for (int i = 0; i < blockLen; i++)
            {
                sb.Append(",v");
                sb.Append(i.ToString(CultureInfo.InvariantCulture));
            }
            sb.Append('\n');

            for (int br = 0; br < m.BlockRows; br++)
            {
                int rowStart = m.RowPtr[br];
                int rowEnd = m.RowPtr[br + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bc = m.ColInd[k];
                    sb.Append(br.ToString(CultureInfo.InvariantCulture));
                    sb.Append(',');
                    sb.Append(bc.ToString(CultureInfo.InvariantCulture));

                    int blockOffset = k * blockLen;
                    for (int i = 0; i < blockLen; i++)
                    {
                        sb.Append(',');
                        sb.Append(((/*+choose[float|double]*/float/*-choose*/)m.Values[blockOffset + i]).ToString(/*+choose["G9"|"G17"]*/"G9"/*-choose*/, CultureInfo.InvariantCulture));
                    }
                    sb.Append('\n');
                }
            }

            return sb.ToString();
        }

        public static void SaveCsv(in fProxyBSR m, string path) => File.WriteAllText(path, ToCsv(in m));
    }
}
