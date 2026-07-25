namespace BULA
{
    // Type-agnostic members of the Fit facade. `Fit` carries no per-type token, so every generated
    // per-dtype file partial-merges into this SAME class -- anything not parameterized by the element
    // type must live here once, or it collides as CS0102/CS0111 across the merged fragments. Same
    // arrangement as Hash.Shared.cs.
    //
    // Codegen hazard: this file must never contain the code generator's per-type placeholder
    // spellings, or it would be templated and reintroduce the collision it exists to prevent.
    public static partial class Fit
    {
        /// <summary>IRLS iteration budget used when a caller passes maxIter &lt;= 0.</summary>
        public const int DefaultIrlsIter = 50;
    }
}
