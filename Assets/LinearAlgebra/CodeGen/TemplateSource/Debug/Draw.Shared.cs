using UnityEngine;

namespace BULA
{
    // Type-agnostic members of Draw. `Draw` carries no per-type token, so every generated per-dtype
    // file partial-merges into this SAME class -- anything not parameterized by the element type must
    // live here once, or it collides as CS0111 across the merged fragments. Same arrangement as
    // Fit.Shared.cs and Hash.Shared.cs.
    //
    // Codegen hazard: this file must never contain the code generator's per-type placeholder
    // spellings, or it would be templated and reintroduce the collision it exists to prevent.
    public static partial class Draw
    {
        // A zero-alpha Color -- what `default` gives -- would draw nothing, so an unset colour becomes
        // white rather than silently invisible.
        internal static Color Resolve(Color c) => c.a > 0f ? c : Color.white;
    }
}
