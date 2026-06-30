namespace LinearAlgebra
{
    /// <summary>
    /// A scalar curve y = f(x) as a Burst struct-functor — the library's "lambda" (managed delegates
    /// can't run in jobs). Shared across subsystems: the optimizers (root-find / minimize) and the
    /// generators (<c>floatGen_OP.sample</c>, the <c>floatEasing</c> / <c>floatWave</c> functors).
    /// Implement it on a small struct holding only blittable fields.
    /// </summary>
    public interface IfloatScalarFunction {
        float Eval(float x);
    }
}
