namespace BULA
{
    /// <summary>
    /// A scalar curve y = f(x) as a Burst struct-functor — the library's "lambda" (managed delegates
    /// can't run in jobs). Shared across subsystems: the optimizers (root-find / minimize) and the
    /// generators (<c>Generate.sample</c>, the <c>fProxyEasing</c> / <c>fProxyWave</c> functors).
    /// Implement it on a small struct holding only blittable fields.
    /// </summary>
    public interface IfProxyScalarFunction {
        fProxy Eval(fProxy x);
    }
}
