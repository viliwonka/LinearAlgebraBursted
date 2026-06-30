namespace LinearAlgebra
{
    /// <summary>
    /// A scalar curve y = f(x) as a Burst struct-functor — the library's "lambda" (managed delegates
    /// can't run in jobs). Shared across subsystems: the optimizers (root-find / minimize) and the
    /// generators (<c>doubleGen_OP.sample</c>, the <c>doubleEasing</c> / <c>doubleWave</c> functors).
    /// Implement it on a small struct holding only blittable fields.
    /// </summary>
    public interface IdoubleScalarFunction {
        double Eval(double x);
    }
}
