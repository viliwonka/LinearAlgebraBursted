# Optimize

Static class `Optimize` in namespace `BULA`.

## Nonlinear least squares

`nlsSolve` - Levenberg-Marquardt with Nielsen damping over a residual functor; optional robust
losses (Huber, Cauchy, Tukey, L1). `curveFit` fits a model's `y = f(x; p)` to `(xdata, ydata)`
through the same engine (numeric forward-difference Jacobian).

## Robust regression

`ladIRLS` - approximate L1 regression by iteratively reweighted least squares. For the exact
solution see `LP.lad` ([LP / LAD](lp-lad.md)).

## Scalar solvers

`bisection`, `newtonRoot`, `goldenSection`, `gradientDescent` over scalar functors.
