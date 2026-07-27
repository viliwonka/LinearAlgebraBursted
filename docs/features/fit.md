# Fit — shape fitting

Static class `Fit` in namespace `BULA`. Fits shapes to point clouds under different metrics,
and samples points back from shapes.

## Shapes

Plane, line, circle, ellipse, sphere, cylinder, cone, torus, capsule, ellipsoid — in 2D and 3D
where each makes sense. Flats and spheres fit in closed form; the solids fit by orthogonal
distance via Levenberg-Marquardt, warm-startable through the incoming axis.

## Metrics & solvers

- `Fit.<shape>(points, …)` — least squares, or any robust loss (Huber, Cauchy, Tukey, L1) via
  the loss-functor overloads. Robust losses handle a few mild outliers.
- `Fit.ransac` / `ransacLo` / `magsac` — consensus fitting for gross contamination; composes
  with any shape model. Find the inliers with RANSAC, then polish with a loss.
- `Fit.nls<TModel>` — Levenberg-Marquardt over any parametric shape.
- Algebraic fits: `Fit.conic` (ellipse-constrained), `Fit.quadric` + `Fit.classify`,
  `Fit.ellipsoid` (ellipsoid-constrained).
- `Fit.linear` (vertical residual) vs `Fit.total` (orthogonal, errors-in-variables).

## Sampling

`Fit.sample` draws uniformly from a shape's own surface — by area, or by arc length for a
curve. Available for shapes of finite measure.
