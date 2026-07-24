# Quadratic & mixed-integer programming (`QP`, `MIP`)

Built on the same [`LP`](lp-lad.md) machinery (dual simplex for feasibility/re-solves), for problems
LP alone can't express: a quadratic objective, or integer variables.

## `QP` — convex quadratic programs

```
minimize    ½xᵀQx + cᵀx
subject to  Aᵢ·x {≤, =, ≥} bᵢ,   xl ≤ x ≤ xu
```

`QP.solve(in Q, in c, in A, in b, senses, in xl, in xu, ref x, out double objective[, maxIter])` —
`Q` must be symmetric (checked) and positive semidefinite (v1 contract, not checked — indefinite `Q`
is out of scope). Finds its own feasible starting point via an auxiliary LP, so `x` is output-only.
A box-free convenience overload drops `xl`/`xu` (unbounded variables). Dense null-space active-set
method (Nocedal & Wright ch. 16): one exact Newton step per equality-constrained sub-solve, with
add/drop logic over inequality/bound rows.

Returns `QPInfo`: `objective`, `iterations`, `status : QPStatus`
(`Optimal`/`Infeasible`/`Unbounded`/`MaxIterations`), plus `stationarityResidual`/
`feasibilityResidual` — cheap KKT diagnostics computed as a byproduct of the solve, not an extra
pass.

## `MIP` — mixed-integer programs

```
minimize    cᵀx
subject to  Aᵢ·x {≤, =, ≥} bᵢ,   xl ≤ x ≤ xu,   xⱼ ∈ ℤ for flagged j
```

`MIP.solve(in A, in b, in c, senses, in xl, in xu, in integrality, ref x, out double objective[,
maxNodes, maxIter, absGap, relGap])` — `integrality[j]` is `0` (continuous) or `1` (integer; a
binary variable is just an integer with `xl=0, xu=1`). Every integer variable needs a finite `xl`.

Branch-and-bound over the warm-started dense dual simplex: pseudocost + reliability branching picks
the branching variable, search is best-bound-with-plunging (dive one child, queue the sibling,
jump to the queue's best node at a leaf instead of backtracking), every node is domain-propagated to
a fixpoint before its LP solve, and a rounding heuristic tries to install an incumbent from each
fractional node. `absGap`/`relGap` stop early once the optimality gap is small enough
(`MIPStatus.GapLimit`) — useful when a proven-optimal solution costs far more nodes than a
provably-close one.

Returns `MIPInfo`: `objective`, `dualBound` (proven lower bound on the true optimum), `gap`,
`nodes`, `lpIterations`, `status : MIPStatus` (`Optimal`/`Infeasible`/`Unbounded`/`GapLimit`/
`NodeLimit`/`MaxIterations`) — on an early stop (`GapLimit`/`NodeLimit`/`MaxIterations`), `x`/
`objective` hold the best incumbent found so far, not a proven optimum.
