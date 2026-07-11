# Research: open-source LP/QP solvers — what to borrow, what to build

Status: RESEARCH NOTES (2026-07-08). Question: are there good free/open-source LP solvers we could borrow
code from, and should we add more solver classes (QP, conic)? Short answer: **don't literally borrow
code** (can't link C++/Rust/Fortran into Burst, and license friction) — instead **port algorithms** from
the permissively-licensed references, and the highest-value additions are **first-order matrix-free
methods**, because they are exactly what this library's BSR + Krylov + deterministic-Burst stack is built
for.

## 1. The "borrowing" reality check

Every serious open-source LP/QP solver is C, C++, Rust, Julia, or Fortran. **None can be linked into a
Unity Burst job** — Burst compiles a restricted C# subset to native, with no P/Invoke into external native
libs inside a job, no managed heap, blittable-only. So "borrow code" always means **reimplement the
algorithm in templated `fProxy` C#**, not copy source. That makes **license matter for *reading and
porting*** (permissive = safe to study and mirror closely; copyleft = risky to mirror):

| Solver | License | Safe to port from? |
|---|---|---|
| **HiGHS** | **MIT** | ✅ yes — cleanest permissive reference for simplex + IPM + presolve + crossover |
| **OSQP** | **Apache-2.0** | ✅ yes |
| **SCS** | **MIT** | ✅ yes |
| **Clarabel** | **Apache-2.0** | ✅ yes — modern IPM for QP/conic |
| **PDLP** (in OR-Tools) | **Apache-2.0** | ✅ yes |
| CLP / COIN-OR | EPL-2.0 (weak, file-level copyleft) | ⚠️ usable but not as clean as MIT/Apache |
| **GLPK** | **GPL** | ❌ avoid — copyleft, contaminates a permissive release |
| **SoPlex** | ZIB academic | ❌ avoid — not OSS-permissive for commercial use |
| lp_solve | LGPL | ⚠️ avoid for a header-style port |

Given this library is heading for a **permissive public release**, restrict inspiration to the MIT/Apache
set (HiGHS, OSQP, SCS, Clarabel, PDLP).

## 2. Two families, and which one fits us

**(a) Factorization-based: simplex + interior point.** HiGHS, CLP, GLPK, SoPlex, Clarabel. These need a
sparse LU basis (simplex) or a sparse Cholesky/LDLᵀ of the KKT system (IPM). We deliberately made both of
those **non-goals** — sparse revised simplex is a large separate build, and a sparse direct Cholesky is
its own feature. Our dense simplex + dense/sparse matrix-free IPM already cover this family at the level
we want. Porting more here means porting a sparse factorization, which is the exact thing we chose not to
do.

**(b) First-order / matrix-free: PDHG (LP) and ADMM (QP/conic).** PDLP, OSQP, SCS. These **never
factorize** — every iteration is matrix-vector products `A·v`, `Aᵀ·v` plus cheap projections/prox steps.
That is *precisely* this library's strength: `IfProxyLinearOperator` over BSR, the whole `Krylov` stack,
and deterministic single-thread Burst. **This is the family worth adding**, and it is nearly free of the
ill-conditioning that is currently making our IPM slow (§ ties into `research-lp-preconditioners.md`).

## 3. The three additions worth building (ranked)

### 3.1 PDLP-style restarted PDHG for large-scale LP  ★ best fit
Restarted Primal-Dual Hybrid Gradient (PDLP, Google OR-Tools, Apache-2.0). Solves `min cᵀx s.t. Ax {≤,=}
b, l ≤ x ≤ u` with **only** `A·v` / `Aᵀ·v`, diagonal step sizes, and periodic restart-to-average. No
normal equations, **no preconditioner problem at all** — it sidesteps the entire ill-conditioned
`AᵀDA` solve that our IPM struggles with. Deterministic, GPU/SIMD-friendly, scales to enormous sparse
LPs. It complements (does not replace) the IPM: IPM = few iterations, high accuracy, moderate size;
PDHG = very cheap iterations, matrix-free, huge scale, medium accuracy. Reuses our BSR operator and
Ruiz-equilibration (once built) directly. Modern refinement: **restarted Halpern PDHG (rHPDHG)** —
better complexity, still matrix-free. **Highest-value new solver for this codebase.**

### 3.2 OSQP-style ADMM for Quadratic Programming  ★ adds QP, reuses Krylov
OSQP (Apache-2.0) solves `min ½xᵀPx + qᵀx s.t. l ≤ Ax ≤ u` by ADMM. Each iteration solves ONE linear
system `(P + σI + ρ AᵀA) x = rhs` with a **fixed** matrix (P, A, ρ constant across iterations — unlike
IPM's changing `D`). Two routes, both of which we already have the pieces for:
- **Direct**: one Cholesky at setup, reused every iteration (our dense `CHO`).
- **Indirect (matrix-free)**: solve that system with **CG over a BSR operator** — literally our
  `Krylov.pcg` + a `P + σI + ρAᵀA` operator (composable like `fProxyNormalOperator`). Because the system
  matrix is *constant*, warm-started CG converges in very few iterations — none of the boundary
  ill-conditioning that plagues IPM.
This is the natural way to give the library **quadratic programming** (the user's ask) with minimal new
machinery — it is mostly the ADMM outer loop + projections wrapped around solvers we already ship.

### 3.3 Dense factorization upgrade: HiGHS-style dual simplex + presolve  ★ "dense needs love"
Matrix-free first-order methods do nothing for DENSE problems (forming/factoring is already cheap there),
so the way to make dense LP faster is to stay in the factorization family and port from **HiGHS (MIT)**:
- **Dual simplex** — usually faster and more numerically robust than our current primal two-phase simplex,
  and warm-starts well. The standard modern default.
- **Presolve** — redundant row/column removal, singleton handling, bound tightening, scaling. Shrinks the
  problem before it ever reaches the solver; benefits every backend (simplex AND interior point).
- **Crossover** — recover an exact vertex from an interior-point solution (for problems that need a basic
  optimal solution). Optional.
This is the concrete answer to "dense matrices need love": keep dense on factorization (the right tool),
and raise it to HiGHS-level with a dual simplex + presolve. Independent of, and complementary to, the
first-order sparse work in §3.1–§3.2.

### 3.4 (DEFERRED) SCS-style conic ADMM for SOCP/SDP
SCS (MIT) applies ADMM to the homogeneous self-dual embedding, with a direct or CG-indirect linear solve
and cone projections (orthant, second-order, PSD — the PSD projection would reuse our `Eigen`). Extends to
second-order-cone and semidefinite programming. **Deferred** by decision — bigger scope, beyond current
need. Recorded as a future direction only.

## 4. Recommendation

- **Don't borrow source.** Port algorithms from the MIT/Apache references (HiGHS for simplex/IPM
  robustness details like presolve + crossover + starting points; OSQP/PDLP/SCS for the first-order
  methods).
- **Build first-order matrix-free solvers** (for SPARSE/large) — the missing family, best fit to our
  BSR+Krylov+Burst design:
  1. **PDLP restarted PDHG** for scalable matrix-free LP (sidesteps the preconditioner problem entirely).
  2. **OSQP ADMM for QP** — delivers quadratic programming, indirect mode = reuse `Krylov.pcg` + a
     composed operator; direct mode = reuse `CHO`.
- **Upgrade the DENSE factorization path** (dense needs love, and matrix-free doesn't help it):
  3. **HiGHS-style dual simplex + presolve** (+ optional crossover) — the right way to make dense LP
     faster; stays in the factorization family where dense belongs.
- **Deferred**: SCS conic ADMM (SOCP/SDP) — future direction only.
- **Keep as non-goals**: sparse revised simplex, sparse direct Cholesky, exact-rational LP (SoPlex
  territory) — large separate efforts against our matrix-free grain.

The throughline: DENSE → factorization (simplex/IPM, keep improving via HiGHS ideas); SPARSE/large →
matrix-free (`A·v` / `Aᵀ·v`) first-order + IPM, our comparative advantage.

## Sources

- [Comparison of open-source LP solvers (technical report, OSTI)](https://www.osti.gov/biblio/1104761) · [HiGHS advanced LP solving (Google OR-Tools docs)](https://developers.google.com/optimization/lp/lp_advanced)
- [OSQP: An Operator Splitting Solver for Quadratic Programs (arXiv:1711.08013)](https://arxiv.org/pdf/1711.08013) · [OSQP docs](https://osqp.org/docs/)
- [SCS — Splitting Conic Solver (referenced via qpsolvers)](https://github.com/qpsolvers/qpsolvers) · [Clarabel: interior-point solver for conic programs (arXiv:2405.12762)](https://arxiv.org/pdf/2405.12762)
- [PDLP: A Practical First-Order Method for Large-Scale Linear Programming (arXiv:2501.07018)](https://arxiv.org/pdf/2501.07018) · [Restarted Halpern PDHG for Linear Programming (arXiv:2407.16144)](https://arxiv.org/pdf/2407.16144)
- [An Overview of GPU-based First-Order Methods for LP (arXiv:2506.02174)](https://arxiv.org/pdf/2506.02174)
