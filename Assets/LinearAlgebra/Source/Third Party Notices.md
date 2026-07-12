# Third Party Notices

This package contains algorithm implementations ported from or derived from the
following third-party projects.

## quantreg (R package) — relicensing permission pending

The exact LAD / quantile-regression solvers `LP.ladBR` (Barrodale-Roberts
specialized simplex) and `LP.ladFN` (Frisch-Newton interior point) are ports of
code by Roger Koenker and co-authors: `rqbr.f` (Koenker & d'Orey) and
`rq_fnm`/`lp_fnm` (Morillo, Koenker; MATLAB translation by Paul Eilers), as
distributed with the R quantreg package.

- Project: https://cran.r-project.org/package=quantreg
- Upstream license: GPL (>= 2)
- Status: permission to distribute these two derived implementations under this
  package's MIT license has been requested from the authors. Until that is
  resolved, this package must not be redistributed. (Precedent: the authors
  granted the same permission to QuantileRegressions.jl in 2015 and to
  quantreg-cpp.)

## HiGHS

The LP solvers (revised primal simplex, dual simplex), the QP active-set solver,
and the MIP branch-and-bound solver (including pseudocost/reliability branching,
domain propagation, and the randomized-rounding primal heuristic) are C# ports
derived from the HiGHS high-performance optimization software.

- Project: https://github.com/ERGO-Code/HiGHS
- License: MIT

```
MIT License

Copyright (c) 2026 HiGHS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
