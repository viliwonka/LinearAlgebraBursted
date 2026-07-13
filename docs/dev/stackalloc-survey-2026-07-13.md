# stackalloc survey — 2026-07-13

Scope: production templates under `Assets/LinearAlgebra/CodeGen/TemplateSource/**`.
Goal: identify (do NOT implement) small per-call `Allocator.Temp` allocations that a
`stackalloc` (or a hoist-once) could replace. Reason over ALL generated variants:
`fProxy` -> {float, double}; double DOUBLES every buffer's byte size, which is the gating
hazard for the stack-size checks below.

## Result summary

- TOP candidates: 1 site (BlockJacobi build loop — 3 per-block allocations).
- Marginal: 2 (QR/LQ `tcolBuf`, tiny but once-per-call).
- DO-NOT: 6 classes (QR/LQ `Tbuf` stack-size; LU/CHO/CHOP block*n unbounded; MIP growable
  UnsafeList(64); Arena.Sparse `blockT` unbounded-BR latent risk; all n/m solver scratch).
- Existing-stackalloc hygiene: 8 sites inventoried; 1 REGRESSION of the ILU0 loop-body class
  (ILU0 `BlockMulRight`), the rest correctly hoisted.

Note on the wrapper types: `Pivot`, `fProxyN`, `fProxyMxN` all wrap `UnsafeList`/records and
have NO stack-backed constructor (Pivot: only `(int size, Allocator)`; fProxyN: `(int,Alloc)`,
`(NativeArray view)`, `(in fProxyN,Alloc)`). So "pure stackalloc" is only reachable for code
paths that operate on raw `fProxy*`/`int*` (losing the wrapper indexers and the
ENABLE_UNITY_COLLECTIONS_CHECKS bounds guards). Where a candidate feeds a wrapper-typed API
(e.g. LU.decompInPlace(ref fProxyMxN, ref Pivot)), the realistic transform is HOIST-ONCE-HEAP,
not stackalloc.

## TOP candidates

### T1 — fProxyBlockJacobi build loop: 3 Temp allocations PER DIAGONAL BLOCK
`Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyBlockJacobi.cs:133,139,154`

Inside `for (int i = 0; i < BlockRows; i++)` (the ctor's diagonal-block inversion loop):
- L133 `var Dcopy = new fProxyMxN(BR, BR, Allocator.Temp, true);`  — BR*BR elems
- L139 `var P = new Pivot(BR, Allocator.Temp);`                    — BR ints
- L154 `var col = new fProxyN(BR, Allocator.Temp, true);`          — BR elems

Hotness: per BUILD, once per diagonal block. `BlockRows` = number of diagonal blocks of the
source BSR; for a large sparse SPD this is thousands. Each iteration does 3 allocate + 3 free
= 6 allocator round-trips, i.e. ~6*BlockRows Temp round-trips for one preconditioner build.

Why the size is effectively fixed: BR is the block dimension of the WHOLE BSR and is IDENTICAL
on every iteration (Dcopy/P/col are the same size each pass). So the allocations are pure
per-iteration churn — nothing about them varies across the loop.

Recommended transform (SAFE, no stackalloc needed): HOIST all three ABOVE the `for i` loop and
reuse them. Dcopy is refilled from A.Values each pass (L135-137); col is refilled per RHS
column (L157-158); P is (re)written by LU.decompInPlace each pass. Only correctness note: after
the singular-block early-return path (L142-150), the hoisted buffers must be disposed once at
the single exit rather than per-iteration — restructure the early return to `goto`/flag + single
cleanup, or keep the buffers in a try/finally-free block. Win: eliminates (BlockRows-1)*6
allocator round-trips per build with zero stack pressure and no BR bound required.

Pure-stackalloc alternative (only if a BR cap is acceptable): BlockJacobi has NO BR guard today
(unlike ILU0, which throws if BR>16). To stackalloc you would (a) add `if (BR>16) throw` to
match ILU0, AND (b) drop the LU wrapper API in favor of a raw-pointer block inverse
(Gauss-Jordan on `fProxy* m/inv = stackalloc fProxy[16*16]`, `int* perm = stackalloc int[16]`,
`fProxy* col = stackalloc fProxy[16]`) — literally the shape ILU0.InvertBlockInPlace already
uses. Bounds at BR=16: 16*16 doubles = 2KB per scratch, safe. This is a bigger change (new
contract + reimplemented numerics path: partial-pivot LU -> Gauss-Jordan) and is NOT required to
capture the win; hoist-once is strictly simpler and covers all BR. Recommend hoist-once; only go
raw-stackalloc if a BR<=16 contract is being added for other reasons.

Hazards: wrapper types have no stack ctor (see header note); the singular-block early return is
the one restructure to get right; z/r alias check in Apply is unrelated.

Which bench shows it: the BUILD cost (not Apply) is what changes. PCGBenchmark / the large-sparse
PCG path builds a BlockJacobi once before the CG loop — measure build time there with a
high-block-count matrix (e.g. the BSR gallery at BR in {2,3} and BlockRows in the thousands).
A focused micro-bench that times only the `new fProxyBlockJacobi(A, ...)` ctor over a
many-block matrix would show it most cleanly (existing PCG benches amortize build into the
solve and may hide it).

## Existing stackalloc inventory + hygiene (regression check of the ILU0 loop-body bug class)

HARD RULE being checked: `stackalloc` frees at METHOD RETURN, not loop-body exit; a `stackalloc`
lexically inside a for/while body accumulates one allocation per iteration on the method frame.

| Site | Size | Hoisted correctly? |
|------|------|--------------------|
| Sparse/fProxyILU0.cs:192 (`tmp`, BlockMulRight) | `fProxy[16]` | NO — inside `for r` loop. **REGRESSION.** |
| Sparse/fProxyILU0.cs:225,226,231 (`m`,`perm`,`inv`, InvertBlockInPlace) | `[16*16]`,`[16]`,`[16*16]` | yes (method top) |
| Sparse/fProxyILU0.cs:313 (`w`, Apply) | `fProxy[16]` | yes (above the nb loop) |
| Sparse/Arena.Sparse.fProxy.cs:140,181 (`blockT`) | `fProxy[BR*BC]` | yes (above loops) — but see DN4 |
| Sparse/UnsafeOP.Sparse.fProxy.cs:1370,1578 (`acc`, sweepLower/Upper) | `fProxy[BR]` | yes (above the row loop) |
| OP/Krylov.fProxy.cs:519,721,961,1230,1549,2029 (`ptrs` alias-check) | `long[6..9]` | yes (method top, before iter loop) |
| OP/Krylov.PBiCGStab.fProxy.cs:33 (`ptrs`) | `long[9]` | yes |
| OP/LOBPCG.fProxy.cs:710 (`ptrs`) | `long[23]` | yes |

### REGRESSION — ILU0.BlockMulRight `tmp` is stackalloc'd inside the row loop
`Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyILU0.cs:185-204`

```
static void BlockMulRight(... int BR)
{
    for (int r = 0; r < BR; r++)
    {
        unsafe
        {
            fProxy* tmp = stackalloc fProxy[16];   // <-- inside the loop body
            ...
        }
    }
}
```

This is the exact anti-pattern the recent fix warned about: `tmp` is re-`localloc`'d every
`r` iteration and only reclaimed at BlockMulRight's return, so the frame grows to BR*16 elems
within one call (BR<=16 by the ILU0 ctor guard, so worst case 256 elems = 2KB double — bounded,
not a stack-overflow, but pure waste and a rule violation). It does NOT accumulate across the
O(nnzb) BlockMulRight calls from FactorizeInPlace (each call is a separate frame that returns).
Severity: low (no incorrectness; wasted stack + violates the stated hoisting rule).
Fix direction: hoist `fProxy* tmp = stackalloc fProxy[16];` to the top of BlockMulRight, above
`for (int r ...)` — identical to how the sibling `w` (Apply) and `acc` (sweeps) are already
hoisted. Zero behavior change.

## Marginal

### M1 — QR/LQ `tcolBuf` (block-sized column scratch)
`OP/QR.fProxy.cs:432` (`new fProxyN(QR_BLOCK, ...)`, QR_BLOCK=32) and
`OP/LQ.fProxy.cs:380,452` (`new fProxyN(LQ_BLOCK, ...)`, LQ_BLOCK=64).

Bounded by a compile-time constant (32 / 64 elems -> <=512B even for double). Safe to stackalloc
on size. BUT allocated ONCE per decomposition call (above the panel loop, passed by ref into the
blocked core which reuses it across panels) — so the win is a single allocator round-trip against
an O(mn^2) decomposition. Negligible. Also feeds wrapper-typed core APIs (`ref fProxyN`), so a
stackalloc conversion needs a raw-pointer rework. Not worth it. Leave as-is.

## DO-NOT

- **DN1 — QR/LQ `Tbuf` (block*block).** `OP/QR.fProxy.cs:430` = QR_BLOCK*QR_BLOCK = 32*32 = 1024
  elems (float 4KB / **double 8KB**); `OP/LQ.fProxy.cs:378,450` = LQ_BLOCK*LQ_BLOCK = 64*64 =
  4096 elems (float 16KB / **double 32KB**). Both exceed the ~4KB job-thread stack budget for the
  double variant (LQ egregiously). Also once-per-call, so no real win. Keep on Allocator.Temp.
- **DN2 — LU/CHO/CHOP/LQ/QR panel buffers scaled by n or m.** e.g. QR `Wbuf`/`VfullBuf`
  (QR_BLOCK*n, m*n), LU `Ubuf` (LU_BLOCK*m), CHO `PT` (CHOL_BLOCK*n), CHOP `QT` (CHOLP_BLOCK*n),
  LQ `Vpanel`/`Y` (LQ_BLOCK*n, m*LQ_BLOCK). The block factor is constant but the other factor is
  the unbounded problem dimension -> not stack-bounded.
- **DN3 — MIP branch-and-bound containers.** `OP/MIP.fProxy.cs:266,267`
  `new UnsafeList<...>(64, Allocator.Temp)` — 64 is only an initial capacity; these GROW via Add
  during search, which reallocs and moves the buffer. stackalloc cannot grow and the pointer must
  not be retained across a resize. Keep on Temp.
- **DN4 — Arena.Sparse `blockT` (BR*BC).** `Sparse/Arena.Sparse.fProxy.cs:140,181` is already
  hoisted correctly, so not a regression — but note it is `stackalloc fProxy[BR*BC]` with NO BR/BC
  cap on the general BSR transpose/mirror path (only ILU0 caps BR<=16; general BSR does not). For
  a large block size (e.g. BR=BC=64 -> 4096 doubles = 32KB) this is a latent stack-overflow risk.
  Out of this survey's "reduce allocation" scope, but worth a defensive cap or a Temp fallback for
  large BR if arbitrary block sizes are ever exercised.
- **DN5 — All n/m-sized solver scratch** in Control/Kalman/UKF/QP/NLS/MPC/LP/SVD/Eigen/Bidiag,
  including the ones nested inside solver iteration loops (e.g. Control.fProxy.cs:148-152,199,
  209-219; QP.fProxy.cs:379-411,929-970 allocate n*n matrices per active-set/DARE iteration).
  These are genuine per-iteration heap churn but the size is the unbounded problem dimension —
  NOT stackalloc candidates. If revisited for perf, the fix is HOIST-ONCE-HEAP above the
  iteration loop, a separate refactor from this survey.
- **DN6 — Krylov O(n) inner vectors** (r, p, q, s, ...): O(n), unbounded, per spec not candidates.

## Verification notes
- ILU0 BR<=16 guard confirmed at fProxyILU0.cs:48-49 (`if (a.BR > 16) throw`).
- BlockJacobi has NO BR guard; general Apply loop handles arbitrary BR (dispatch unrolls only
  BR in {1,2,3,4,6}); DInv buffer is BlockRows*BR*BR from the passed allocator.
- IC0 and SSOR preconditioners use neither Allocator.Temp nor stackalloc in their build/apply
  (they operate directly on block storage) — nothing to convert there.
- Block constants: QR_BLOCK=32, LU_BLOCK=32, CHOL_BLOCK=32, CHOLP_BLOCK=32, LQ_BLOCK=64.
