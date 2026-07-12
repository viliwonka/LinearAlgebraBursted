# Release scan 2026-07-12 — area: qr-lq

Scanned 9 template files (core). Findings: total 4 — confirmed 4, uncertain 0, unverified 0, refuted 0; high 0, medium 0, low 4.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QR.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QR.Workspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QRCP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QRCP.Workspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LQ.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LQ.Workspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LQ.MinNormWorkspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LQRP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LQRP.Workspace.fProxy.cs

## Findings

### 1. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LQ.fProxy.cs:188 — Comment claims LQ_BLOCK 'Matches QR_BLOCK' but the two constants differ (64 vs 32).

**Evidence**

```
// files, so a class-level const of the same name would collide (CS0102). Matches QR_BLOCK.
const int LQ_BLOCK = 64;  // QR.fProxy.cs uses `const int QR_BLOCK = 32;` everywhere
```

The comment asserts parity with QR_BLOCK, but QR.fProxy.cs uses `const int QR_BLOCK = 32;` everywhere while LQ_BLOCK is 64.

**Verifier**

LQ.fProxy.cs:188-189 declares `const int LQ_BLOCK = 64;` with a trailing comment "Matches QR_BLOCK." Verified via grep in QR.fProxy.cs — every declaration/usage of QR_BLOCK (lines 228, 319, 410, 460) uses value 32, not 64. The comment's claim of parity is literally false; suggested fix (drop the clause or state the actual value/rationale) is appropriate. Low-severity, purely a comment/naming defect with no runtime impact.

**Suggested fix**

Drop the 'Matches QR_BLOCK' clause (or state the actual value / rationale). QR_BLOCK is 32; LQ_BLOCK is 64, so they do not match.

### 2. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LQRP.fProxy.cs:326 — XML doc asserts QRCP.decompInPlace is 'allocation-free', but that method allocates vn1/vn2 (2n) plus w (n) from Allocator.Temp.

**Evidence**

```
/// Unlike QRCP.decompInPlace (allocation-free), this internally allocates one m × n Allocator.Temp
```

But QRCP.decompInPlace does allocate: `var vn1 = new fProxyN(n, Allocator.Temp,...); var vn2 = ...` and decompInPlaceCore allocs `w`.

**Verifier**

The XML doc at LQRP.fProxy.cs:326 explicitly cref-targets the 4-arg QRCP.decompInPlace overload and calls it "allocation-free". That overload (QRCP.fProxy.cs:47-69) allocates two length-n Allocator.Temp vectors (vn1, vn2) at lines 63-64, and its callee decompInPlaceCore allocates a third length-n Allocator.Temp vector w at line 141. So the referenced overload allocates 3n scratch scalars from Allocator.Temp — the "allocation-free" label is factually wrong. The suggested rewording (only O(n) scratch vs an additional m x n copy) captures the actual, intended contrast.

**Suggested fix**

Reword to the intended contrast: QRCP.decompInPlace needs no full m×n matrix copy (only O(n) scratch), whereas LQRP.decompInPlace additionally copies A to an m×n buffer. Do not call QRCP.decompInPlace 'allocation-free'.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QR.fProxy.cs:117 — Exception message says 'Matrix R must be square or tall' but the validated shape is the input matrix A_to_Q, not R (R is always n×n square).

**Evidence**

```
if (A_to_Q.M_Rows < A_to_Q.N_Cols)
    throw new ArgumentException("QR.decompInPlace: Matrix R must be square or tall (more or equal rows than cols)");
```

The check validates the shape of the input matrix A_to_Q, but the message refers to R, which is always n×n square.

**Verifier**

All three decompInPlace overloads (lines 117, 413, 463) validate A_to_Q.M_Rows < A_to_Q.N_Cols — the input's shape — but throw with a message referring to "Matrix R". The XML doc on the decomp wrapper (line 482) explicitly documents R as "N_Cols x N_Cols", i.e. always square, so the "or tall" wording cannot describe R at all; it describes the A input. Genuine (low-severity) contradiction between exception text and the property actually being checked. Fix is a message rewording in the template only; no numerical behavior is affected.

**Suggested fix**

Change 'Matrix R' to 'Matrix A' / 'A_to_Q' in the three decompInPlace overloads (lines 117, 413, 463). The tall/square constraint is on the input, not R.

### 4. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QR.fProxy.cs:277 — Code comment carries a benchmark verdict (throughput number) which the project comment policy requires to live in DEVLOG, not in code.

**Evidence**

```
// GEMM call per panel — UnsafeOP.wyVtC/wySubVW already reach full GEMM
// throughput (~70 GFLOP/s, matched matMatDot) at this width without tiling.
```

A concrete throughput number and rejected-alternative note in a code comment, which the CLAUDE.md comment policy routes to DEVLOG.md.

**Verifier**

Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QR.fProxy.cs lines 275-277 verbatim contain "reach full GEMM throughput (~70 GFLOP/s, matched matMatDot) at this width without tiling." This is a concrete throughput number plus a rejected-alternative note (tiling considered and rejected as unnecessary at this width). CLAUDE.md's strict comment policy names "benchmark results, perf verdicts, rejected alternatives" as content that must live in DEVLOG.md and never in code comments. The contract half ("One untiled GEMM call per panel") is fine; the perf-verdict tail is a genuine policy violation. Low severity, as claimed — cosmetic, no runtime impact.

**Suggested fix**

Move the '~70 GFLOP/s, matched matMatDot' benchmark verdict to the folder DEVLOG.md; keep only the contract (one GEMM flush per panel).

## Scanner notes

Scope covered in full: QR.fProxy.cs, QR.Workspace.fProxy.cs, QRCP.fProxy.cs (1826 lines), QRCP.Workspace.fProxy.cs, LQ.fProxy.cs, LQ.Workspace.fProxy.cs, LQ.MinNormWorkspace.fProxy.cs, LQRP.fProxy.cs (1299 lines), LQRP.Workspace.fProxy.cs. Also read UnsafeOP.formT to verify blocked-core scratch sizing.

Correctness checks that PASSED (no defect found): Householder construction (||v||^2=2, sign-safe), compact-WY T/Tᵀ direction (QR factor=Tᵀ/recon=T; LQ flipped factor=T/recon=Tᵀ), norm-downdating guard matches LAPACK dlaqps exactly ((1+r)(1-r) form, tol3z=sqrt(eps), decay-since-exact via vn2), rank detection from non-increasing R/L diagonal (NaN/zero R[0,0] -> rank 0), un-permute scatter, blocked panel column/row restriction invariants, and F-row swap on pivot. Buffer sizing: formT G-scratch (pb*pb) fits Y=m*LQ_BLOCK because blocked LQ is gated at m>=256; QR Wbuf=32*n covers pb*cw and pb*pb. Disposal verified on all paths incl. the fusedSolve branch in QRCP.decompInPlaceBlockedCore (Vpanel/Tbuf/tcolBuf/VfullBuf only allocated and only disposed when !fusedSolve; F/acc/mark/Wbuf always both). Validate-before-alloc is consistently applied so caller-error throws cannot leak Temp.

Borderline comment-policy items not separately reported to avoid over-reporting: LQ.fProxy.cs:336-343 (the LQ_BLOCK_MIN_M rationale contains measured-crossover / 'err HIGH on purpose' tuning verdicts that arguably belong in DEVLOG); these read more as gate contracts than pure history, so I left them out of the findings list.
