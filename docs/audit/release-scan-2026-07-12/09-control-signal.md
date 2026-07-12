# Release scan 2026-07-12 — area: control-signal

Scanned 10 template files (core). Findings: total 2 — 2 confirmed, 0 uncertain, 0 unverified, 0 refuted; severity: 0 high, 0 medium, 2 low.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Control.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Control.Info.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/FFT.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/FFT.Workspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Wave.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Easing.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/ResampleOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/ResampleEnums.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/WindowType.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Realtime/RollingWindow.fProxy.cs

## Findings

### 1. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/FFT.Workspace.fProxy.cs:21 — Code comment carries a performance/memory verdict (contracts-only policy violation).

**Evidence**

```
"Bandwidth tradeoff: full table uses ~2× twiddle memory (~8 MB at N=1M for float), offset by halving the number of full-array passes (log4(N) vs log2(N) passes)."
```

This is a perf tradeoff verdict with benchmark-style memory figures embedded in an XML doc, which CLAUDE.md restricts to DEVLOG.md (comments state contracts only).

**Verifier**

Lines 21-22 of the XML doc contain a bandwidth tradeoff verdict with a concrete benchmark-style memory figure ("~8 MB at N=1M for float") and pass-count rationale — not a contract. CLAUDE.md explicitly restricts code comments/XML docs to contracts only and routes perf/memory verdicts to DEVLOG.md. The preceding contract sentence about radix-4 twiddles reaching index 3n/4 is fine to keep; the "Bandwidth tradeoff…" sentence should move to the OP/DEVLOG.md. (Lines 366-370 carry the same tradeoff rationale as an internal comment banner, reinforcing the leak; the claim only flags the XML-doc instance.)

**Suggested fix**

Keep the contract sentence ("full-circle table required by radix-4 paths") and move the ~8MB/pass-count tradeoff rationale into the folder DEVLOG.md.

### 2. [low/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/FFT.Workspace.fProxy.cs:203 — rfft/irfft table overloads validate with RequireFftWorkspace, which does not check the full-circle twReFull/twImFull lengths their inner radix-4 cores index.

**Evidence**

```
rfft: `RequireFftWorkspace(in ws, n, "rfft");` then `FftCoreRadix4(ref cz, ref sz, ref twReFull, ref twImFull, ws.n, false)`
```

FftCoreRadix4Ptr reads twr[tw3] with tw3 up to ~3*ws.n/4, requiring twReFull.N>=n, but RequireFftWorkspace only checks twRe/twIm(n/2), cz/sz, visited — not twReFull.N==n. fft/ifft correctly use RequireRadix4Workspace (lines 140/170).

**Verifier**

rfft (line 203) and irfft (line 297) call RequireFftWorkspace, which only checks twRe/twIm/cz/sz/visited lengths and ws.n; it does NOT verify twReFull.N==n or twImFull.N==n. Both functions then dispatch to FftCoreRadix4/FftCoreRadix4Mixed which forward twReFull/twImFull into FftCoreRadix4Ptr, whose tw3 index reaches ~3*ws.n/4 and thus requires the full-circle table. Sibling fft (line 140) and ifft (line 170) correctly use the stronger RequireRadix4Workspace. Impact is nil for factory-built caches (Arena.fProxyFFTCache always allocates the full-circle tables at length n) but fProxyFFTCache has public fields — a hand-built struct with an undersized twReFull/twImFull would silently overrun. Guard inconsistency and latent OOB read confirmed; suggested fix (swap RequireFftWorkspace for RequireRadix4Workspace at lines 203 and 297) is correct.

**Suggested fix**

Have rfft (line 203) and irfft (line 297) call RequireRadix4Workspace instead of RequireFftWorkspace so a workspace missing the full-circle table is rejected rather than read out of bounds. Impact is nil for factory-built caches (fProxyFFTCache always allocates it) but the guard is inconsistent with fft/ifft and a hand-built struct could overrun.

## Scanner notes

Verified clean (no defect) on the paths most likely to hide bugs: (1) Control SDA recursion A_{k+1}/G_{k+1}/H_{k+1} match Chiang-Fan-Lin with the correct (I+GH)^-1 vs (I+HG)^-1 solves and the Gk(I+HkGk)^-1=(I+GkHk)^-1Gk push-through identity; DARE RiccatiStep S=Q+A^TSA-A^TSB(R+B^TSB)^-1B^TSA is assembled correctly (BSA^T*K relies on S symmetry, which is enforced by SymmetrizeInPlace each step). (2) All Allocator.Temp temporaries are disposed on every path incl. early n==1 returns and Diverged breaks; RiccatiStep/SDACore skip inner allocations on the !Solved branch so nothing leaks. (3) FFT conjugate-trick inverses, radix-4 base-4 digit reversal, mixed-radix in-place cycle-following de-interleave (bijection + correct fixed points at 0 and size-1), and table twiddle strides W_len^j=T_tableN[j*(tableN/len)] are all correct; dft uses (long)k*t %n to avoid int overflow. (4) Easing constants (c1=1.70158, c2=c1*1.525, c4=2π/3, c5=2π/4.5, bounce n1/d1) and piecewise branches match the canonical definitions; Catmull-Rom stencil is standard. (5) RollingWindow ring index math (OldestRow/RingRow/Push wrap) and endpoint-pinning in Resample are correct. Also noted but not flagged as defects: minor perf/spec-reference phrasing in Control.Info.cs constant comments ("spec estimate", "not a target", cross-file REFACTOR_INTERVAL/ABS_GAP reference) and the "~10-25 steps typically" hint in Control.lqr XML doc — borderline contracts-only-policy items, left unreported to avoid over-flagging.
