# Headless Burst disassembly (Burst Inspector via console)

`bcl.exe` — the Burst compiler CLI shipped inside the package cache — compiles methods from
the built script assemblies and dumps the generated x64 assembly, without opening the editor
UI. Used 2026-07-14 to solve the packed-W microkernel collapse (see
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/DEVLOG.md`).

## Recipe

```powershell
$proj  = "C:\Users\viliv\Documents\LinearAlgebraBursted"
$unity = "C:\Program Files\Unity\Hub\Editor\6000.3.2f1\Editor\Data"
$bcl   = Get-ChildItem "$proj\Library\PackageCache\com.unity.burst@*\.Runtime\bcl.exe" | Select-Object -First 1

& $bcl.FullName --platform=Windows --target=AVX2 --dump=Asm `
    --assembly-folder="$proj\Library\ScriptAssemblies" `
    --assembly-folder="$unity\Managed" `
    --assembly-folder="$unity\Managed\UnityEngine" `
    --assembly-folder="$unity\NetStandard\ref\2.1.0" `
    --assembly-folder="$unity\NetStandard\compat\2.1.0\shims\netstandard" `
    --assembly="$proj\Library\ScriptAssemblies\BurstLinearAlgebra.dll" `
    --type="LinearAlgebra.Internal.UnsafeOP" `
    --output="out\prefix" > dump.txt 2>&1
```

## Hard-won details

- **Repeat `--assembly-folder`** per directory. The documented `path1;path2` list form throws
  `NotSupportedException` on Windows (the whole string is treated as one path).
- **All paths absolute.** Relative paths also throw `NotSupportedException`.
- The assembly listing goes to **stdout** (redirect it); `--output` receives the object/dll.
- `--type=<TypeFullName>` compiles all compilable **static** methods of the type as roots.
  Job structs' instance `Execute` won't root this way. Nested classes use `Outer/Nested`.
- `--method=` wants a Cecil-style name (`Type,Assembly::Method(args)`) whose argument parser
  rejects pointer types (`System.Single*`) — for pointer-heavy kernels use `--type` on a
  small probe class instead: a temporary `public static class` nested in the kernels' class
  (so it can call private kernels), one pass-through per kernel under test. `NoInlining`
  callees appear in the dump as separate functions **with readable names**; roots get
  hash-only names.
- The build must be current: run a headless compile first
  (`Unity.exe -batchmode -quit -projectPath <proj> -logFile <log>`), ~40 s.
- Useful greps on the dump: `vmovaps/vmovups ymmword ptr [rsp` (spill stores),
  `vpalignr|vperm2` (shuffle glue — a healthy dense kernel has none), `.seh_proc`/
  `.seh_endproc` (function boundaries), instruction histogram of the inner loop between the
  loop label and its backward branch.
- `--float-precision` / `--float-mode` accept the same values as `[BurstCompile]`; match the
  job under investigation when precision-sensitive codegen is in question.
