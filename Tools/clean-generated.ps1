<#
.SYNOPSIS
  Deletes ALL codegen output (Source, SourceTests/Generated) so the next
  regen.ps1 run rebuilds it from scratch.

.DESCRIPTION
  Codegen (TemplateConverter) overwrites generated files at predictable paths but
  never deletes files whose SOURCE template was renamed, moved, or removed — those
  become orphaned stale copies that still get compiled, and either silently duplicate
  a class (CS0111) or block compilation entirely once a template-side rename lands
  (chicken-and-egg: codegen needs the project to compile to run, but a stale generated
  file references a class/member that no longer exists).

  Selectively guessing which generated files are "stale" and hand-deleting or hand-
  patching them (as opposed to a full clean) is error-prone — a file can look stale
  by one heuristic while still being the CURRENT definition of a class other code
  depends on. A full clean removes that ambiguity entirely: delete everything under
  the two Generated/ trees, then regen.ps1 rebuilds all of it fresh from templates.

  Run this whenever a rename/move/delete touches CodeGen/TemplateSource or
  CodeGen/TemplateSourceTests, before the next regen.ps1 (or just use regen-and-test.ps1
  after this — an empty Generated/ tree is a safe starting point for it).

.PARAMETER WhatIf
  List what would be deleted without deleting anything.
#>
param([switch]$WhatIf)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_unity-common.ps1"

$root = Get-ProjectRoot
# Source\ IS the generated tree (the UPM package root); only package.json and the asmdef (and
# their .meta files) are hand-placed there, so cleaning = delete everything else under it.
$sourceRoot = Join-Path $root "Assets\LinearAlgebra\Source"
$sourceKeep = @("package.json", "package.json.meta",
                "BurstLinearAlgebra.asmdef", "BurstLinearAlgebra.asmdef.meta")
$targets = @(
  @{ Dir = $sourceRoot;                                                Keep = $sourceKeep },
  @{ Dir = (Join-Path $root "Assets\LinearAlgebra\SourceTests\Generated"); Keep = @() }
)

$total = 0
foreach ($t in $targets) {
  $dir = $t.Dir
  if (-not (Test-Path $dir)) { continue }
  # Delete CONTENTS, not the folder itself, so its own .meta (and Unity's GUID for it) survives.
  $items = Get-ChildItem -Path $dir -Force | Where-Object { $t.Keep -notcontains $_.Name }
  $total += $items.Count
  if ($WhatIf) {
    Write-Host "Would delete $($items.Count) item(s) under $dir"
    continue
  }
  Write-Host "Cleaning $dir ($($items.Count) item(s))..."
  $items | Remove-Item -Recurse -Force
}

if ($WhatIf) {
  Write-Host "`n-WhatIf: $total item(s) would be deleted. Re-run without -WhatIf to actually clean."
  exit 0
}

Write-Host "`nClean complete. Run Tools\regen.ps1 (or regen-and-test.ps1) to rebuild Generated/ from templates."
