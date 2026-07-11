<#
.SYNOPSIS
  Deletes generated .cs files whose source template no longer produces them (a
  template was renamed, moved, or deleted) - WITHOUT touching files still backed
  by a current template, and WITHOUT deleting .meta files.

.DESCRIPTION
  Codegen (TemplateConverter) overwrites generated files at predictable paths on
  every run, but never deletes a file whose template disappeared - that becomes an
  orphaned stale copy that still gets compiled alongside the fresh output, and
  either silently duplicates a class (CS0111) or blocks compilation outright once
  a rename lands (the fresh file references a renamed class the stale one still
  defines under the old name, or vice versa).

  This script computes the SAME source-path -> generated-path mapping
  TemplateConverter.cs uses (mirrored here, not invoked - this runs without Unity):
    - a "singular" file (contains //singularFile//, or has neither "fProxy" nor
      "iProxy" anywhere in its filename or content) maps 1:1, path unchanged.
    - a non-singular file with "fProxy" in its FILENAME maps to one path per
      float/double, substituting fProxy->float/fProxy->double (and FProxy->Float/
      FProxy->Double) across the whole relative path, not just the filename -
      folders named "fProxy"/"iProxy" get substituted too.
    - a non-singular file without "fProxy" in its filename maps to one path per
      int/short/long the same way (iProxy->int/short/long, IProxy->Int/Short/Long) -
      PLUS one path per //alsoExpand[type,...]// entry (e.g. uint), mirroring
      TemplateConverter's ResolveAlsoExpand.
    - files matching TemplateConverter's IgnoreFile (name contains "proxyStructs",
      "markers", or "proxyShims") produce no generated output at all.

  Deleting the .cs but keeping the .meta is deliberate: for paths a live template
  DOES still produce, regen.ps1 immediately writes a fresh .cs at that same path,
  and Unity reuses the surviving .meta's GUID (no reference/GUID churn). For a
  genuine orphan (no template produces that path anymore), the leftover .meta is
  inert - Unity cleans it up on its own asset refresh once it notices the .cs
  is gone for good; deleting it ourselves risks racing Unity's own bookkeeping.

  Safe: only ever deletes files that don't correspond to any live template - it
  never touches Source/Debug, Source/OP, or anything else hand-written outside
  the two Generated/ trees, and never blanket-wipes the way clean-generated.ps1
  intentionally does for a full-rebuild scenario.

.PARAMETER WhatIf
  List what would be deleted without deleting anything.
#>
param([switch]$WhatIf)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_unity-common.ps1"

$root = Get-ProjectRoot
$fProxy = "fProxy"; $iProxy = "iProxy"
$capFProxy = "FProxy"; $capIProxy = "IProxy"
$floatTypes = @("float", "double"); $capFloatTypes = @("Float", "Double")
$intTypes   = @("int", "short", "long"); $capIntTypes = @("Int", "Short", "Long")
# Caps spellings for alsoExpand-able extra types - mirrors GenUtils' (type, caps) pairs.
$extraCapsMap = @{ "uint" = "UInt" }

# Mirrors TemplateConverter.ResolveAlsoExpand: extra int-family expansion targets
# declared per-file via //alsoExpand[type,...]//. Returns @() when absent.
function Get-AlsoExpandTypes($content) {
  $m = [regex]::Match($content, '//alsoExpand\[([^\]]+)\]//')
  if (-not $m.Success) { return @() }
  return @($m.Groups[1].Value.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Test-IgnoredFile($name) {
  return ($name -match "proxyStructs" -or $name -match "markers" -or $name -match "proxyShims")
}

function Get-ExpectedPaths($templateRoot, $generatedRoot) {
  $expected = New-Object System.Collections.Generic.HashSet[string]
  $templates = Get-ChildItem -Path $templateRoot -Recurse -Filter *.cs -File
  foreach ($f in $templates) {
    if (Test-IgnoredFile $f.Name) { continue }
    $rel = $f.FullName.Substring($templateRoot.Length + 1)
    $content = [System.IO.File]::ReadAllText($f.FullName)

    $isSingular = $content.Contains("//singularFile//") -or
      (-not ($f.Name.Contains($fProxy) -or $content.Contains($fProxy) -or
             $f.Name.Contains($iProxy) -or $content.Contains($iProxy)))

    if ($isSingular) {
      [void]$expected.Add((Join-Path $generatedRoot $rel))
      continue
    }

    if ($f.Name.Contains($fProxy)) {
      for ($i = 0; $i -lt $floatTypes.Length; $i++) {
        $target = $rel.Replace($fProxy, $floatTypes[$i]).Replace($capFProxy, $capFloatTypes[$i])
        [void]$expected.Add((Join-Path $generatedRoot $target))
      }
    } else {
      $types = @($intTypes); $caps = @($capIntTypes)
      foreach ($t in (Get-AlsoExpandTypes $content)) {
        $types += $t
        if ($extraCapsMap.ContainsKey($t)) { $caps += $extraCapsMap[$t] }
        else { $caps += ($t.Substring(0,1).ToUpper() + $t.Substring(1)) }
      }
      for ($i = 0; $i -lt $types.Length; $i++) {
        $target = $rel.Replace($iProxy, $types[$i]).Replace($capIProxy, $caps[$i])
        [void]$expected.Add((Join-Path $generatedRoot $target))
      }
    }
  }
  return $expected
}

$pairs = @(
  @{ Template = (Join-Path $root "Assets\LinearAlgebra\CodeGen\TemplateSource");           Generated = (Join-Path $root "Assets\LinearAlgebra\Source") },
  @{ Template = (Join-Path $root "Assets\LinearAlgebra\CodeGen\TemplateSourceTests");      Generated = (Join-Path $root "Assets\LinearAlgebra\SourceTests\Generated") },
  @{ Template = (Join-Path $root "Assets\LinearAlgebra\CodeGen\TemplateSourceBenchmarks"); Generated = (Join-Path $root "Assets\LinearAlgebra\Benchmarks\Generated") }
)

$totalOrphans = 0
foreach ($pair in $pairs) {
  if (-not (Test-Path $pair.Generated)) { continue }
  $expected = Get-ExpectedPaths $pair.Template $pair.Generated
  $actual = Get-ChildItem -Path $pair.Generated -Recurse -Filter *.cs -File
  foreach ($f in $actual) {
    if ($expected.Contains($f.FullName)) { continue }
    $totalOrphans++
    $relDisplay = $f.FullName.Substring($root.Length + 1)
    if ($WhatIf) {
      Write-Host "Would delete orphan: $relDisplay"
    } else {
      Remove-Item $f.FullName -Force
      Write-Host "Deleted orphan: $relDisplay"
    }
  }
}

if ($totalOrphans -eq 0) {
  Write-Host "No orphaned generated files found."
} elseif ($WhatIf) {
  Write-Host "`n-WhatIf: $totalOrphans orphan(s) would be deleted (.meta files left for Unity to reconcile)."
} else {
  Write-Host "`nPruned $totalOrphans orphaned generated file(s). .meta files left in place."
}
