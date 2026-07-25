<#
.SYNOPSIS
  Runs the determinism conformance harness headlessly, or diffs two previously-written reports.

.DESCRIPTION
  Mirrors Tools/benchmark.ps1 (regen -> -executeMethod -> echo results), minus CPU-affinity pinning
  (hashes are timing-independent). Runs every op/group inside a [BurstCompile] IJob.Run() under
  FloatMode.Strict and writes TestResults/determinism-report.txt: a HarnessRev line, ROOT/ROOT-B
  hashes, and one GROUP/OP line per registered case. See
  docs/dev/spec-determinism-conformance-harness.md for the full contract.

  Requires the project to currently compile. Close the Unity Editor first (Unity locks the project
  for headless runs).

.PARAMETER NoRegen
  Skip the template->source codegen step and run whatever is already generated. Use when iterating
  on non-templated harness code.

.PARAMETER CompareA
.PARAMETER CompareB
  Diff mode (no Unity run): pass BOTH report file paths (-CompareA <file> -CompareB <file>). Refuses
  to compare reports with different `rev` lines. Prints the first diverging GROUP and its diverging OP
  lines; section-B divergences are labeled "expected under native-math builds". Exit 1 on any
  section-A divergence, 0 otherwise (including the zero-divergence case, which is also this script's
  own byte-identity self-check).

.EXAMPLE
  ./Tools/determinism-report.ps1
  ./Tools/determinism-report.ps1 -CompareA TestResults\run-x86.txt -CompareB TestResults\run-arm.txt
#>
param(
  [switch]$NoRegen,
  # Diff two existing reports (e.g. from two machines). Pass BOTH paths:
  #   determinism-report.ps1 -CompareA run-x86.txt -CompareB run-arm.txt
  # (two explicit params rather than one array so it binds correctly under `powershell -File`).
  [string]$CompareA,
  [string]$CompareB
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_unity-common.ps1"

function Read-Utf8NoBom([string]$path) {
  # File.ReadAllText, not Get-Content: PS 5.1 misdecodes BOM-less UTF-8 (report is written that way).
  return [System.IO.File]::ReadAllText($path)
}

function Get-RevLine([string[]]$lines) {
  foreach ($l in $lines) { if ($l.StartsWith("rev ")) { return $l } }
  return $null
}

# Parses "GROUP <id> <hex>" / "OP <group>/<case> <hex>" lines into an ordered map id -> hex, plus
# which ids belong to section B (everything from the "=== section B" fence onward).
function Get-HashMap([string[]]$lines) {
  $map = [ordered]@{}
  $sectionB = $false
  foreach ($l in $lines) {
    if ($l.StartsWith("=== section B")) { $sectionB = $true; continue }
    if ($l.StartsWith("GROUP ") -or $l.StartsWith("OP ") -or $l.StartsWith("ROOT")) {
      $parts = $l -split ' '
      $id = $parts[0] + " " + $parts[1]
      $hex = $parts[-1]
      $map[$id] = @{ hex = $hex; sectionB = $sectionB }
    }
  }
  return $map
}

if ($CompareA -or $CompareB) {
  if (-not ($CompareA -and $CompareB)) {
    Write-Host "FAIL: comparison requires BOTH -CompareA <file> and -CompareB <file>."
    exit 1
  }
  $pathA = $CompareA; $pathB = $CompareB
  if (-not (Test-Path $pathA)) { Write-Host "FAIL: not found: $pathA"; exit 1 }
  if (-not (Test-Path $pathB)) { Write-Host "FAIL: not found: $pathB"; exit 1 }

  $linesA = (Read-Utf8NoBom $pathA) -split "`n"
  $linesB = (Read-Utf8NoBom $pathB) -split "`n"

  $revA = Get-RevLine $linesA
  $revB = Get-RevLine $linesB
  if ($revA -ne $revB) {
    Write-Host "FAIL: revision mismatch ('$revA' vs '$revB') -- refusing to compare reports from different harness revisions."
    exit 1
  }

  $mapA = Get-HashMap $linesA
  $mapB = Get-HashMap $linesB

  $diffCount = 0
  $sectionADiff = $false
  $allIds = @()
  foreach ($k in $mapA.Keys) { $allIds += $k }
  foreach ($k in $mapB.Keys) { if (-not $mapA.Contains($k)) { $allIds += $k } }

  foreach ($id in $allIds) {
    $a = $mapA[$id]
    $b = $mapB[$id]
    if ($null -eq $a -or $null -eq $b) {
      Write-Host "DIFF (missing): $id -- present in $(if ($null -eq $a) {'B only'} else {'A only'})"
      $diffCount++
      if (-not ($a.sectionB -or $b.sectionB)) { $sectionADiff = $true }
      continue
    }
    if ($a.hex -ne $b.hex) {
      $tag = if ($a.sectionB) { " (expected under native-math builds)" } else { "" }
      Write-Host "DIFF: $id  A=$($a.hex)  B=$($b.hex)$tag"
      $diffCount++
      if (-not $a.sectionB) { $sectionADiff = $true }
    }
  }

  if ($diffCount -eq 0) {
    Write-Host "OK: reports are identical (rev match, zero GROUP/OP/ROOT divergences)."
    exit 0
  }
  Write-Host "`n$diffCount divergence(s) found."
  if ($sectionADiff) {
    Write-Host "FAIL: at least one section-A (deterministic-core) divergence -- this is a real bug, not an expected native-math difference."
    exit 1
  }
  Write-Host "OK: only section-B (native-math-sensitive) divergences -- expected under LINALG_NATIVE_MATH."
  exit 0
}

# ---- normal run mode ----

if (-not $NoRegen) {
  & "$PSScriptRoot\regen.ps1"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "`nStopping: codegen failed."
    exit $LASTEXITCODE
  }
  Write-Host "`n=== Codegen OK -> running determinism report ===`n"
}

$root   = Get-ProjectRoot
$Method = "BULA.Benchmarks.DeterminismReport.Run"
$Log    = Join-Path $root "TestResults\determinism.log"

Write-Host "Running determinism report ($Method)..."
$exit = Invoke-Unity `
  -Arguments @("-nographics", "-quit", "-executeMethod", $Method) `
  -LogFile $Log
Write-Host "Unity exit code: $exit"

$compileErrors = Get-CompileErrors $Log
if ($compileErrors.Count -gt 0) {
  Write-Host "`nFAIL: project does not compile, so the determinism report could not run:"
  $compileErrors | ForEach-Object { Write-Host "  $_" }
  exit 1
}
if (Select-String -Path $Log -Pattern "executeMethod method.*could not|couldn't be found" -Quiet) {
  Write-Host "`nFAIL: Unity could not find $Method. See $Log"
  exit 1
}

$logText = [System.IO.File]::ReadAllText($Log)
if ($logText -match "Determinism report FAILED[^\r\n]*") {
  Write-Host "`nFAIL: $($Matches[0])"
  exit 1
}

$m = [regex]::Matches($logText, "Determinism report written to (\S+)")
if ($m.Count -gt 0) {
  $Results = $m[$m.Count - 1].Groups[1].Value.Trim()
  if (-not [System.IO.Path]::IsPathRooted($Results)) { $Results = Join-Path $root $Results }
  if (Test-Path $Results) {
    Write-Host ""
    [System.IO.File]::ReadAllText($Results)
    exit 0
  }
}

Write-Host "`nFAIL: no report file produced. Last 40 log lines:"
if (Test-Path $Log) { Get-Content $Log -Tail 40 }
exit 1
