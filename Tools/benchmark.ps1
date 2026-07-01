<#
.SYNOPSIS
  Runs the Burst performance benchmarks headlessly and prints the results table.

.DESCRIPTION
  Invokes a benchmark entry point via -executeMethod (mirrors regen.ps1). The
  benchmark runs each kernel inside a [BurstCompile] IJob.Run() so it measures
  native code, not the Mono interpreter, then writes a results table to
  TestResults/benchmark-all.txt which this script echoes.

  Requires the project to currently compile. Close the Unity Editor first
  (Unity locks the project for headless runs).

.PARAMETER Method
  Fully-qualified static entry point. Defaults to the combined kernel suite
  (GEMM, LU, Cholesky, QR). Pass a single kernel, e.g.
  LinearAlgebra.Benchmarks.QRBenchmark.Run, to run just one.
#>
param(
  [string]$Method = "LinearAlgebra.Benchmarks.AllBenchmarks.Run",
  # Pin the benchmark to the FREQUENCY CCD (the non-V-Cache die) so single-thread
  # timings are repeatable on a dual-CCD X3D chip (e.g. 9950X3D). By default we
  # pin to the UPPER half of logical processors, which on the standard enumeration
  # is the second CCD (cores 8-15 = the frequency die; the V-Cache die is CCD0).
  # If your BIOS enumerates the V-Cache die as CCD1, pass -PinLower to flip it, or
  # -NoAffinity to disable pinning entirely. Pinning affects TIMING ONLY -- numeric
  # results are identical on any core.
  [switch]$NoAffinity,
  [switch]$PinLower
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_unity-common.ps1"

$root    = Get-ProjectRoot
$Log     = Join-Path $root "TestResults\benchmark.log"

# Build the CPU affinity mask for one CCD (half the logical processors). Upper
# half by default (frequency die); lower half with -PinLower. 0 = no pin.
$mask = [long]0
if (-not $NoAffinity) {
  $lp   = [int][Environment]::ProcessorCount
  $half = [int]($lp / 2)
  if ($PinLower) { for ($i = 0;     $i -lt $half; $i++) { $mask = $mask -bor ([long]1 -shl $i) } }
  else           { for ($i = $half; $i -lt $lp;   $i++) { $mask = $mask -bor ([long]1 -shl $i) } }
  $ccd = if ($PinLower) { "lower ($half LPs, cores 0..$($half/2-1))" } else { "upper ($half LPs, cores $($half/2)..$($lp/2-1))" }
  Write-Host "Affinity: pinning benchmark to $ccd -- mask 0x$($mask.ToString('X')). (Use -NoAffinity to disable, -PinLower to flip CCD.)"
}

Write-Host "Running benchmark ($Method)..."
$exit = Invoke-Unity `
  -Arguments @("-nographics", "-quit", "-executeMethod", $Method) `
  -LogFile $Log `
  -AffinityMask $mask
Write-Host "Unity exit code: $exit"

$compileErrors = Get-CompileErrors $Log
if ($compileErrors.Count -gt 0) {
  Write-Host "`nFAIL: project does not compile, so the benchmark could not run:"
  $compileErrors | ForEach-Object { Write-Host "  $_" }
  exit 1
}
if (Select-String -Path $Log -Pattern "executeMethod method.*could not|couldn't be found" -Quiet) {
  Write-Host "`nFAIL: Unity could not find $Method. See $Log"
  exit 1
}

# The benchmark logs the file it wrote ("Benchmark results written to <path>"); echo that file.
$written = Select-String -Path $Log -Pattern "Benchmark results written to (.+)$" |
  Select-Object -Last 1
if ($written) {
  $Results = $written.Matches[0].Groups[1].Value.Trim()
  if (Test-Path $Results) {
    Write-Host ""
    Get-Content $Results
    exit 0
  }
}

Write-Host "`nFAIL: no results file produced. Last 40 log lines:"
if (Test-Path $Log) { Get-Content $Log -Tail 40 }
exit 1
