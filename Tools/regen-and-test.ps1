<#
.SYNOPSIS
  Everyday "I edited a template" loop: regenerate sources, then compile + test.

.DESCRIPTION
  Runs codegen, and if that succeeds, runs the EditMode test suite (which does a
  full compile of production + test code, so it doubles as the compile check).
  Stops at the first failing step.

  This is two Unity launches (~1 min startup each). To run the tests alone
  (fast, no regen) use run-tests.ps1; to inspect the regen diff without testing
  use regen.ps1.

.PARAMETER Filter
  Optional test name filter forwarded to run-tests.ps1 (e.g. "*SVD*").
#>
param([string]$Filter)
$ErrorActionPreference = "Stop"

& "$PSScriptRoot\regen.ps1"
if ($LASTEXITCODE -ne 0) {
  Write-Host "`nStopping: codegen failed."
  exit $LASTEXITCODE
}

Write-Host "`n=== Codegen OK -> running tests ===`n"
if ($Filter) { & "$PSScriptRoot\run-tests.ps1" -Filter $Filter }
else         { & "$PSScriptRoot\run-tests.ps1" }
exit $LASTEXITCODE
