<#
.SYNOPSIS
  Everyday "I edited a template" loop: regenerate sources, then compile + test.

.DESCRIPTION
  run-tests.ps1 now regenerates from templates by default (pass -NoRegen there
  to skip), so this script is a thin forwarder to it for back-compat with
  existing muscle-memory/callers. To run the tests alone (fast, no regen) use
  run-tests.ps1 -NoRegen; to inspect the regen diff without testing use
  regen.ps1.

.PARAMETER Filter
  Optional test name filter forwarded to run-tests.ps1 (e.g. "*SVD*").
#>
param([string]$Filter)
$ErrorActionPreference = "Stop"

if ($Filter) { & "$PSScriptRoot\run-tests.ps1" -Filter $Filter }
else         { & "$PSScriptRoot\run-tests.ps1" }
exit $LASTEXITCODE
