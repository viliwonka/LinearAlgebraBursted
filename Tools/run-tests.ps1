<#
.SYNOPSIS
  Runs the project's Unity Test Framework tests headlessly and reports a summary.

.DESCRIPTION
  Runs in -batchmode and trusts the NUnit XML result file rather than Unity's
  exit code, which is unreliable across versions. The test run also compiles the
  test assemblies (UNITY_INCLUDE_TESTS), so compile errors are reported too.

  NOTE: Unity locks the project, so close the Unity Editor before running this.

.PARAMETER Platform
  EditMode (default) or PlayMode.

.PARAMETER Filter
  Optional test name filter passed to Unity's -testFilter (e.g. "*SVD*").

.EXAMPLE
  ./Tools/run-tests.ps1
  ./Tools/run-tests.ps1 -Filter "*Eigen*"
#>
param(
  [ValidateSet("EditMode", "PlayMode")]
  [string]$Platform = "EditMode",
  [string]$Filter
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_unity-common.ps1"

$Out     = Join-Path (Get-ProjectRoot) "TestResults"
$Results = Join-Path $Out "$Platform.xml"
$Log     = Join-Path $Out "$Platform.log"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Remove-Item $Results -ErrorAction SilentlyContinue

Write-Host "Platform: $Platform"
Write-Host "Running tests (first run imports the project and can take several minutes)..."

$unityArgs = @("-runTests", "-testPlatform", $Platform, "-testResults", $Results)
# PlayMode needs a graphics device; EditMode can run headless.
if ($Platform -eq "EditMode") { $unityArgs += "-nographics" }
if ($Filter) { $unityArgs += @("-testFilter", $Filter) }

$exit = Invoke-Unity -Arguments $unityArgs -LogFile $Log
Write-Host "Unity exit code: $exit"

# Compile errors block the run before any XML is produced - surface them clearly.
$compileErrors = Get-CompileErrors $Log
if (-not (Test-Path $Results)) {
  if ($compileErrors.Count -gt 0) {
    Write-Host "`nFAIL: compilation errors prevented the test run:"
    $compileErrors | ForEach-Object { Write-Host "  $_" }
  } else {
    Write-Host "`nFAIL: no results file produced. Last 40 log lines:"
    if (Test-Path $Log) { Get-Content $Log -Tail 40 }
  }
  exit 1
}

[xml]$xml = Get-Content $Results
$run = $xml."test-run"
Write-Host ("Result={0} total={1} passed={2} failed={3} skipped={4} duration={5}s" -f `
  $run.result, $run.total, $run.passed, $run.failed, $run.skipped, $run.duration)

if ($run.result -eq "Passed") { exit 0 }

# Print the failing test names and messages to make CI/agent output actionable.
Write-Host "`n--- Failures ---"
$failures = $xml.SelectNodes("//test-case[@result='Failed']")
foreach ($f in $failures) {
  Write-Host ("FAILED: {0}" -f $f.fullname)
  if ($f.failure.message) { Write-Host ("  {0}" -f ($f.failure.message.Trim())) }
}
if (-not $failures -or $failures.Count -eq 0) {
  Write-Host "(no per-test failures parsed; tail of log:)"
  Get-Content $Log -Tail 60
}
exit 1
