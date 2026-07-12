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
  Optional test name filter. Unity's -testFilter is a regex; this script also
  accepts glob-style '*' (converted to '.*'), so "*Eigen*" and "Eigen" both work.

.PARAMETER NoRegen
  Skip the template->source codegen step and test whatever is already
  generated. Use when iterating on non-templated test harness code.

.EXAMPLE
  ./Tools/run-tests.ps1
  ./Tools/run-tests.ps1 -Filter "*Eigen*"
#>
param(
  [ValidateSet("EditMode", "PlayMode")]
  [string]$Platform = "EditMode",
  [string]$Filter,
  [switch]$NoRegen
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_unity-common.ps1"

# Regenerate sources from templates first (unless -NoRegen), so the tests
# never run against stale generated code after a template edit. Mirrors
# benchmark.ps1 / regen-and-test.ps1; regen.ps1 is headless (no Unity) and fast.
if (-not $NoRegen) {
  & "$PSScriptRoot\regen.ps1"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "`nStopping: codegen failed."
    exit $LASTEXITCODE
  }
  Write-Host "`n=== Codegen OK -> running tests ===`n"
}

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
if ($Filter) {
  # Unity's -testFilter is a REGEX, so a glob like "*Cholesky*" is an invalid
  # pattern and aborts the run. Accept glob-style '*' by converting it to '.*'.
  $regexFilter = $Filter -replace '\*', '.*'
  $unityArgs += @("-testFilter", $regexFilter)
}

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

# File.ReadAllText, not Get-Content: PS 5.1 misdecodes BOM-less UTF-8 (test messages can carry non-ASCII).
[xml]$xml = [System.IO.File]::ReadAllText($Results)
$run = $xml."test-run"
Write-Host ("Result={0} total={1} passed={2} failed={3} skipped={4} duration={5}s" -f `
  $run.result, $run.total, $run.passed, $run.failed, $run.skipped, $run.duration)

if ([int]$run.total -eq 0) {
  Write-Host "`nFAIL: 0 tests matched (result=$($run.result)). Check -Filter for a typo."
  exit 1
}

if ($run.result -eq "Passed") { exit 0 }

# Print the failing test names and messages to make CI/agent output actionable.
Write-Host "`n--- Failures ---"
$failures = $xml.SelectNodes("//test-case[@result='Failed']")
foreach ($f in $failures) {
  Write-Host ("FAILED: {0}" -f $f.fullname)
  # NUnit's <failure><message> can be an XmlElement (CDATA / nested) rather than a
  # plain string, so extract inner text robustly before trimming.
  $msg = $f.failure.message
  if ($msg -is [System.Xml.XmlElement]) { $msg = $msg.InnerText }
  if ($msg) { Write-Host ("  {0}" -f ([string]$msg).Trim()) }
}
if (-not $failures -or $failures.Count -eq 0) {
  Write-Host "(no per-test failures parsed; tail of log:)"
  Get-Content $Log -Tail 60
}
exit 1
