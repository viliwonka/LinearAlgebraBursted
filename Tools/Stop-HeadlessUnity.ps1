<#
  Kill ONLY the headless TEST-RUNNER Unity.exe launched by run-tests.ps1 against
  THIS project -- never an interactive editor and never its AssetImportWorker
  children (which DO run with -batchmode + this project path, so -batchmode is NOT
  a safe discriminator). Matches on "-runTests" (only the test runner passes it,
  Tools/run-tests.ps1) AND this project's resolved path in the command line.

  Use this instead of `Get-Process Unity | Stop-Process` (which would also kill an
  open editor). -WhatIf lists what would be killed without killing anything.
#>
param([switch]$WhatIf)

. "$PSScriptRoot\_unity-common.ps1"
$root = (Get-ProjectRoot)

$targets = @()
Get-CimInstance Win32_Process -Filter "Name='Unity.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
  $cl = $_.CommandLine
  if ($cl -and $cl.Contains('-runTests') -and $cl.Contains($root)) {
    $targets += $_
  }
}

if (-not $targets) { Write-Output "No headless Unity for this project is running."; return }

foreach ($t in $targets) {
  if ($WhatIf) {
    Write-Output "WOULD kill headless Unity PID $($t.ProcessId)"
  } else {
    try { Stop-Process -Id $t.ProcessId -Force -ErrorAction Stop; Write-Output "Killed headless Unity PID $($t.ProcessId)" }
    catch { Write-Output "Could not kill PID $($t.ProcessId): $_" }
  }
}
