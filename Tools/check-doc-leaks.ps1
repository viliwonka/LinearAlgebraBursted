<#
.SYNOPSIS
  Release guard: fails if internal dev-process artifacts leak into shipped code
  comments or public docs.

.DESCRIPTION
  Scans the SHIPPED surfaces plus all generated code (templates are covered transitively —
  run after regen so Generated reflects them):
    - Assets/LinearAlgebra/Source  (generated library code = the UPM package root)
    - Assets/LinearAlgebra/SourceTests/Generated   (generated tests)
    - Assets/LinearAlgebra/Benchmarks/Generated    (generated benchmarks)
    - docs/features                          (public feature docs)
    - README.md, CHANGELOG.md

  for patterns that belong in per-folder DEVLOG.md files or docs/dev/ instead
  (see CLAUDE.md "Comment policy"):
    - internal spec/doc paths (docs/dev/..., draft-spec-..., spec-*.md, rfc-*.md)
    - ticket/round codes (R6a, OQ-7, "STAGE n (", "failure mode N", FM2)
    - development history ("an earlier version", "used to", "previously we", dates of work)
    - agent/workflow references ("coder report", "test-writer", "third-review")
    - commit hashes referenced as provenance

  Exit 0 = clean, exit 1 = leaks found (each printed as path:line: text).
  Run after regen so Generated reflects the templates.

.PARAMETER ShowAll
  Print every hit (default caps output at 50 lines).
#>
param([switch]$ShowAll)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot

$targets = @(
  (Join-Path $root "Assets\LinearAlgebra\Source"),
  (Join-Path $root "Assets\LinearAlgebra\SourceTests\Generated"),
  (Join-Path $root "Assets\LinearAlgebra\Benchmarks\Generated"),
  (Join-Path $root "docs\features"),
  (Join-Path $root "README.md"),
  (Join-Path $root "CHANGELOG.md")
)

# Each entry: label + regex. `cs` = case-sensitive (-cmatch) — ticket codes are uppercase by
# convention; a case-insensitive match would hit ordinary code like `(r1, r2)` or the
# Barrodale-Roberts algorithm's own lowercase "stage 2" terminology.
$patterns = @(
  @{ label = "internal spec path";  rx = 'docs[/\\](dev[/\\]|draft-spec-|spec-|research-)|draft-spec-[a-z-]+\.md|rfc-[a-z-]+\.md' },
  # Round codes are letter-suffixed (R6a) -- requiring the suffix keeps code like `(R2, Id)` clean.
  @{ label = "ticket/round code";   rx = '\bOQ-\d+\b|\bKrylov R\d+[a-z]?\b|\(R\d+[a-z][/,)]|STAGE \d+ \(|\bFM\d\b|failure mode \d|\bPRE-R\d+[a-z]?\b'; cs = $true },
  # "previously computed/row/Active..." is algorithmic prose, not history -- match only narration forms.
  @{ label = "dev history";         rx = 'an earlier version|used to (be|have|repeatedly)|previously (we|the|both|1/|dereferenced|returned)|was (wrongly|renamed|removed) |old (field|version|shared gates)|pre-change code path|no longer load-bearing|changed from \w+ to (bool|this)|\(spec estimate\)|a later phase' },
  @{ label = "agent/workflow ref";  rx = "coder('s)? (final )?report|test-writer|third-review|mini-spec|per the spec\b|fable-caught|the orchestrator|review's CRITICAL|adversarial review" },
  @{ label = "commit provenance";   rx = '\b(commit|see) [0-9a-f]{7,10}\b|\bcommit 2(\.5)?\b' },
  @{ label = "perf verdict";        rx = '~?\d+(\.\d+)?x (faster|slower)|measured \(float32' }
)

# Deliberate markdown links into docs/dev (e.g. "[rfc-memory-model.md](../dev/rfc-memory-model.md)")
# are allowed in public .md docs — only bare prose citations are leaks.
$mdLinkExempt = '\]\((\.\./)*dev/'

$hits = New-Object System.Collections.Generic.List[string]

foreach ($target in $targets) {
  if (-not (Test-Path $target)) { continue }
  $files = if ((Get-Item $target).PSIsContainer) {
    Get-ChildItem $target -Recurse -File | Where-Object { $_.Extension -in ".cs", ".md" }
  } else { @(Get-Item $target) }

  foreach ($f in $files) {
    $lineNo = 0
    foreach ($line in [System.IO.File]::ReadLines($f.FullName)) {
      $lineNo++
      if ($f.Extension -eq ".md" -and $line -match $mdLinkExempt) { continue }
      foreach ($p in $patterns) {
        # A changelog describes changes by definition -- past-tense phrasing there is not a leak.
        if ($f.Name -eq "CHANGELOG.md" -and $p.label -eq "dev history") { continue }
        $isMatch = if ($p.cs) { $line -cmatch $p.rx } else { $line -imatch $p.rx }
        if ($isMatch) {
          $rel = $f.FullName.Substring($root.Length + 1)
          $trimmed = $line.Trim()
          if ($trimmed.Length -gt 160) { $trimmed = $trimmed.Substring(0, 160) + "..." }
          $hits.Add("[$($p.label)] ${rel}:${lineNo}: $trimmed")
          break
        }
      }
    }
  }
}

if ($hits.Count -eq 0) {
  Write-Host "check-doc-leaks: clean (no internal artifacts in shipped surfaces)." -ForegroundColor Green
  exit 0
}

Write-Host "check-doc-leaks: $($hits.Count) leak(s) found:" -ForegroundColor Red
$show = if ($ShowAll) { $hits } else { $hits | Select-Object -First 50 }
$show | ForEach-Object { Write-Host "  $_" }
if (-not $ShowAll -and $hits.Count -gt 50) { Write-Host "  ... and $($hits.Count - 50) more (rerun with -ShowAll)" }
exit 1
