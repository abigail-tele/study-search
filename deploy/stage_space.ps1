<#
.SYNOPSIS
  Assemble a ready-to-push Hugging Face Space folder (Docker SDK). Does NOT push, log in, or create anything online.
.PARAMETER Out           Destination folder (default: ..\study-search-space).
.PARAMETER IncludeModel  Also copy checkpoints/flan-t5-pdq-best (~1 GB) and set live summaries possible.
.PARAMETER Db            Source database (default: data\trials_ship.db; falls back to data\trials.db).
.EXAMPLE
  powershell -File deploy\stage_space.ps1
  powershell -File deploy\stage_space.ps1 -IncludeModel
#>
param(
  [string]$Out = "",
  [switch]$IncludeModel,
  [string]$Db = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Out) { $Out = Join-Path (Split-Path -Parent $root) "study-search-space" }
if (-not $Db) {
  $Db = Join-Path $root "data\trials_ship.db"
  if (-not (Test-Path $Db)) { $Db = Join-Path $root "data\trials.db"; Write-Warning "trials_ship.db missing; using the full trials.db (bigger)." }
}

# Fail early if an artifact is missing (better than a broken Space build).
$need = @($Db, "data\faiss.index", "data\faiss_rowmap.json", "data\cancer_types.json", "data\umbrella_terms.json",
          "data\glossary.json", "data\zip_centroids.csv", "data\drug_lexicon.txt", "NOTICE", "LICENSE", "data\README.md", "models\all-MiniLM-L6-v2\config.json", "backend\app.py", "frontend\index.html")
foreach ($n in $need) {
  $p = if ([IO.Path]::IsPathRooted($n)) { $n } else { Join-Path $root $n }
  if (-not (Test-Path $p)) { throw "Missing required file: $p" }
}
if ($IncludeModel -and -not (Test-Path (Join-Path $root "checkpoints\flan-t5-pdq-best\model.safetensors"))) {
  throw "checkpoints\flan-t5-pdq-best\model.safetensors not found"
}

if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force $Out | Out-Null
function Copy-Tree($rel, [string[]]$xd = @()) {
  $args2 = @((Join-Path $root $rel), (Join-Path $Out $rel), "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/XD", "__pycache__") + $xd
  robocopy @args2 | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "robocopy failed for $rel" }
}
Copy-Tree "backend"
Copy-Tree "frontend" @("mock")   # mock JSON is only for offline UI work
Copy-Item (Join-Path $root "Dockerfile") $Out
Copy-Item (Join-Path $root ".dockerignore") $Out
New-Item -ItemType Directory -Force (Join-Path $Out "deploy") | Out-Null
Copy-Item (Join-Path $root "deploy\requirements-deploy.txt") (Join-Path $Out "deploy")
Copy-Item (Join-Path $root "LICENSE") $Out -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root "NOTICE") $Out -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root "CREDITS.md") $Out -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root "AI_USE.md") $Out -ErrorAction SilentlyContinue

# data/: the ship DB is renamed trials.db so backend/config.py needs no env var.
$data = Join-Path $Out "data"; New-Item -ItemType Directory -Force $data | Out-Null
Copy-Item $Db (Join-Path $data "trials.db")
foreach ($f in "faiss.index","faiss_rowmap.json","cancer_types.json","umbrella_terms.json","glossary.json","zip_centroids.csv") {
  Copy-Item (Join-Path $root "data\$f") $data
}
# data/README.md and drug_lexicon.txt: copied unchanged. The lexicon is read by faithfulness_check (needed with -IncludeModel).
foreach ($f in "README.md","drug_lexicon.txt") { Copy-Item (Join-Path $root "data\$f") $data }
Copy-Tree "models\all-MiniLM-L6-v2"
$ck = Join-Path $Out "checkpoints"; New-Item -ItemType Directory -Force $ck | Out-Null
if ($IncludeModel) { Copy-Tree "checkpoints\flan-t5-pdq-best" } else { Set-Content (Join-Path $ck ".gitkeep") "" }

# Hugging Face reads the YAML block at the top of README.md to configure the Space.
$live = if ($IncludeModel) { "1" } else { "0" }
@"
---
title: Cancer Trials Plain-Language Finder
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

Search cancer clinical trials in plain words. Trial data comes from ClinicalTrials.gov.
For education, not medical advice.
"@ | Set-Content -Encoding utf8 (Join-Path $Out "README.md")

# Git LFS: every big binary. (*.db, *.index, *.safetensors, etc. must NOT be stored as plain git blobs.)
@"
*.db filter=lfs diff=lfs merge=lfs -text
*.index filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
data/faiss_rowmap.json filter=lfs diff=lfs merge=lfs -text
"@ | Set-Content -Encoding ascii (Join-Path $Out ".gitattributes")

$mb = [math]::Round(((Get-ChildItem $Out -Recurse -File | Measure-Object Length -Sum).Sum) / 1MB)
Write-Host "Staged Space folder: $Out  ($mb MB)  live-summaries-possible=$live"
Write-Host "Next (human): cd `"$Out`"; git init; git lfs install; git add .; git commit -m 'Initial Space'; git remote add space https://huggingface.co/spaces/<user>/<name>; git push space main"
