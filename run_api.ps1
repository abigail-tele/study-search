# Runs the whole app (API + frontend) at http://127.0.0.1:8000
# --no-access-log: rule 8, request URLs (search text, ZIPs) must never be logged.
Set-Location -Path $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No .venv found. Run: py -3.11 -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    exit 1
}
Write-Host "Open http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop."
& $python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --no-access-log
