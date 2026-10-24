# Serves the frontend at http://localhost:8765 (mock data until the API exists).
# Why a server: browsers block fetch() of the mock JSON when index.html is opened as a file.
Set-Location -Path $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No .venv found. Run: py -3.11 -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    exit 1
}
Write-Host "Open http://localhost:8765 in your browser. Press Ctrl+C to stop."
& $python -m http.server 8765 --directory frontend
