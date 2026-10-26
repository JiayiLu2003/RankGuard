$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path "artifacts/latest/experiment.json")) {
    uv run rankguard-train --quick
}
Push-Location frontend
npm install
npm run build
Pop-Location
uv run rankguard-api
