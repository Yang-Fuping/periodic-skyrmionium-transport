$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ProjectRoot
try {
    & python -m unittest discover -s tests -v
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
