param(
    [string]$Script = "scripts/run_stage0.py",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ProjectRoot
try {
    & python $Script @ScriptArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
