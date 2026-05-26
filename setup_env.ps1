param(
    [string]$PythonCommand = "python",
    [string]$KernelName = "ml-workshop",
    [string]$DisplayName = "Python (ML Workshop)"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Invoke-CheckedCommand {
    param(
        [string]$Description,
        [scriptblock]$Command
    )

    Write-Host $Description
    & $Command

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Description"
    }
}

Write-Host "Creating virtual environment in .venv..."
& $PythonCommand -m venv .venv
if ($LASTEXITCODE -ne 0) {
    throw "Unable to create virtual environment with '$PythonCommand'."
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Invoke-CheckedCommand "Upgrading pip..." {
    & $VenvPython -m pip install --upgrade pip
}

Invoke-CheckedCommand "Installing project requirements..." {
    & $VenvPython -m pip install -r requirements.txt
}

Invoke-CheckedCommand "Registering Jupyter kernel '$KernelName'..." {
    & $VenvPython -m ipykernel install --user --name $KernelName --display-name $DisplayName
}

Write-Host ""
Write-Host "Environment is ready."
Write-Host "Start Jupyter with:"
Write-Host "  .\.venv\Scripts\jupyter notebook"
Write-Host ""
Write-Host "Then open notebooks\ml_workshop_support_ticket_classification.ipynb"
Write-Host "and choose kernel: $DisplayName"
