# Install the TP environment on native Windows (PowerShell), CPU-only.
# For an NVIDIA GPU on Windows, use WSL2 + install.sh instead (see README.md > Install > Windows).
#
# Usage (Anaconda PowerShell Prompt):
#   .\install.ps1
#   $env:ENV_NAME="myenv"; .\install.ps1

$ErrorActionPreference = "Stop"

$EnvName = if ($env:ENV_NAME) { $env:ENV_NAME } else { "tp-hil" }
$PythonVersion = if ($env:PYTHON_VERSION) { $env:PYTHON_VERSION } else { "3.12" }
$LeRobotVersion = if ($env:LEROBOT_VERSION) { $env:LEROBOT_VERSION } else { "0.6.1" }  # pin the version the TP was tested with

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "conda not found. Install Miniconda first: https://docs.anaconda.com/miniconda/, then reopen an Anaconda Prompt."
    exit 1
}

$envExists = (conda env list) -match "^\s*$EnvName\s"
if ($envExists) {
    Write-Host ">> conda env '$EnvName' already exists, reusing it"
} else {
    Write-Host ">> creating conda env '$EnvName' (python $PythonVersion)"
    conda create -y -n $EnvName "python=$PythonVersion"
}

Write-Host ">> ffmpeg (needed to encode dataset videos)"
conda install -y -n $EnvName -c conda-forge "ffmpeg>=6,<8"

Write-Host ">> LeRobot $LeRobotVersion with the HIL-SERL extra"
conda run -n $EnvName pip install "lerobot[hilserl]==$LeRobotVersion"
if ($LASTEXITCODE -ne 0) {
    Write-Host ">> PyPI install failed, installing from source (tag v$LeRobotVersion)"
    if (Test-Path .lerobot_src) { Remove-Item -Recurse -Force .lerobot_src }
    git clone --depth 1 --branch "v$LeRobotVersion" https://github.com/huggingface/lerobot.git .lerobot_src
    conda run -n $EnvName pip install -e ".lerobot_src[hilserl]"
}

Write-Host ">> analysis and input-device tools"
conda run -n $EnvName pip install pandas matplotlib "wandb>=0.24,<0.28" pynput pygame hidapi

Write-Host ""
Write-Host "Done. This is a CPU-only install (no CUDA on native Windows); training will be slower"
Write-Host "than on a GPU lab machine, and that's expected -- see the note in TP.md Setup."
Write-Host ""
Write-Host "Next steps (in an Anaconda Prompt / PowerShell):"
Write-Host "  conda activate $EnvName"
Write-Host "  python prefetch.py        # download demos + vision encoder (once per machine)"
Write-Host "  python check_setup.py     # verify device, display, controller"
Write-Host "  wandb login"
Write-Host "Then follow TP.md."
