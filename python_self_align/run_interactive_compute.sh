#!/bin/bash -l

# This script is intended to be run directly on an interactive compute node
# already allocated via srun (e.g., after 'srun ... --pty bash').
# It assumes your 'python_self_align' project directory (containing this script)
# is located within your /scratch space (e.g., /scratch/phan/ss5235/python_self_align).

# --- User Configuration (Verify this is correct) ---
UCID="ss5235"
# Assuming your PI's group for scratch might be different or same, adjust if needed.
# Using UCID for the main scratch path as per your `pwd` output.
SCRATCH_USER_ROOT="/scratch/phan/$UCID" # Base path in scratch where your code and outputs will live

echo "Starting interactive pipeline script on $(hostname)..."
echo "User: $(whoami)"
echo "UCID: $UCID"
echo "Running from SCRATCH_USER_ROOT: $SCRATCH_USER_ROOT"

# --- Environment Setup ---
# Load necessary modules (safer to include even if done interactively)
echo "Loading Python module..."
module load foss/2022b Python/3.10.8
if [ $? -ne 0 ]; then
    echo "Error loading Python module. Exiting."
    exit 1
fi
echo "Loaded modules:"
module list

# --- Project and Path Definitions ---
# The directory containing your Python code, setup.py, requirements.txt.
# This script should be INSIDE this directory, which is itself in SCRATCH_USER_ROOT.
# Example: $SCRATCH_USER_ROOT/python_self_align/run_interactive_compute.sh
CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Pip cache directory within your scratch space
PIP_CACHE_DIR="$SCRATCH_USER_ROOT/.cache/pip_interactive"
mkdir -p "$PIP_CACHE_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Could not create pip cache directory: $PIP_CACHE_DIR"
    exit 1
fi
export PIP_CACHE_DIR
echo "PIP_CACHE_DIR set to: $PIP_CACHE_DIR"

# Virtual environment will be stored within the CODE_DIR (which is on scratch)
VENV_NAME_INTERACTIVE="venv_interactive_scratch"
FULL_VENV_DIR_INTERACTIVE="$CODE_DIR/$VENV_NAME_INTERACTIVE"

# Output directory for the pipeline (on scratch)
OUTPUT_DIR_NAME_INTERACTIVE="pipeline_output_interactive_scratch_$(date +%Y%m%d-%H%M%S)" # Unique name
FULL_OUTPUT_DIR_INTERACTIVE="$SCRATCH_USER_ROOT/$OUTPUT_DIR_NAME_INTERACTIVE"

# Log file for the Python script's stdout/stderr (on scratch)
PYTHON_LOG_FILE_NAME_INTERACTIVE="pipeline_run_interactive_scratch_$(date +%Y%m%d-%H%M%S).log" # Unique name
FULL_PYTHON_LOG_FILE_INTERACTIVE="$SCRATCH_USER_ROOT/$PYTHON_LOG_FILE_NAME_INTERACTIVE"

# Define where persistent project outputs *could* go if you copy them later
PROJECT_PERSISTENT_ROOT="/project/phan/$UCID"

echo "--------------------------------------------------"
echo "Paths Configuration for Interactive Run (Primarily on Scratch):"
echo "User Scratch Root: $SCRATCH_USER_ROOT"
echo "Code directory (script location, venv parent - on scratch): $CODE_DIR"
echo "Python venv to be created/used at (on scratch): $FULL_VENV_DIR_INTERACTIVE"
echo "Pip cache directory (on scratch): $PIP_CACHE_DIR"
echo "Pipeline output directory (on scratch): $FULL_OUTPUT_DIR_INTERACTIVE"
echo "Python script stdout/stderr log (on scratch): $FULL_PYTHON_LOG_FILE_INTERACTIVE"
echo "Persistent project root (for manual copy later): $PROJECT_PERSISTENT_ROOT"
echo "--------------------------------------------------"

# Navigate to the code directory (where setup.py and requirements.txt are)
cd "$CODE_DIR"
if [ $? -ne 0 ]; then
    echo "Error changing to code directory $CODE_DIR. Exiting."
    exit 1
fi
echo "Current working directory: $(pwd)"

# --- Virtual Environment and Dependencies ---
echo "Setting up Python virtual environment at $FULL_VENV_DIR_INTERACTIVE..."
if [ ! -d "$FULL_VENV_DIR_INTERACTIVE" ]; then
    echo "Attempting to create virtual environment '$VENV_NAME_INTERACTIVE'..."
    python -m venv "$FULL_VENV_DIR_INTERACTIVE"
    VENV_CREATE_STATUS=$?
    if [ $VENV_CREATE_STATUS -ne 0 ]; then
        echo "Error: Python venv creation command failed with status $VENV_CREATE_STATUS."
        echo "The venv directory $FULL_VENV_DIR_INTERACTIVE might be incomplete or corrupted."
        exit 1
    fi
    if [ ! -f "$FULL_VENV_DIR_INTERACTIVE/bin/activate" ]; then
        echo "Error: Virtual environment '$VENV_NAME_INTERACTIVE' was not created successfully (activate script missing)."
        exit 1
    fi
    echo "Virtual environment '$VENV_NAME_INTERACTIVE' created."
else
    echo "Virtual environment '$VENV_NAME_INTERACTIVE' already exists. Reusing."
fi

if [ ! -f "$FULL_VENV_DIR_INTERACTIVE/bin/activate" ]; then
    echo "Critical Error: Activate script not found at $FULL_VENV_DIR_INTERACTIVE/bin/activate. Cannot proceed."
    exit 1
fi

echo "Activating virtual environment: $FULL_VENV_DIR_INTERACTIVE/bin/activate"
source "$FULL_VENV_DIR_INTERACTIVE/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error activating virtual environment. Exiting."
    exit 1
fi
echo "Python executable: $(which python)"
echo "Pip executable: $(which pip)"

echo "Upgrading pip..."
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Error upgrading pip. Continuing, but this might cause issues."
fi

# Install PyTorch with CUDA 12.1 (suitable for A100 GPUs)
echo "Installing PyTorch with CUDA 12.1 support..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if [ $? -ne 0 ]; then
    echo "Error installing PyTorch. Exiting."
    deactivate
    exit 1
fi
echo "PyTorch installed."

echo "Installing dependencies from requirements.txt..."
python -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing requirements.txt. Exiting."
    deactivate
    exit 1
fi

echo "Installing the python_self_align package (pip install .)..."
python -m pip install .
if [ $? -ne 0 ]; then
    echo "Error installing the package using 'pip install .'. Exiting."
    deactivate
    exit 1
fi
echo "All Python packages installed."

# --- Set Environment Variables for the Pipeline ---
echo "Setting environment variables for VLLM/pipeline..."
export OPENAI_API_KEY="EMPTY" # As per your instructions
export OPENAI_BASE_URL="http://0.0.0.0:8000/v1/" # As per your instructions
echo "OPENAI_API_KEY and OPENAI_BASE_URL set."
echo ""
echo "Model Path Note: Your instructions mention Model Path: /project/phan/codellama/StarCoder"
echo "Ensure your run_full_pipeline.py script uses this path if needed."
echo ""

# --- Prepare Output Directory (on scratch) ---
echo "Creating pipeline output directory (if it doesn't exist): $FULL_OUTPUT_DIR_INTERACTIVE"
mkdir -p "$FULL_OUTPUT_DIR_INTERACTIVE"
if [ $? -ne 0 ]; then
    echo "Error creating output directory $FULL_OUTPUT_DIR_INTERACTIVE. Exiting."
    deactivate
    exit 1
fi

# --- Run the Pipeline ---
echo "Running the Python pipeline..."
echo "Command: python -m python_self_align.run_full_pipeline --output_dir \"$FULL_OUTPUT_DIR_INTERACTIVE\" --num_samples 5000 --clean"
echo "Python script stdout/stderr will be redirected to: $FULL_PYTHON_LOG_FILE_INTERACTIVE"
SECONDS=0 # Start timer

python -m python_self_align.run_full_pipeline \
    --output_dir "$FULL_OUTPUT_DIR_INTERACTIVE" \
    --num_samples 5000 \
    --clean > "$FULL_PYTHON_LOG_FILE_INTERACTIVE" 2>&1

PIPELINE_EXIT_CODE=$?
DURATION=$SECONDS

echo "--------------------------------------------------"
if [ $PIPELINE_EXIT_CODE -ne 0 ]; then
    echo "Python pipeline exited with ERROR code $PIPELINE_EXIT_CODE."
else
    echo "Python pipeline completed SUCCESSFULLY."
fi
echo "Output is in: $FULL_OUTPUT_DIR_INTERACTIVE (ON SCRATCH - REMEMBER TO COPY IF NEEDED)"
echo "Python script log: $FULL_PYTHON_LOG_FILE_INTERACTIVE (ON SCRATCH)"
echo "Pipeline execution duration: $((DURATION / 60)) minutes and $((DURATION % 60)) seconds."
echo "--------------------------------------------------"

# --- Cleanup ---
echo "Deactivating virtual environment..."
deactivate

echo "Interactive script finished. Final exit code: $PIPELINE_EXIT_CODE"
echo "Remember that data in $SCRATCH_USER_ROOT is temporary and will be purged after 30 days."
echo "Copy any important results from $FULL_OUTPUT_DIR_INTERACTIVE to $PROJECT_PERSISTENT_ROOT if needed."
exit $PIPELINE_EXIT_CODE 
