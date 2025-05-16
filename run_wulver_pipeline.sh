#!/bin/bash -l

#SBATCH -J python_pipeline      # Job name
#SBATCH -o slurm_output_%j.out  # Standard output log (%j expands to Job ID)
#SBATCH -e slurm_error_%j.err   # Standard error log (%j expands to Job ID)
#SBATCH -p gpu                  # Partition: gpu, general, debug, bigmem
#SBATCH --gres gpu:1            # Number of GPUs per node
#SBATCH -n 1                    # Number of tasks (usually 1 for single-node jobs)
#SBATCH --cpus-per-task=16      # Number of CPUs per task (adjust as needed for your workload)
#SBATCH --mem=64G               # Total memory for the job (e.g., 64G)
#SBATCH --qos=low               # Quality of Service: low, standard, high_$PI, debug
#SBATCH --account=phan          # SLURM account (e.g., phan or your PI's UCID group)
#SBATCH --time=72:00:00         # Wall time limit (HH:MM:SS)
#SBATCH --mail-type=ALL         # Email notifications: BEGIN, END, FAIL, ALL
#SBATCH --mail-user=your_ucid_here@njit.edu

# --- EXAMPLE DEBUG QUEUE CONFIGURATION (Comment out the GPU lines above and uncomment these) ---
# #SBATCH -J python_debug         # Job name for debug
# #SBATCH -o slurm_debug_%j.out   # Standard output log
# #SBATCH -e slurm_debug_%j.err   # Standard error log
# #SBATCH -p debug                # Partition: debug
# #SBATCH --cpus-per-task=4       # Max 4 CPUs for debug partition
# #SBATCH --mem=16G               # Adjust memory for debug (e.g., 16G, max for 4 cores at 4GB/core)
# #SBATCH --qos=debug             # Quality of Service: debug
# #SBATCH --account=phan          # SLURM account
# #SBATCH --time=01:00:00         # Max 8 hours, but 1 hour is good for a quick debug
# # IMPORTANT: Debug queue does not have GPUs. Ensure your code can run in CPU-only mode or test a CPU-only part.
# --- END DEBUG QUEUE EXAMPLE ---

# --- User Configuration ---
UCID="your_ucid_here"

# --- Sanity Checks (No changes needed here if UCID and mail-user are correct) ---
if [ "$UCID" == "your_ucid_here" ] || [[ "$SLURM_JOB_MAIL_USER" == *your_ucid_here* ]]; then
    echo "ERROR: Please replace 'your_ucid_here' with your actual NJIT UCID if not already done."
    # exit 1 # Temporarily commented out for automated runs if user has already set it.
fi
if [[ "$SLURM_JOB_MAIL_USER" == "your_ucid_here@njit.edu" ]]; then
    echo "WARNING: SLURM mail user is still set to the placeholder."
fi


# --- Environment Setup ---
echo "Starting job setup on $(hostname)..."
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node List: $SLURM_JOB_NODELIST"
echo "User: $(whoami)"
echo "UCID: $UCID"

# Load necessary modules
echo "Loading Python module..."
module load foss/2022b Python/3.10.8
if [ $? -ne 0 ]; then
    echo "Error loading Python module. Exiting."
    exit 1
fi
echo "Loaded modules:"
module list

# --- Project and Path Definitions ---
# This script assumes it is placed INSIDE the 'python_self_align' directory on the HPC.
# Example HPC path for this script: /project/phan/$UCID/python_self_align/run_wulver_pipeline.sh

# The root directory for your project on HPC (e.g., /project/phan/$UCID)
PROJECT_ROOT_ON_HPC="/project/phan/$UCID"

# The directory containing your Python code, setup.py, requirements.txt.
# This is the directory where this script is located.
CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Name of the virtual environment directory (will be created inside CODE_DIR on /project)
VENV_DIR_NAME="venv_pipeline"
FULL_VENV_DIR="$CODE_DIR/$VENV_DIR_NAME"

# Output directory for the pipeline (will be created in PROJECT_ROOT_ON_HPC on /project)
OUTPUT_DIR_NAME="pipeline_output_test_wulver" # Added _wulver to distinguish
FULL_OUTPUT_DIR="$PROJECT_ROOT_ON_HPC/$OUTPUT_DIR_NAME"

# Log file for the Python script's stdout/stderr (will be created in PROJECT_ROOT_ON_HPC on /project)
PYTHON_LOG_FILE_NAME="pipeline_run_wulver.log" # Added _wulver to distinguish
FULL_PYTHON_LOG_FILE="$PROJECT_ROOT_ON_HPC/$PYTHON_LOG_FILE_NAME"

# Scratch directory setup
SCRATCH_ROOT="/scratch/$UCID"
PIP_CACHE_DIR="$SCRATCH_ROOT/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"
export PIP_CACHE_DIR
echo "PIP_CACHE_DIR set to: $PIP_CACHE_DIR"

echo "--------------------------------------------------"
echo "Paths Configuration:"
echo "Project root on HPC (for outputs/logs): $PROJECT_ROOT_ON_HPC"
echo "Code directory (script location, venv parent): $CODE_DIR"
echo "Python venv to be created at: $FULL_VENV_DIR"
echo "Pipeline output directory: $FULL_OUTPUT_DIR"
echo "Python script stdout/stderr log: $FULL_PYTHON_LOG_FILE"
echo "SLURM output log: $(pwd)/slurm_output_$SLURM_JOB_ID.out (in submission directory)"
echo "SLURM error log: $(pwd)/slurm_error_$SLURM_JOB_ID.err (in submission directory)"
echo "--------------------------------------------------"

# Navigate to the code directory (where setup.py and requirements.txt are)
cd "$CODE_DIR"
if [ $? -ne 0 ]; then
    echo "Error changing to code directory $CODE_DIR. Exiting."
    exit 1
fi
echo "Current working directory: $(pwd)"

# --- Virtual Environment and Dependencies ---
echo "Setting up Python virtual environment at $FULL_VENV_DIR..."
if [ ! -d "$FULL_VENV_DIR" ]; then
    python -m venv "$FULL_VENV_DIR"
    echo "Virtual environment '$VENV_DIR_NAME' created."
else
    echo "Virtual environment '$VENV_DIR_NAME' already exists. Reusing."
fi

echo "Activating virtual environment: $FULL_VENV_DIR/bin/activate"
source "$FULL_VENV_DIR/bin/activate"
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
# This is done BEFORE requirements.txt in case any dependency needs torch
echo "Installing PyTorch with CUDA 12.1 support..."
# Make sure bitsandbytes is compatible with this PyTorch and CUDA version.
# bitsandbytes often requires specific CUDA toolkit versions to be discoverable by PyTorch.
# The pre-built PyTorch wheels usually bundle necessary CUDA runtime components.
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
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://0.0.0.0:8000/v1/"
echo "OPENAI_API_KEY and OPENAI_BASE_URL set."
echo ""
echo "NOTE: Ensure your Python script (run_full_pipeline.py) is configured to use"
echo "the model path: /project/phan/codellama/StarCoder if it's required."
echo "This might be handled via a script argument, another environment variable, or hardcoded within the script."
echo ""

# --- Prepare Output Directory ---
echo "Creating pipeline output directory (if it doesn't exist): $FULL_OUTPUT_DIR"
mkdir -p "$FULL_OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "Error creating output directory $FULL_OUTPUT_DIR. Exiting."
    deactivate
    exit 1
fi

# --- Run the Pipeline ---
echo "Running the Python pipeline..."
echo "Command: python -m python_self_align.run_full_pipeline --output_dir "$FULL_OUTPUT_DIR" --num_samples 5000 --clean"
echo "Python script stdout/stderr will be redirected to: $FULL_PYTHON_LOG_FILE"
SECONDS=0 # Start timer

python -m python_self_align.run_full_pipeline \
    --output_dir "$FULL_OUTPUT_DIR" \
    --num_samples 5000 \
    --clean > "$FULL_PYTHON_LOG_FILE" 2>&1

PIPELINE_EXIT_CODE=$?
DURATION=$SECONDS

echo "--------------------------------------------------"
if [ $PIPELINE_EXIT_CODE -ne 0 ]; then
    echo "Python pipeline exited with ERROR code $PIPELINE_EXIT_CODE."
    echo "Please check logs for details:"
    echo "  Python script log: $FULL_PYTHON_LOG_FILE"
    echo "  SLURM error log (in submission dir): $(pwd)/slurm_error_$SLURM_JOB_ID.err"
    echo "  SLURM output log (in submission dir): $(pwd)/slurm_output_$SLURM_JOB_ID.out"
else
    echo "Python pipeline completed SUCCESSFULLY."
    echo "Output is in: $FULL_OUTPUT_DIR"
    echo "Python script log: $FULL_PYTHON_LOG_FILE"
fi
echo "Pipeline execution duration: $((DURATION / 60)) minutes and $((DURATION % 60)) seconds."
echo "--------------------------------------------------"

# --- Cleanup ---
echo "Deactivating virtual environment..."
deactivate

echo "Job finished. Final exit code: $PIPELINE_EXIT_CODE"
exit $PIPELINE_EXIT_CODE 