# Define variables
$VENV_DIR = "venv"  # Name of your virtual environment folder
$SCRIPT = "src/plot.py"  # Your Python script
$PORT = 8501         # Port to run the Streamlit app

# Check if the virtual environment exists
if (-Not (Test-Path $VENV_DIR)) {
    Write-Host "Virtual environment not found. Creating one..."
    python -m venv $VENV_DIR
    & "$VENV_DIR\Scripts\Activate.ps1"
    pip install -r requirements.txt
}
else {
    & "$VENV_DIR\Scripts\Activate.ps1"
}

# Start the Streamlit app
Write-Host "Starting Streamlit application..."
streamlit run $SCRIPT --server.port $PORT

# Deactivate virtual environment
deactivate
