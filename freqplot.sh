#!/bin/bash

# Define variables
VENV_DIR="venv"  # Name of your virtual environment folder
SCRIPT="src/plot.py"  # Your Python script
PORT=8501         # Port to run the Streamlit app

# Step 1: Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv $VENV_DIR
    pip3 install -r requiements.txt
fi

# Step 2: Activate the virtual environment
source $VENV_DIR/bin/activate

# Step 3: Run the Streamlit application
echo "Starting Streamlit application..."
streamlit run $SCRIPT --server.port $PORT

# Step 4: Deactivate the virtual environment after the script exits
deactivate
