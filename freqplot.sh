#!/bin/bash

# Define variables
VENV_DIR="venv"  # Name of your virtual environment folder
SCRIPT="src/plot.py"  # Your Python script
PORT=8501         # Port to run the Streamlit app

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip3 install -r requirements.txt
else 
    source $VENV_DIR/bin/activate
fi

echo "Starting Streamlit application..."
streamlit run $SCRIPT --server.port $PORT

deactivate
