import pandas as pd
from io import StringIO
from owNTLog_v049 import cNTLogFileReader, cNTLogOptions
import os

# Create a mock class for loadLogThread to handle 'canceled' and 'emitProgress'
class MockLoadLogThread:
    def __init__(self):
        self.canceled = False
    
    def emitProgress(self, value):
        pass  # Do nothing or print progress if needed

def read_ntlog_file(filepath):
    log_reader = cNTLogFileReader()
    
    # Create a log options object and configure it as needed
    log_options = cNTLogOptions()
    log_options.createFullTraffic = True  # Example configuration
    
    # Use the mock thread to avoid the 'NoneType' error
    mock_thread = MockLoadLogThread()
    
    # Read the log file with the specified log options
    trafficlog, datalog, infolog, auxiliary_data, gyroflowlog = log_reader.readLogFile(mock_thread, filepath, log_options)
    
    # Combine datalog and return as a pandas DataFrame
    data_lines = ''.join(datalog)  # datalog contains the log items as tab-separated lines
    data_io = StringIO(data_lines)
    
    # Read as a tab-separated pandas DataFrame
    df = pd.read_csv(data_io, delimiter='\t')
    
    return df

if __name__ == "__main__":

    file_path = "C:/Users/emilo/OneDrive - Danmarks Tekniske Universitet/MonoPulse/frequency_analysis/data/LOG0003_with_NF_135_20.NTLOG"
    
    # Check if the file exists and print its size
    if os.path.exists(file_path):
        # Get the file size in bytes
        file_size = os.path.getsize(file_path)
        # Convert to MB
        file_size_mb = file_size / (1024 * 1024)
        print(f"File found. Size: {file_size_mb:.2f} MB")
    else:
        print("File not found.")
    
    # Call the function to read the file and get the DataFrame
    df = read_ntlog_file(file_path)
    
    # Display DataFrame for verification
    print(df.head())
