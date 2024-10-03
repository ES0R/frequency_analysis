import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import welch
from scipy.stats import gaussian_kde
import pandas as pd
from io import StringIO
from owNTLog_v049 import cNTLogFileReader, cNTLogOptions

def convert_ntlog_to_df(file):
    # Example of parsing based on specific NTLOG structure
    with open(file, 'r') as f:
        # Read and process the NTLOG file structure here, e.g., line by line
        # Adjust the delimiter or parsing based on the file format
        data = f.readlines()  # Adjust based on actual content
    # After processing, convert it to a DataFrame
    df = pd.DataFrame(data)  # Adjust based on the actual data structure
    return df

def check_columns_in_csv(df):
    required_columns = ['Time', 'ax1raw', 'ax2raw', 'ay1raw', 'ay2raw', 'az1raw', 'az2raw', 'gx1raw', 'gx2raw', 'gy1raw', 'gy2raw', 'gz1raw']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if not missing_columns:
        st.write("All required columns are present.")
        return df
    else:
        st.write("Missing columns:", missing_columns)
        return None

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
    df = pd.read_csv(data_io, delimiter='\t',low_memory=False)
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

def calculate_sampling_rate(df):
    time_diffs = np.diff(df['Time'])
    avg_time_diff = np.mean(time_diffs)
    sampling_rate = 1 / avg_time_diff
    return sampling_rate

def plot_fft(df, *args, sampling_rate):
    num_rows = len(args)
    num_cols = len(args[0]) if args else 0
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*6, num_rows*4))
    color = ['r', 'g', 'b']

    for row_idx, data_group in enumerate(args):
        for col_idx, sensor_data in enumerate(data_group):
            signal = df[sensor_data] - np.mean(df[sensor_data])
            N = len(signal)
            yf = fft(signal)
            xf = np.fft.fftfreq(N, 1 / sampling_rate)
            idx = np.where(xf >= 0)
            xf = xf[idx]
            yf = np.abs(yf[idx])

            ax = axs[row_idx, col_idx] if num_rows > 1 else axs[col_idx]
            ax.plot(xf, yf, label=f'FFT of {sensor_data}', color=color[col_idx])
            ax.set_xlim(0, np.max(xf))
            ax.set_title(f'FFT of {sensor_data}')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            ax.grid(True)

    plt.tight_layout()
    st.pyplot(fig)


def plot_psd(df, *args, sampling_rate, overlap):
    num_rows = len(args)  # Number of rows based on the number of sensor groups passed
    num_cols = len(args[0]) if args else 0  # Number of columns based on the number of elements in a sensor group
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*6, num_rows*4))
    color = ['r', 'g', 'b']
    for row_idx, data_group in enumerate(args):
        for col_idx, sensor_data in enumerate(data_group):
            freqs, psd = welch(df[sensor_data], fs=sampling_rate)
            ax = axs[row_idx, col_idx] if num_rows > 1 else axs[col_idx]  # Correct subplot indexing
            ax.semilogy(freqs, psd, label=f'PSD of {sensor_data}', color=color[col_idx])
            ax.set_xlim(0, np.max(freqs))
            ax.set_title(f'PSD of {sensor_data}')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power/Frequency (dB/Hz)')
            ax.grid(True)

    plt.tight_layout()
    st.pyplot(fig)



def calculate_rms(df, data):
    rms_values = {}
    for col in data:
        rms = np.sqrt(np.mean(df[col] ** 2))
        rms_values[col] = rms
    return rms_values

def plot_prob(df, *args):
    num_rows = len(args)
    num_cols = len(args[0]) if args else 0
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*6, num_rows*4))
    colors = ['r', 'b', 'g', 'y', 'm', 'c']  # Extended color range for more sensors

    for row_idx, data_group in enumerate(args):
        for col_idx, sensor_data in enumerate(data_group):
            data_col = df[sensor_data] #.dropna()
            kde = gaussian_kde(data_col, bw_method='scott')
            x_vals = np.linspace(min(data_col), max(data_col), 500)  # Range based on actual data
            y_vals = kde(x_vals)

            ax = axs[row_idx, col_idx] if num_rows > 1 else axs[col_idx]
            ax.plot(x_vals, y_vals, color=colors[col_idx % len(colors)], label=f'Probability Density of {sensor_data}')
            ax.set_title(f'Probability Density of {sensor_data}')
            ax.set_xlabel('Angle [o]')
            ax.set_ylabel('Probability Density')
            ax.grid(True)
            ax.set_xscale('symlog')
            ax.set_xlim(-100, 100)

    plt.tight_layout()
    st.pyplot(fig)


def plot_trans(df, data1, data2, sampling_rate):
    num_pairs = len(data1)  # Assuming data1 and data2 have the same length and correspond to x, y, z, etc.
    fig, axs = plt.subplots(1, num_pairs, figsize=(6 * num_pairs, 6))  # One row, multiple columns
    colors = ['r', 'g', 'b']  # Colors for each subplot

    for i, (sensor1, sensor2) in enumerate(zip(data1, data2)):
        imu1_signal = df[sensor1] - np.mean(df[sensor1])
        imu2_signal = df[sensor2] - np.mean(df[sensor2])

        N = len(imu1_signal)
        if N != len(imu2_signal):
            raise ValueError("Data arrays must have the same length.")
        
        imu1_fft = fft(imu1_signal)
        imu2_fft = fft(imu2_signal)

        freq = np.fft.fftfreq(N, 1 / sampling_rate)[:N // 2]
        transmissibility = np.abs(imu2_fft[:N // 2]) / np.abs(imu1_fft[:N // 2])

        ax = axs[i]  # Select the appropriate subplot
        ax.plot(freq, transmissibility, label=f'{sensor1} to {sensor2}', color=colors[i], lw=2)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0, max(freq))
        ax.set_ylim(min(transmissibility)/10, max(transmissibility)*10)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5)  # Set general grid properties
        ax.minorticks_on()  # Enable minor ticks which are necessary for minor grid lines
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray')  # Customize major grid lines
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')  # Customize minor grid lines
        ax.set_title(f'Transmissibility from {sensor1} to {sensor2}')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Transmissibility')
        ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

