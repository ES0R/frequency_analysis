import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import welch, butter, filtfilt, savgol_filter
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


def plot_psd(df, imu1, imu2, sampling_rate):
    num_cols = len(imu1)  # Assuming IMU1 and IMU2 have the same length
    fig, axs = plt.subplots(1, num_cols, figsize=(num_cols * 6, 4))
    colors = ['g', 'r', 'b']  # Same colors for corresponding axes

    for col_idx, (sensor_data_imu1, sensor_data_imu2) in enumerate(zip(imu1, imu2)):
        # PSD for IMU1
        freqs_imu1, psd_imu1 = welch(df[sensor_data_imu1], fs=sampling_rate)
        # PSD for IMU2
        freqs_imu2, psd_imu2 = welch(df[sensor_data_imu2], fs=sampling_rate)

        # Plotting on the same axis
        ax = axs[col_idx]
        ax.semilogy(freqs_imu1, psd_imu1, color=colors[col_idx % len(colors)], linestyle='-', label=f'{sensor_data_imu1}')
        ax.semilogy(freqs_imu2, psd_imu2, color=colors[col_idx % len(colors)], linestyle='--', label=f'{sensor_data_imu2}')

        ax.set_title(f'PSD - {sensor_data_imu1} vs {sensor_data_imu2}')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power/Frequency (dB/Hz)')
        ax.grid(True)
        ax.set_xlim(0, max(freqs_imu1))
        ax.legend()

    plt.tight_layout()
    st.pyplot(fig)



def calculate_rms(df, data):
    rms_values = {}
    for col in data:
        rms = np.sqrt(np.mean(df[col] ** 2))
        rms_values[col] = rms
    return rms_values

from scipy.stats import gaussian_kde

def plot_prob(df, imu1, imu2):
    num_cols = len(imu1)  # Assuming IMU1 and IMU2 have the same length
    fig, axs = plt.subplots(1, num_cols, figsize=(num_cols * 6, 4))
    colors = ['g', 'r', 'b']  # Same colors for corresponding axes

    for col_idx, (sensor_data_imu1, sensor_data_imu2) in enumerate(zip(imu1, imu2)):
        # Data for IMU1
        data_imu1 = df[sensor_data_imu1]
        kde_imu1 = gaussian_kde(data_imu1, bw_method='scott')
        x_vals_imu1 = np.linspace(min(data_imu1), max(data_imu1), 500)
        y_vals_imu1 = kde_imu1(x_vals_imu1)

        # Data for IMU2
        data_imu2 = df[sensor_data_imu2]
        kde_imu2 = gaussian_kde(data_imu2, bw_method='scott')
        x_vals_imu2 = np.linspace(min(data_imu2), max(data_imu2), 500)
        y_vals_imu2 = kde_imu2(x_vals_imu2)

        # Plotting on the same axis
        ax = axs[col_idx]
        ax.plot(x_vals_imu1, y_vals_imu1, color=colors[col_idx % len(colors)], linestyle='-', label=f'{sensor_data_imu1}')
        ax.plot(x_vals_imu2, y_vals_imu2, color=colors[col_idx % len(colors)], linestyle='--', label=f'{sensor_data_imu2}')

        ax.set_title(f'Probability Density - {sensor_data_imu1} vs {sensor_data_imu2}')
        ax.set_xlabel('Angle [°]')
        ax.set_ylabel('Probability Density')
        ax.grid(True)
        ax.set_xscale('symlog')
        ax.set_xlim(-100, 100)
        ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

def apply_filter(data, filter_type, **kwargs):

    if filter_type == 'savgol':
        window_length = kwargs.get('window_length', 11)
        polyorder = kwargs.get('polyorder', 2)
        return savgol_filter(data, window_length=window_length, polyorder=polyorder)

    elif filter_type == 'moving_average':
        window_size = kwargs.get('window_size', 10)
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    elif filter_type == 'bandpass':
        low_cutoff = kwargs.get('low_cutoff', 0.1)
        high_cutoff = kwargs.get('high_cutoff', 30.0)
        sampling_rate = kwargs.get('sampling_rate', 1000)  # Hz
        order = kwargs.get('order', 4)
        
        nyquist = 0.5 * sampling_rate
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    else:
        raise ValueError("Unsupported filter type. Choose from 'savgol', 'moving_average', or 'bandpass'.")


def plot_trans(df, data1, data2, sampling_rate, filter_type='moving_average', cutoff_freq=10, filter_order=5, **filter_kwargs):
    num_pairs = len(data1)
    fig, axs = plt.subplots(1, num_pairs, figsize=(6 * num_pairs, 6))
    colors = ['r', 'g', 'b']

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

        # Apply filter to the transmissibility data using the apply_filter function
        filtered_transmissibility = apply_filter(transmissibility, filter_type, **filter_kwargs)

        ax = axs[i]
        # Plot original transmissibility with lower opacity
        ax.plot(freq, transmissibility, label=f'{sensor1} to {sensor2} (Original)', color=colors[i], lw=2, alpha=0.3)
        # Plot filtered transmissibility with full opacity
        ax.plot(freq, filtered_transmissibility, label=f'{sensor1} to {sensor2} (Filtered)', color=colors[i], lw=2)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0, max(freq))
        ax.set_ylim(min(filtered_transmissibility) / 10, max(filtered_transmissibility) * 10)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray')
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
        ax.set_title(f'Transmissibility from {sensor1} to {sensor2}')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Transmissibility')
        ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
