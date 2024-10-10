import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from misc import *
import tempfile

########
# TODO 
# 1. Dynamically find out how many IMU, ACC and GYRO exist


IMU1 = ['Imu1Roll', 'Imu1Pitch', 'Imu1Yaw']
IMU2 = ['Imu2Roll', 'Imu2Pitch', 'Imu2Yaw']
ACC1 = ['ax1raw', 'ay1raw', 'az1raw']
ACC2 = ['ax2raw', 'ay2raw', 'az2raw']
GYRO1 = ['gx1raw', 'gy1raw', 'gz1raw']
GYRO2 = ['gx2raw', 'gy2raw', 'gz2raw']

st.set_page_config(
    initial_sidebar_state="auto",
    page_title="Gimbal Performance Analysis",
)

st.title("Gimbal Performance Analysis")

uploaded_file = st.file_uploader("Upload a NTLOG file", type="NTLOG")  # Update the file type if necessary

if uploaded_file is not None:
    st.write("Processing the uploaded file...")

    # Show a spinner during processing
    with st.spinner('Reading and converting NTLOG file...'):
        file_bytes = uploaded_file.read()

        # Debug: Check the size of the uploaded file
        file_size = len(file_bytes)
        st.write(f"Uploaded file size: {file_size} bytes")

        # Use a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        # Now pass the temporary file path to the read_ntlog_file function
        df = read_ntlog_file(temp_file_path)
    
    if df is not None:
        if df.iloc[0].isnull().all(): # Fixes none values at index 0 
            df.iloc[0] = 0

        # Display progress bar for processing steps
        progress_bar = st.progress(0)

        # Convert time from ms to seconds
        st.write("Converting time units...")
        df['Time'] = df['Time'] / 1000
        progress_bar.progress(10)

        # Calculate sampling rate
        sampling_rate = calculate_sampling_rate(df)

        # Plot FFT ACC
        st.write("## FFT ACC")
        plot_fft(df, ACC1, ACC2, sampling_rate=sampling_rate)
        progress_bar.progress(20)

        # # Plot FFT GYRO
        # st.write("## FFT GYRO")
        # plot_fft(df, GYRO1, GYRO2, sampling_rate=sampling_rate)

        # # Plot FFT IMU
        # st.write("## FFT IMU")
        # plot_fft(df, IMU1, IMU2, sampling_rate=sampling_rate)

        # Plot PSD ACC
        st.write("## PSD ACC")
        plot_psd(df, ACC1, ACC2, sampling_rate=sampling_rate)
        progress_bar.progress(30)

        # Plot PSD GYRO
        st.write("## PSD GYRO")
        plot_psd(df, GYRO1, GYRO2, sampling_rate=sampling_rate)
        progress_bar.progress(40)

        # Plot PSD IMU
        st.write("## PSD IMU")
        plot_psd(df, IMU1, IMU2, sampling_rate=sampling_rate)
        progress_bar.progress(50)

        # Calculate and display RMS values
        st.write("## RMS Values")
        rms_values = calculate_rms(df, ACC1 + ACC2)
        rms_df = pd.DataFrame(list(rms_values.items()), columns=['Signal', 'RMS'])
        st.table(rms_df)
        progress_bar.progress(60)

        st.write("## Probability Density IMU")
        plot_prob(df, IMU1, IMU2)  # Passing IMU data separately to maintain clear sensor type separation
        progress_bar.progress(70)


        st.write("## Transmissibility function ACC")
        plot_trans(df, ACC1, ACC2, sampling_rate=sampling_rate)  # Explicit naming for clarity
        progress_bar.progress(80)

        st.write("## Transmissibility function GYRO")
        plot_trans(df, GYRO1, GYRO2, sampling_rate=sampling_rate)  # Explicit naming for clarity
        progress_bar.progress(90)

        st.write("## Transmissibility function IMU")
        plot_trans(df, IMU1, IMU2, sampling_rate=sampling_rate)  
        progress_bar.progress(100)

        # Show final completion message
        st.success("Analysis completed successfully!")
