import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from misc import *

IMU1 = ['Imu1Roll', 'Imu1Pitch', 'Imu1Yaw']
IMU2 = ['Imu2Roll', 'Imu2Pitch', 'Imu2Yaw']
ACC1 = ['ax1raw', 'ay1raw', 'az1raw']
ACC2 = ['ax2raw', 'ay2raw', 'az2raw']
GYRO1 = ['gx1raw', 'gy1raw', 'gz1raw']
GYRO2 = ['gx2raw', 'gy2raw', 'gz2raw']

st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto",
    page_title="Gimbal Performance Analysis",
)

st.title("Gimbal Performance Analysis")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, delimiter='\t')
    df = check_columns_in_csv(df)

    if df is not None:
        sampling_rate = calculate_sampling_rate(df)

        # st.write("## FFT ACC")
        # plot_fft(df, ACC1, ACC2, sampling_rate=sampling_rate)  # Ensure sampling_rate is named explicitly

        # st.write("## PSD ACC")
        # plot_psd(df, ACC1, ACC2, sampling_rate=sampling_rate, overlap=True)  # Explicit naming for clarity

        # st.write("## PSD GYRO")
        # plot_psd(df, GYRO1, GYRO2, sampling_rate=sampling_rate, overlap=False)  # Explicit naming for clarity

        # st.write("## PSD IMU")
        # plot_psd(df, IMU1, IMU2, sampling_rate=sampling_rate, overlap=False)  # Explicit naming for clarity

        # st.write("## RMS Values")
        # rms_values = calculate_rms(df, ACC1 + ACC2)  # Summing lists to pass all ACC data
        # rms_df = pd.DataFrame(list(rms_values.items()), columns=['Signal', 'RMS'])
        # st.table(rms_df)

        # st.write("## Probability Density IMU")
        # plot_prob(df, IMU1, IMU2)  # Passing IMU data separately to maintain clear sensor type separation

        # st.write("## Transmissibility function ACC")
        # plot_trans(df, ACC1, ACC2, sampling_rate=sampling_rate)  # Explicit naming for clarity

        # st.write("## Transmissibility function GYRO")
        # plot_trans(df, GYRO1, GYRO2, sampling_rate=sampling_rate)  # Explicit naming for clarity

        st.write("## Transmissibility function IMU")
        plot_trans(df, IMU1, IMU2, sampling_rate=sampling_rate)  