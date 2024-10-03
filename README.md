# Frequency Analysis

This repo has the code to host a `Streamlit` app for visualizing frequency analysis. The app is hosted a static IP address where the hostdevice is a raspberry pi.

## Usage
 
Simply drag and drop your binary NT-log file for data visualization. The following plots will be generated:

- FFT (ACC, GYRO, IMU)
- Power Spectral Density (ACC, GYRO, IMU)
- Probability Density (IMU)
- Transmissibility (ACC, GYRO, IMU)

### Local

1. Clone and enter repo
```
git clone https://github.com/ES0R/frequency_analysis.git
cd frequency_analysis
```
2. Optional: Make script executable (Linux)
```
chmod +x freqplot.sh
```
3. Start script
```
./freqplot.sh # Linux
./freqplot.ps1 # Windows
```
4. Access app by entering below IP address into URL
```
 http://localhost:8501
```

### Network
The Raspberry Pi hosts the Streamlit app on a static IP address. To access the app from other devices, ensure they are connected to the Monopulse Wi-Fi network. Enter the Pi's IP address:

```http://172.20.10.8:8501```

#### Setup


