# Frequency Analysis

This repo has the code to host a `Streamlit` app for visualizing frequency analysis. The app is hosted a static IP address where the hostdevice is a raspberry pi.

Software from `olliw STorM32BGC NT` is being used in this project. Specfically, the ``NTLoggerTool``.  

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
1. Clone and enter repo
```
git clone https://github.com/ES0R/frequency_analysis.git
cd frequency_analysis
```
2. Optional: Make script executable (Linux)
```
chmod +x freqplot.sh
```
3. Run on boot:
```
sudo nano /etc/systemd/system/streamlit-app.service
```
*  Copy and paste: 
```
[Unit]
Description=Streamlit Frequency Analysis App
After=network.target

[Service]
ExecStart=/home/monopulse/frequency_analysis/start_streamlit.sh
WorkingDirectory=/home/monopulse/frequency_analysis
StandardOutput=inherit
StandardError=inherit
Restart=always
User=monopulse # Change if your username is different

[Install]
WantedBy=multi-user.target
```
**Replace pi with user and ensure correct path to repo** 

4. Enable and start the service
```
sudo systemctl enable streamlit-app
sudo systemctl start streamlit-app
```
Optional: `sudo systemctl status streamlit-app` to inspect status of service.

5. Set static IP by inserting line into `cmdline.txt` on SD card
```
ip=172.20.10.10::172.20.10.1:255.255.255.0:rpi:wlan0
```
6. Access the application on network 
```
http://172.20.10.10:8501
```
