# Time Series Anomaly Detection: Device Malfunctioning prevention
This research work was an AI-based Self-diagnosis integrated control system for efficient operation of IT equipment.  
Specifically, it was was focused on Time Series Forecasting for detection of critical points in
which devices' CPU and Temperature values must avoid. Failure to do so wuold put the devices' 
operation into jeopardy. 

<div align="center">
<img src="https://github.com/user-attachments/assets/26e55e02-f269-46f1-afe8-af12a09e4653" width=80% height=80%>
</div><br />

Time Series data visualization of the Devices

<div align="center">
<img src="https://github.com/user-attachments/assets/612a93d9-8808-4d0c-9c5b-6d03e47c9099" width=80% height=100%>
</div><br />

### Objective metric scores of the Project:
* Accuracy (80%)
* Precision (50%)
* FVU (1 - R-squared) (0.25)

1-step Time series forecasting metrics:

<div align="center">
<img src="https://github.com/user-attachments/assets/024f369d-b89d-423d-a411-ad15e35a53c6" width=85% height=75%>
</div><br />

The time-series forecasting model this work was Long Short-Term Memory (LSTM), which was finetuned using
Bayesian Optimization. 

<div align="center">
<img src="https://github.com/user-attachments/assets/7d507bd2-fc22-41ea-b51b-a638b0464fe4" width=70% height=70%>
</div><br />

An N-step forecasting system was developed. Such development was made to offer a long-term futuristic 
forecasting system for long-term observance and management of the usage of the devices. 

<div align="center">
<img src="https://github.com/user-attachments/assets/5963f544-25e4-412c-95bb-176592fcea8e" width=70% height=70%>
</div><br />

Prevention of disappearance of Precision metric was made using the Quantile loss function
and an N-step forecasting system was developed. As seen below, every Precision score starting from
9th time step was restored.

<div align="center">
<img src="https://github.com/user-attachments/assets/8b411f77-ed4a-443c-863d-6111c4bee61e" width=70% height=70%>
</div><br />

