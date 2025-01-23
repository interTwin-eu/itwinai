# Models sensetivity analysis

## Overview
These notebooks are part of a project that analyzes the performance of Deep Learning (DL) models trained to detect and analyze signals in spectral data, specifically focusing on pulses and Radio Frequency Interferences (RFIs). Each notebook evaluates the accuracy of models under various conditions and parameters such as Signal-to-Noise Ratio (SNR), Dispersion Measure (DM), intensity, and width of the pulse or RFI, and the number of RFIs on a spectrogram.


1. Pulse_analysis.ipynb
In this notebook, the focus is on analyzing pulses in the spectral data. The model's accuracy is evaluated under different  Signal-noise-ratio (SNR), Dispersion Measure DM, and width of the pulse, helping in understanding limits of the model's performance wit different parameters of the pulses. 


2. NBRFI_analysis.ipynb
This notebook deals with the evaluation of models concerning Narrowband RFI (NBRFI). It examines the models under varying conditions like intensity, the number of RFIs in a spectrogram, and width of the RFI, helping in understanding limits of the model's performance wit different parameters of the RFIs. 


3. BBRFI_analysis.ipynb
BBRFI_analysis.ipynb centers on Broadband RFI (BBRFI) analysis. It evaluates how the models perform with different intensity levels, the number of RFIs in a spectrogram, and width of the RFI, helping in understanding limits of the model's performance wit different parameters of the RFIs. 