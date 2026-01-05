# LTE CQI Prediction with Machine Learning

This repository contains a MATLAB simulation of an LTE downlink scenario. The goal is to predict the **Channel Quality Indicator (CQI)** based on user position and signal conditions using Machine Learning algorithms.

##  Key Features
* **LTE Physics Simulation:** Implements Path Loss models, Log-Normal Shadowing, and SNR calculation based on standard LTE parameters (20 MHz BW).
* **Machine Learning Comparison:** Compares the performance of two regression models:
  * Linear Regression (`fitlm`)
  * Decision Trees (`fitrtree`)
* **Metrics:** Evaluates accuracy using Root Mean Square Error (RMSE).

## Tech Stack
* MATLAB
* Statistics and Machine Learning Toolbox

##  How it works
1. **Data Generation:** Users are randomly distributed in a cell.
2. **Channel Modeling:** Path loss and shadowing are applied to calculate Received Power and SNR.
3. **Training:** The dataset is split (70/30) to train the models.
4. **Evaluation:** RMSE is calculated over multiple simulation rounds to ensure statistical stability.
