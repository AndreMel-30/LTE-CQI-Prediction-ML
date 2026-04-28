# LTE CQI Prediction with Machine Learning

This project contains a simple MATLAB simulation of an LTE downlink scenario, combined with a small machine learning experiment to predict the **Channel Quality Indicator (CQI)** from simulated radio features.

The goal is not to reproduce a full LTE stack, but to build a compact and understandable workflow that connects:
- basic wireless channel modeling,
- synthetic dataset generation,
- and regression-based CQI prediction.

## Project idea

In each simulation round, a set of users is randomly distributed inside a cell.  
For each user, the script estimates:
- distance from the base station,
- path loss,
- shadowing,
- received power,
- and SNR.

Starting from the simulated SNR, a CQI value is assigned through fixed threshold mapping.  
The generated dataset is then used to compare two machine learning models:
- **Linear Regression**
- **Decision Tree Regression**

The models are evaluated over multiple rounds using **RMSE**.

## What the script does

The workflow is the following:

1. Generate random user positions inside the cell.
2. Compute distance from the base station.
3. Apply path loss and log-normal shadowing.
4. Estimate received power and SNR.
5. Map SNR values to CQI levels.
6. Split the dataset into training and test sets.
7. Train the regression models.
8. Compare their prediction error across multiple rounds.
9. Plot per-round and average RMSE.

## Main assumptions

This is a simplified academic project, so a few assumptions are intentionally kept simple:

- Single-cell LTE downlink scenario.
- CQI derived from SNR through fixed thresholds.
- No detailed modeling of interference, scheduling, fading dynamics, or full protocol behavior.
- Synthetic data generated directly inside the script.

Because of these assumptions, the project should be seen as a **didactic simulation**, not as a full LTE performance model.

## Technologies used

- **MATLAB**
- **Statistics and Machine Learning Toolbox**

Main MATLAB functions used:
- `fitlm`
- `fitrtree`
- `cvpartition`
- plotting utilities for result visualization

## How to run

1. Open the script in MATLAB.
2. Run it from the MATLAB environment.
3. Enter the number of simulation rounds when prompted.

Example:

```matlab
Inserisci numero di round di simulazione (es. 50): 20
```

If no value is provided, the script uses a default number of rounds.

## Output

At the end of the execution, the script shows:
- the RMSE trend over all simulation rounds,
- the average RMSE of the two models,
- and a simple visual comparison between Linear Regression and Decision Tree.

## Repository structure

- `main.m` → main simulation and training script

If you rename the script, update this section accordingly.

