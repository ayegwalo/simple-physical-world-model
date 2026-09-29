
# Simple Physical World Model

A beginner project exploring physical-system state prediction using Raspberry Pi, Python, and machine learning.

## Project Overview

This project uses a physical pushbutton connected to a Raspberry Pi to collect real-world observations and predict the next button state.

The system records button readings, constructs a time-series dataset, and trains a logistic regression model to predict the next state based on the previous five observations.

## Hardware

- Raspberry Pi
- Pushbutton
- Breadboard
- Jumper wires

## Software and Libraries

- Python
- gpiozero
- NumPy
- Pandas
- scikit-learn
- joblib

## How It Works

1. Read the button state through GPIO 17.
2. Collect button observations at regular intervals.
3. Create input sequences of five consecutive readings.
4. Train a logistic regression model to predict the next state.
5. Evaluate the model against a simple baseline.
6. Save the trained model for future predictions.

## Initial Experiment

The first experiment produced the following results:

| Metric | Result |
|---|---:|
| Test samples | 58 |
| Logistic regression accuracy | 86.21% |
| Baseline accuracy | 79.31% |

The model performed better than the baseline on this test set.

The dataset is small, and the observations are temporally correlated. Further experiments using independent recordings are needed to assess generalisation.

## Run the Project

Create and activate a virtual environment:

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the data collection and training program:

```bash
python button_ml_model.py
```

## Future Improvements

- Evaluate on completely independent recordings.
- Compare logistic regression with other machine-learning models.
- Investigate PyTorch for sequence prediction.
- Explore camera and sensor data for physical-system prediction.

## Author

Christopher Idanesi Ayegwalo

MSc Physics, University of Vienna
