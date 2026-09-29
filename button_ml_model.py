
import time
import csv
import os

import numpy as np
import pandas as pd

from gpiozero import Button
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier
import joblib


# ==========================================
# 1. SETTINGS
# ==========================================

GPIO_PIN = 17

SAMPLE_INTERVAL = 0.1   # Read every 0.1 seconds
COLLECTION_TIME = 60    # Collect data for 60 seconds

WINDOW_SIZE = 5

CSV_FILE = "button_ml_data.csv"
MODEL_FILE = "button_model.pkl"


# ==========================================
# 2. CONNECT TO THE BUTTON
# ==========================================

button = Button(GPIO_PIN, pull_up=True)

# With gpiozero's internal pull-up:
# 1 = button pressed
# 0 = button released


# ==========================================
# 3. COLLECT DATA
# ==========================================

history = []

print("\n================================")
print("BUTTON MACHINE LEARNING PROJECT")
print("================================")

print("\nThe program will collect data for 60 seconds.")
print("Press and release the button several times.")
print("Try different patterns and durations.")
print("Keep your Raspberry Pi connected.\n")

start_time = time.time()

try:

    while time.time() - start_time < COLLECTION_TIME:

        # Read the physical button
        if button.is_pressed:
            state = 1
        else:
            state = 0

        # Store the observation
        history.append(state)

        elapsed = time.time() - start_time

        print(
            f"Time: {elapsed:5.1f}s | "
            f"Observed state: {state}"
        )

        time.sleep(SAMPLE_INTERVAL)

except KeyboardInterrupt:
    print("\nData collection stopped by user.")

finally:
    button.close()


# ==========================================
# 4. SAVE RAW DATA
# ==========================================

if len(history) < 20:
    print("\nNot enough data collected.")
    print("Run the program again and collect more data.")
    raise SystemExit

with open(CSV_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "time_step",
        "button_state"
    ])

    for i, state in enumerate(history):

        writer.writerow([i, state])

print(f"\nRaw data saved to {CSV_FILE}")

print("Total observations:", len(history))

print("Released readings:", history.count(0))
print("Pressed readings:", history.count(1))


# ==========================================
# 5. PREPARE THE MACHINE LEARNING DATA
# ==========================================

# Input:
# The previous five button readings.
#
# Target:
# The next button reading.

X = []
y = []

for i in range(WINDOW_SIZE, len(history)):

    previous_readings = history[
        i - WINDOW_SIZE:i
    ]

    next_state = history[i]

    X.append(previous_readings)
    y.append(next_state)

X = np.array(X)
y = np.array(y)

print("\nMachine learning dataset created.")

print("Input shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# 6. SPLIT INTO TRAINING AND TEST DATA
# ==========================================

# Use the first 80% for training.
# Use the last 20% for testing.
#
# This preserves the time order.

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 7. TRAIN THE MODEL
# ==========================================

if len(np.unique(y_train)) < 2:

    print("\nThe training data contains only one state.")
    print("The model needs both pressed and released readings.")
    print("Run the program again and press the button more.")

    raise SystemExit

model = LogisticRegression()

model.fit(X_train, y_train)

print("\nMachine learning model trained!")


# ==========================================
# 8. TEST THE MODEL
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n================================")
print("MODEL EVALUATION")
print("================================")

print(
    f"Model accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Released", "Pressed"],
        zero_division=0
    )
)


# ==========================================
# 9. COMPARE WITH A SIMPLE BASELINE
# ==========================================

# The baseline always predicts the most
# common state in the training data.

baseline = DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("\nBaseline accuracy:",
      round(baseline_accuracy * 100, 2), "%")

print("ML model accuracy:",
      round(accuracy * 100, 2), "%")

print(
    "\nCompare these two numbers to see "
    "whether the trained model improves "
    "on the simple baseline."
)


# ==========================================
# 10. SAVE THE TRAINED MODEL
# ==========================================

joblib.dump(model, MODEL_FILE)

print(f"\nTrained model saved as {MODEL_FILE}")


# ==========================================
# 11. MAKE A PREDICTION
# ==========================================

print("\n================================")
print("LIVE PREDICTION")
print("================================")

print(
    "Reading the current button state "
    "and predicting the next state..."
)

live_history = []

live_button = Button(
    GPIO_PIN,
    pull_up=True
)

try:

    # Collect five readings
    for i in range(WINDOW_SIZE):

        if live_button.is_pressed:
            state = 1
        else:
            state = 0

        live_history.append(state)

        time.sleep(SAMPLE_INTERVAL)

    # Predict the next state
    live_input = np.array(
        live_history
    ).reshape(1, -1)

    predicted_state = model.predict(
        live_input
    )[0]

    probabilities = model.predict_proba(
        live_input
    )[0]

    confidence = max(probabilities) * 100

    print("\nLast five readings:", live_history)

    if predicted_state == 1:
        print("Predicted next state: PRESSED")
    else:
        print("Predicted next state: RELEASED")

    print(
        f"Model confidence: {confidence:.2f}%"
    )

except KeyboardInterrupt:
    print("\nLive prediction stopped.")

finally:
    live_button.close()

print("\nProject completed!")
