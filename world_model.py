
from gpiozero import Button
from time import sleep
from collections import defaultdict, Counter

# GPIO 17, using the internal pull-up resistor
button = Button(17, pull_up=True)

# Store the observed states
history = []

# Store transitions between consecutive states
transitions = defaultdict(Counter)

print("Simple World Model")
print("Press the button to collect data.")
print("Press Ctrl+C to stop.\n")

try:
    while True:

        # Observe the physical system
        if button.is_pressed:
            current_state = 1
        else:
            current_state = 0

        # Record the observation
        history.append(current_state)

        print("Observed state:", current_state)

        # Learn transitions from previous observations
        if len(history) >= 2:

            previous_state = history[-2]

            transitions[previous_state][current_state] += 1

        # Predict the next state
        if current_state in transitions:

            prediction = transitions[current_state].most_common(1)[0][0]

            print("Predicted next state:", prediction)

        else:
            print("Not enough data to predict yet.")

        print("--------------------")

        sleep(0.5)

except KeyboardInterrupt:
    print("\nStopped collecting data.")

    print("\nLearned transitions:")
    for state, outcomes in transitions.items():
        print(state, "->", dict(outcomes))

finally:
    button.close()
