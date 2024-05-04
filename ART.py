from artlearn import ART1
import numpy as np

# Define the parameters for the ART1 network
num_categories = 3
num_features = 8  # 8-bit binary representation for each alphabet
rho = 0.5  # vigilance parameter

# Create an instance of the ART1 network
art = ART1(num_categories, num_features, rho)

# Define binary patterns representing alphabets (A, B, C)
input_patterns = [
    [0, 1, 0, 0, 0, 0, 0, 1],  # A: 01000001
    [0, 1, 0, 0, 0, 0, 1, 0],  # B: 01000010
    [0, 1, 0, 0, 0, 0, 1, 1]   # C: 01000011
]

# Fit the network to the input patterns
art.fit(input_patterns)

# Define a function to classify a new pattern
def classify_alphabet(pattern):
    category = art.predict(np.array(pattern).reshape(1, -1))
    if category == 0:
        return "A"
    elif category == 1:
        return "B"
    elif category == 2:
        return "C"
    else:
        return "Unknown"

# User input for a new pattern
print("Enter the 8-bit binary pattern for the new alphabet:")
new_pattern = []
for i in range(num_features):
    while True:
        value = input(f"Enter bit {i + 1}: ")
        if len(value) == 1 and value in ['0', '1']:
            new_pattern.append(int(value))
            break
        else:
            print("Please enter a single binary digit (0 or 1).")

# Classify the user-input pattern
result = classify_alphabet(new_pattern)
print(f"The new pattern represents alphabet {result}.")
