import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

# Load your data into a DataFrame
data = pd.read_csv("labels/train.csv")

# Exclude non-label columns (id, subj_id)
label_columns = data.columns[1:-1]

# Calculate the number of samples
n_samples = len(data)

# Initialize class weights dictionary
class_weights = {}

# Iterate over each label to calculate the class weight
for label in label_columns:
    # Count the number of positive samples for the current label
    n_positive = data[label].sum()

    # Apply the class weight formula
    class_weight = n_samples / (2 * n_positive) if n_positive > 0 else 0

    # Store the calculated class weight
    class_weights[label] = class_weight

# Print out the calculated class weights
print(class_weights)
