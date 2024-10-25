import pandas as pd

# Read the dataset
data = pd.read_csv("test.csv")

# List of columns representing diseases
disease_columns = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
    "Pneumoperitoneum",
    "Pneumomediastinum",
    "Subcutaneous Emphysema",
    "Tortuous Aorta",
    "Calcification of the Aorta",
    "No Finding",
]


# Create a function to combine diseases into a single string
def combine_diseases(row):
    diseases = [disease for disease in disease_columns if row[disease] == 1]
    return ", ".join(diseases) if diseases else "No Disease"


# Apply the function to each row
data["Diseases"] = data.apply(combine_diseases, axis=1)

# Create the final dataframe with two columns: image name and diseases
final_df = data[["id", "Diseases"]]

# Save the result to a new CSV file
final_df.to_csv("test0.csv", index=False)

# Display the final dataframe
print(final_df)
