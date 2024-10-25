import gc
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Load the saved model
model = tf.keras.models.load_model("final_efficientnet_model.h5")

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
PROCESSING_BATCH_SIZE = 200  # Reduced for memory efficiency
NUM_CLASSES = 20  # Adjust based on your dataset
TEST_SIZE = 0.2  # 20% of data for test set
VAL_SIZE = 0.1  # 10% of remaining data for validation set

# Paths
data_path = "images"
train_labels_file = "labels/train.csv"


def load_and_preprocess_data(labels_file, batch_size=PROCESSING_BATCH_SIZE):
    df = pd.read_csv(labels_file)
    disease_columns = df.columns[1:-1].tolist()
    num_samples = len(df)
    all_images, all_labels = [], []

    for start_idx in tqdm(
        range(0, num_samples, batch_size), desc=f"Processing {labels_file}"
    ):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_df = df.iloc[start_idx:end_idx]
        batch_images, batch_labels = [], []

        for _, row in batch_df.iterrows():
            try:
                img_path = os.path.join(data_path, row["id"])
                img = image.load_img(img_path, target_size=IMAGE_SIZE)
                img_array = image.img_to_array(img) / 255.0  # Normalize
                label = row[disease_columns].values.astype(np.float32)

                batch_images.append(img_array)
                batch_labels.append(label)
            except Exception as e:
                print(f"Error processing {row['id']}: {str(e)}")
                continue

        all_images.extend(batch_images)
        all_labels.extend(batch_labels)
        gc.collect()

    return np.array(all_images), np.array(all_labels)


# Assuming X_test and y_test are already loaded
X, y = load_and_preprocess_data(train_labels_file)

# Split data into train, test, and validation sets
# First split the data into training + validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)

# Then split the remaining training + validation set into actual train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=VAL_SIZE, random_state=42
)

# If not, reload X_test and y_test from your preprocessed dataset

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert to binary predictions

# Calculate metrics
precision = precision_score(y_test, y_pred_binary, average="macro")
recall = recall_score(y_test, y_pred_binary, average="macro")
f1 = f1_score(y_test, y_pred_binary, average="macro")

# Display the metrics
print("\nTest Set Metrics:")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
