import gc
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)
from tqdm import tqdm

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
PROCESSING_BATCH_SIZE = 200
NUM_CLASSES = 20
THRESHOLD = 0.5
data_path = "images"
train_labels_file = "labels/train.csv"

# Define the column names for consistency
LABEL_COLUMNS = [
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


def load_and_preprocess_image(image_path, target_size=IMAGE_SIZE):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None


def prepare_data(data_directory, csv_path):
    """Load and prepare the dataset"""
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    valid_indices = []

    print("Loading and preprocessing images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(data_directory, row["id"])
        if os.path.exists(img_path):
            img_array = load_and_preprocess_image(img_path)
            if img_array is not None:
                images.append(img_array)
                label = row[LABEL_COLUMNS].values
                labels.append(label)
                valid_indices.append(idx)

    return np.array(images), np.array(labels), valid_indices


def create_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """Create the model architecture"""
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = False

    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )
    return model


def train_model(X_train, y_train, X_val, y_val, class_weights):
    """Train the model with the given data"""
    model = create_model()

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator()

    # Create data generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

    val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            "best_model.keras", monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]

    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // BATCH_SIZE
    validation_steps = len(X_val) // BATCH_SIZE

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    return model, history


def main():
    # Load and prepare data
    print("Loading data...")
    X, y, valid_indices = prepare_data(data_path, train_labels_file)

    # Split the data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )

    # Convert class weights dict to array format
    class_weights = dict(
        enumerate(
            [
                5.955926146515783,
                23.64066193853428,
                16.420361247947454,
                46.728971962616825,
                6.195786864931846,
                28.571428571428573,
                26.109660574412533,
                312.5,
                4.004805766920304,
                15.220700152207002,
                10.460251046025105,
                20.70393374741201,
                56.81818181818182,
                15.748031496062993,
                181.8181818181818,
                833.3333333333334,
                45.248868778280546,
                54.34782608695652,
                120.48192771084338,
                0.8050881571532083,
            ]
        )
    )

    # Train the model
    print("Training model...")
    model, history = train_model(X_train, y_train, X_val, y_val, class_weights)

    # Save the final model
    model.save("final_resnet50_model.h5")
    print("Model saved as final_resnet50_model.h5")

    # Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > THRESHOLD).astype(int)

    # Calculate metrics for multi-label classification
    precision = precision_score(y_test, y_pred_binary, average="weighted")
    recall = recall_score(y_test, y_pred_binary, average="weighted")
    f1 = f1_score(y_test, y_pred_binary, average="weighted")

    print("\nModel Performance Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Generate detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_binary, target_names=LABEL_COLUMNS))

    # Clear memory
    gc.collect()


if __name__ == "__main__":
    main()
