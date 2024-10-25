import gc
import os
from typing import Tuple, List
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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
NUM_CLASSES = 20
THRESHOLD = 0.5
PROCESSING_BATCH_SIZE = 200

# Define paths using os.path.join for platform independence
BASE_DIR = os.path.join("D:", os.sep, "IMP", "Agile_Avengers", "train")
DATA_PATH = os.path.join(BASE_DIR, "images_001", "images")
TRAIN_LABELS_FILE = os.path.join(BASE_DIR, "CT-scan", "labels", "train.csv")

# Label columns
LABEL_COLUMNS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural Thickening", "Pneumonia", "Pneumothorax", "Pneumoperitoneum",
    "Pneumomediastinum", "Subcutaneous Emphysema", "Tortuous Aorta",
    "Calcification of the Aorta", "No Finding"
]

def verify_paths():
    """Verify that all necessary paths exist and are accessible"""
    errors = []
    
    print("\nChecking paths:")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"TRAIN_LABELS_FILE: {TRAIN_LABELS_FILE}\n")
    
    if not os.path.exists(BASE_DIR):
        errors.append(f"Base directory not found: {BASE_DIR}")
        parent_dir = os.path.dirname(BASE_DIR)
        if os.path.exists(parent_dir):
            print(f"Contents of parent directory ({parent_dir}):")
            try:
                print("\n".join(f"- {item}" for item in os.listdir(parent_dir)))
            except PermissionError:
                print("Permission denied to list parent directory contents")
    
    if not os.path.exists(DATA_PATH):
        errors.append(f"Images directory not found: {DATA_PATH}")
    elif not os.access(DATA_PATH, os.R_OK):
        errors.append(f"Cannot read images directory: {DATA_PATH}")
    
    labels_dir = os.path.dirname(TRAIN_LABELS_FILE)
    if not os.path.exists(labels_dir):
        errors.append(f"Labels directory not found: {labels_dir}")
    elif not os.access(labels_dir, os.R_OK):
        errors.append(f"Cannot read labels directory: {labels_dir}")
    
    if not os.path.exists(TRAIN_LABELS_FILE):
        errors.append(f"Labels file not found: {TRAIN_LABELS_FILE}")
    elif not os.access(TRAIN_LABELS_FILE, os.R_OK):
        errors.append(f"Cannot read labels file: {TRAIN_LABELS_FILE}")
    
    if errors:
        print("\nPath verification failed:")
        for error in errors:
            print(f"- {error}")
        print("\nTroubleshooting suggestions:")
        print("1. Verify that the drive letter is correct")
        print("2. Check if the directories are created with correct names")
        print("3. Ensure you have read permissions for all directories")
        print("4. Try using absolute paths instead of relative paths")
        print(f"5. Current working directory: {os.getcwd()}")
        return False
    
    print("All paths verified successfully!")
    return True

def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def prepare_data_efficient(data_directory: str, csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Memory-efficient data preparation"""
    print(f"Loading CSV from: {csv_path}")
    print(f"Looking for images in: {data_directory}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV with {len(df)} entries")
        print(f"CSV columns: {df.columns.tolist()}")
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")
    
    images = []
    labels = []
    valid_indices = []
    
    missing_columns = [col for col in LABEL_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in CSV: {missing_columns}")
    
    print("Loading and preprocessing images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(data_directory, row["id"])
        if os.path.exists(img_path):
            img_array = load_and_preprocess_image(img_path)
            if img_array is not None:
                images.append(img_array)
                label = row[LABEL_COLUMNS].values.astype('float32')
                labels.append(label)
                valid_indices.append(idx)
        
        # Process in batches to manage memory
        if len(images) % PROCESSING_BATCH_SIZE == 0:
            gc.collect()
    
    if not images:
        raise ValueError(f"No valid images found in {data_directory}")
    
    print(f"Found {len(images)} valid images out of {len(df)} entries")
    return np.array(images), np.array(labels), valid_indices

def create_model(input_shape: Tuple[int, int, int] = (224, 224, 3), num_classes: int = NUM_CLASSES) -> Sequential:
    """Create the model architecture"""
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()]
    )
    return model

def train_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, class_weights: dict) -> Tuple[Sequential, dict]:
    """Train the model with the given data"""
    model = create_model()

    # Modified data augmentation setup
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Convert data to float32
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    y_train = y_train.astype('float32')
    y_val = y_val.astype('float32')

    # Custom training data generator
    def train_generator():
        batch_size = BATCH_SIZE
        num_samples = len(X_train)
        indices = np.arange(num_samples)
        while True:
            np.random.shuffle(indices)
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]
                
                # Apply augmentation to the batch
                augmented = next(train_datagen.flow(
                    batch_X, 
                    batch_y,
                    batch_size=len(batch_indices),
                    shuffle=False
                ))
                
                yield augmented[0], augmented[1]


    # Custom validation data generator
    def val_generator():
        batch_size = BATCH_SIZE
        num_samples = len(X_val)
        indices = np.arange(num_samples)
        while True:
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X_val[batch_indices]
                batch_y = y_val[batch_indices]
                yield batch_X, batch_y

    # Create tf.data.Dataset objects
    train_dataset = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    )

    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
    ]

    steps_per_epoch = len(X_train) // BATCH_SIZE
    validation_steps = len(X_val) // BATCH_SIZE

    # Train the model using tf.data.Dataset
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2
    )

    return model, history

# Rest of the code remains the same...
def main():
    print("Starting medical image processing...")
    
    if not verify_paths():
        print("Please fix the path issues and try again.")
        return
    
    try:
        # Load and prepare data
        print("Loading data...")
        X, y, valid_indices = prepare_data_efficient(DATA_PATH, TRAIN_LABELS_FILE)

        # Split the data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1, random_state=42
        )

        # Define class weights
        class_weights = dict(enumerate([
            5.96, 23.64, 16.42, 46.73, 6.20, 28.57, 26.11, 312.5,
            4.00, 15.22, 10.46, 20.70, 56.82, 15.75, 181.82, 833.33,
            45.25, 54.35, 120.48, 0.81
        ]))

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

        # Calculate and print metrics
        print("\nModel Performance Metrics:")
        print_metrics(y_test, y_pred_binary)

    except Exception as e:
        print(f"\nError occurred during execution:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nPlease verify your directory structure and file permissions.")
        return
    
    finally:
        gc.collect()

if __name__ == "__main__":
    main()