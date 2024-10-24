import numpy as np
import pandas as pd
import os
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    img = load_img(image_path, target_size=target_size, color_mode='rgb')
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

def prepare_data(data_directory, csv_path):
    """Prepare image data and labels from directory structure and CSV"""
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    valid_entries = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        img_path = os.path.join(data_directory, row['image_path'])
        if os.path.exists(img_path):
            try:
                img_array = load_and_preprocess_image(img_path)
                images.append(img_array)
                labels.append(row['binary_label'])  # Binary label: 0 (Normal), 1 (Pneumonia)
                valid_entries.append(index)
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
        else:
            print(f"Image not found: {img_path}")
    
    labels = np.array(labels)
    if not np.issubdtype(labels.dtype, np.number):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    else:
        label_mapping = {0: 'Normal', 1: 'Pneumonia'}
    
    valid_df = df.iloc[valid_entries].copy()
    
    return np.array(images), labels, label_mapping, valid_df

def create_model(input_shape=(224, 224, 3), num_classes=2):
    """Create a CNN model using ResNet50 as a base model"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_pneumonia_model(data_directory, csv_path, validation_split=0.2):
    X, y, label_mapping, valid_df = prepare_data(data_directory, csv_path)
    y_categorical = to_categorical(y)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=validation_split, random_state=42, stratify=y)

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # Set up data augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Calculate class weights to handle imbalance
    class_weights = {0: 1.5, 1: 1.0}  # Higher weight for the Normal class

    # Create and train the model
    model = create_model()

    # Train the model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=25,
        validation_data=(X_val, y_val),
        steps_per_epoch=len(X_train) // 32,
        class_weight=class_weights,  # Apply class weights
        verbose=2
    )

    return model, history, label_mapping

def predict_image(model, image_path, label_mapping, threshold=0.5):
    """Make prediction for a single image with a custom threshold"""
    img_array = load_and_preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence >= threshold:  # Adjust threshold to reduce false positives/negatives
        predicted_class = label_mapping[predicted_index]
    else:
        predicted_class = 'Uncertain'  # Adding uncertainty handling if below threshold

    return predicted_class, confidence

def evaluate_model(model, test_directory, test_csv_path, label_mapping):
    X_test, y_test, _, test_df = prepare_data(test_directory, test_csv_path)
    y_test_categorical = to_categorical(y_test)

    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=[label_mapping[i] for i in range(len(label_mapping))]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))

    return test_loss, test_accuracy

# Example usage:
if __name__ == "__main__":
    train_directory = "D:/IMP/Agile_Avengers/train/archive(1)/chest_xray/train"
    test_directory = "D:/IMP/Agile_Avengers/train/archive(1)/chest_xray/test"
    train_csv_path = "chest_xray_dataset_train.csv"
    test_csv_path = "chest_xray_dataset_test.csv"

    model, history, label_mapping = train_pneumonia_model(train_directory, train_csv_path)
    model.save('pneumonia_model_resnet50.h5')

    # Evaluate model
    test_loss, test_accuracy = evaluate_model(model, test_directory, test_csv_path, label_mapping)

    # Single image prediction
    test_image_path = "archive(1)/chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg"
    predicted_class, confidence = predict_image(model, test_image_path, label_mapping, threshold=0.6)
    print(f"\nPredicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")
