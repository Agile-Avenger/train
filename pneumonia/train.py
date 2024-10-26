import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    img = load_img(image_path, target_size=target_size, color_mode='rgb')
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    
    # Add contrast enhancement
    mean = np.mean(img_array)
    std = np.std(img_array)
    img_array = (img_array - mean) / (std + 1e-7)
    
    return img_array

def prepare_data(data_directory, csv_path):
    """Prepare image data and labels"""
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        img_path = os.path.join(data_directory, row['image_path'])
        if os.path.exists(img_path):
            try:
                img_array = load_and_preprocess_image(img_path)
                if np.std(img_array) > 0.1:  # Filter out low contrast images
                    images.append(img_array)
                    labels.append(row['binary_label'])
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
    
    return np.array(images), np.array(labels)

def create_model(input_shape=(224, 224, 3), num_classes=2):
    """Create model with ResNet50 base"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze early layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_model(data_directory, csv_path, validation_split=0.2):
    """Train the model with simplified data handling"""
    # Load and prepare data
    X, y = prepare_data(data_directory, csv_path)
    y_categorical = to_categorical(y)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical,
        test_size=validation_split,
        random_state=42,
        stratify=y
    )
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_precision',
            mode='max',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Create and train model
    model = create_model()
    
    # Calculate class weights
    class_counts = np.bincount(y)
    total = np.sum(class_counts)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )
    
    return model, history

def predict_image(model, image_path, threshold=0.75):
    """Make prediction with confidence threshold"""
    img_array = load_and_preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    if confidence >= threshold:
        result = 'Pneumonia' if predicted_class == 1 else 'Normal'
    else:
        result = 'Uncertain'
    
    return result, confidence

def evaluate_model(model, test_directory, test_csv_path):
    """Evaluate model performance"""
    # Load test data
    X_test, y_test = prepare_data(test_directory, test_csv_path)
    y_test_categorical = to_categorical(y_test)
    
    # Evaluate
    print("Evaluating model...")
    results = model.evaluate(X_test, y_test_categorical, verbose=2)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=['Normal', 'Pneumonia']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))
    
    return results

# Example usage:
if __name__ == "__main__":
    train_directory = "D:/IMP/Agile_Avengers/train/archive(1)/chest_xray/train"
    test_directory = "D:/IMP/Agile_Avengers/train/archive(1)/chest_xray/test"
    train_csv_path = "chest_xray_dataset_train.csv"
    test_csv_path = "chest_xray_dataset_test.csv"

    model, history = train_model(train_directory, train_csv_path)
    
    # Save model
    model.save('pneumonia_model_improved.h5')
    
    # Evaluate model
    results = evaluate_model(model, test_directory, test_csv_path)
    # Single image prediction
    test_image_path = "archive(1)/chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg"
    predicted_class, confidence = predict_image(model, test_image_path, label_mapping, threshold=0.6)
    print(f"\nPredicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")
