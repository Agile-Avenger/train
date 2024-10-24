import numpy as np
import pandas as pd
import os
import tensorflow as tf
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
                labels.append(row['label'])
                valid_entries.append(index)
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
        else:
            print(f"Image not found: {img_path}")
    
    labels = np.array(labels)
    unique_label_texts = df['label_text'].unique()
    label_mapping = {i: label for i, label in enumerate(sorted(unique_label_texts))}
    valid_df = df.iloc[valid_entries].copy()
    
    return np.array(images), labels, label_mapping, valid_df

def create_model(input_shape=(224, 224, 3), num_classes=2):
    """Create a CNN model using ResNet50 as a base model"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model

def train_tb_model(data_directory, csv_path, train_size=0.7, val_size=0.15):
    """Train the TB detection model"""
    # Prepare data
    X, y, label_mapping, valid_df = prepare_data(data_directory, csv_path)
    y_categorical = to_categorical(y)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical,
        train_size=train_size,
        random_state=42,
        stratify=y
    )

    val_ratio = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_ratio,
        random_state=42,
        stratify=np.argmax(y_temp, axis=1)
    )

    # Print dataset information
    print("\nDataset Information:")
    print(f"Number of classes: {len(label_mapping)}")
    for label_idx, label_text in label_mapping.items():
        count = np.sum(y == label_idx)
        print(f"{label_text}: {count} images")
    
    print(f"\nSplit sizes:")
    print(f"Training set: {len(X_train)} images ({train_size*100:.1f}%)")
    print(f"Validation set: {len(X_val)} images ({val_size*100:.1f}%)")
    print(f"Test set: {len(X_test)} images ({(1-train_size-val_size)*100:.1f}%)")

    # Create data generator
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0
    )

    # Convert numpy arrays to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Configure datasets for performance
    BATCH_SIZE = 16
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # Calculate class weights
    total_samples = len(y)
    class_weights = {}
    for label_idx in range(len(label_mapping)):
        class_count = np.sum(y == label_idx)
        class_weights[label_idx] = (1 / class_count) * (total_samples / len(label_mapping))

    # Create and train model
    model = create_model(num_classes=len(label_mapping))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_tb_model.keras',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=val_dataset,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )

    # Evaluate on test set
    print("\nEvaluating on test set:")
    test_metrics = model.evaluate(test_dataset, verbose=2)
    metrics_names = model.metrics_names
    
    for name, value in zip(metrics_names, test_metrics):
        print(f"{name}: {value:.4f}")

    # Generate predictions for detailed metrics
    y_pred = model.predict(test_dataset, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes,
                              target_names=[label_mapping[i] for i in range(len(label_mapping))]))

    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
    print(conf_matrix)

    if len(label_mapping) == 2:
        fpr, tpr, _ = roc_curve(y_test_classes, y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        print(f"\nROC AUC: {roc_auc:.4f}")

    return model, history, label_mapping, (X_test, y_test)

def predict_image(model, image_path, label_mapping, threshold=0.7):
    """Make prediction for a single image"""
    img_array = load_and_preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    predicted_class = label_mapping[predicted_index] if confidence >= threshold else 'Uncertain'
    prediction_details = {
        label_mapping[i]: float(pred) for i, pred in enumerate(prediction[0])
    }

    return predicted_class, confidence, prediction_details

if __name__ == "__main__":
    data_directory = "D:/IMP/Agile_Avengers/train/archive(2)/data/Tuberculosis"
    csv_path = "chest_xray_tb_dataset.csv"

    model, history, label_mapping, test_data = train_tb_model(
        data_directory, 
        csv_path,
        train_size=0.7,
        val_size=0.15
    )
    
    model.save('tb_detection_model_final.h5')
    
    import json
    with open('label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)

    test_image_path = "D:/IMP/Agile_Avengers/train/archive(2)/data/Tuberculosis/Tuberculosis-20.png"
    predicted_class, confidence, prediction_details = predict_image(
        model, test_image_path, label_mapping, threshold=0.7
    )
    print(f"\nPrediction Results:")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")
    print("\nDetailed class probabilities:")
    for class_name, probability in prediction_details.items():
        print(f"{class_name}: {probability:.3f}")