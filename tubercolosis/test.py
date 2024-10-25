import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
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
    print(f"Total images in CSV: {len(df)}")
    
    images = []
    labels = []
    valid_entries = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        img_path = os.path.join(data_directory, row['image_path'])
        
        if os.path.exists(img_path):
            try:
                img_array = load_and_preprocess_image(img_path)
                images.append(img_array)
                labels.append(row['binary_label'])
                valid_entries.append(index)
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
        else:
            print(f"Image not found: {img_path}")
    
    print(f"Successfully loaded {len(images)} images")
    
    labels = np.array(labels)
    if not np.issubdtype(labels.dtype, np.number):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    else:
        label_mapping = {'Normal': 0, 'Pneumonia': 1}
    
    valid_df = df.iloc[valid_entries].copy()
    
    return np.array(images), labels, label_mapping, valid_df

def create_model(input_shape=(224, 224, 3), num_classes=2):
    """Create a CNN model using ResNet50 as a base model"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze early layers
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

def predict_image(model, image_path, label_mapping=None):
    """Make prediction for a single image with improved error handling"""
    try:
        if label_mapping is None:
            label_mapping = {'Normal': 0, 'Pneumonia': 1}
            
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        
        img_array = load_and_preprocess_image(image_path)
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        confidence = float(prediction[0][predicted_class_idx])
        predicted_class = reverse_mapping.get(predicted_class_idx, 'Unknown')
        
        prediction_details = {
            reverse_mapping[i]: float(pred) for i, pred in enumerate(prediction[0])
        }
        
        return predicted_class, confidence, prediction_details
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return "Error", 0.0, {}

def train_pneumonia_model(data_directory, csv_path, validation_split=0.2, epochs=25):
    """Train the pneumonia classification model"""
    X, y, label_mapping, valid_df = prepare_data(data_directory, csv_path)
    print(f"Label mapping: {label_mapping}")
    print(f"Class distribution: {pd.Series(y).value_counts()}")
    
    # Calculate class weights
    total_samples = len(y)
    class_weights = {}
    for label_idx in range(len(np.unique(y))):
        class_count = np.sum(y == label_idx)
        class_weights[label_idx] = (1 / class_count) * (total_samples / 2)
    
    y_categorical = to_categorical(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, 
        test_size=validation_split, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0
    )
    
    model = create_model()
    
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
            'best_pneumonia_model.keras',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    return model, history, label_mapping

def save_model_with_mapping(model, label_mapping, base_filename):
    """Save both the model and its label mapping"""
    # Save model in .keras format
    model_path = f'{base_filename}.keras'
    model.save(model_path)
    
    # Save label mapping
    mapping_path = f'{base_filename}_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f)
    
    return model_path, mapping_path

def load_existing_model(model_path):
    """Load a pre-trained model from either .h5 or .keras format"""
    try:
        # Get the base path without extension
        base_path = os.path.splitext(model_path)[0]
        
        # Try loading the model
        if model_path.endswith('.h5'):
            model = load_model(model_path)
        elif model_path.endswith('.keras'):
            model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("Model file must have either .h5 or .keras extension")
        
        # Try loading the label mapping
        mapping_path = f'{base_path}_mapping.json'
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                label_mapping = json.load(f)
        else:
            print("Warning: Label mapping file not found, using default mapping")
            label_mapping = {'Normal': 0, 'Tubercolosis': 1}
        
        return model, label_mapping
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def train_pneumonia_model(data_directory, csv_path, validation_split=0.2, epochs=25):
    """Train the pneumonia classification model"""
    # ... (previous training code remains the same until saving)
    
    # Save model and mapping
    base_filename = 'best_pneumonia_model'
    model_path, mapping_path = save_model_with_mapping(model, label_mapping, base_filename)
    print(f"Model saved as {model_path}")
    print(f"Label mapping saved as {mapping_path}")
    
    return model, history, label_mapping

if __name__ == "__main__":
    train_directory = "D:/IMP/Agile_Avengers/train/archive(2)/data/Tuberculosis"
    test_image_path = "D:/IMP/Agile_Avengers/train/archive(2)/data/Tuberculosis/Tuberculosis-28.png"
    train_csv_path = "chest_xray_dataset_train.csv"
    
    choice = input("Do you want to train a new model or use an existing one? (train/load): ").strip().lower()
    
    model = None
    label_mapping = {'Normal': 0, 'Tubercolosis': 1}
    
    if choice == "train":
        model, history, label_mapping = train_pneumonia_model(train_directory, train_csv_path)
    elif choice == "load":
        model_path = input("Enter the path to the saved model (.keras or .h5 file): ").strip()
        model, label_mapping = load_existing_model(model_path)
        if model is None:
            print("Failed to load model. Exiting...")
            exit(1)
        print(f"Loaded model from {model_path}")
        if label_mapping:
            print(f"Using label mapping: {label_mapping}")
    
    if model is not None:
        predicted_class, confidence, prediction_details = predict_image(model, test_image_path, label_mapping)
        if predicted_class != "Error":
            print(f"\nResults:")
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.2f}")
            print("\nDetailed class probabilities:")
            for class_name, probability in prediction_details.items():
                print(f"{class_name}: {probability:.3f}")
        else:
            print("Prediction failed. Please check the error messages above.")