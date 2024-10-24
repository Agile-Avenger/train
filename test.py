import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
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
        # Fixed label mapping to be consistent with model output
        label_mapping = {'Normal': 0, 'Pneumonia': 1}
    
    valid_df = df.iloc[valid_entries].copy()
    
    return np.array(images), labels, label_mapping, valid_df

def create_model(input_shape=(224, 224, 3), num_classes=2):
    """Create the CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def predict_image(model, image_path, label_mapping=None):
    """Make prediction for a single image with improved error handling"""
    try:
        # If label_mapping is not provided, use default mapping
        if label_mapping is None:
            label_mapping = {'Normal': 0, 'Pneumonia': 1}
            
        # Create reverse mapping (from index to label)
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        
        # Load and preprocess image
        img_array = load_and_preprocess_image(image_path)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get prediction
        prediction = model.predict(img_array, verbose=0)
        
        # Get predicted class index and confidence
        predicted_class_idx = np.argmax(prediction)
        confidence = float(prediction[0][predicted_class_idx])
        
        # Convert predicted index to class label
        predicted_class = reverse_mapping.get(predicted_class_idx, 'Unknown')
        
        # Debug information
        print(f"Debug Info:")
        print(f"Prediction array: {prediction}")
        print(f"Predicted class index: {predicted_class_idx}")
        print(f"Label mapping: {label_mapping}")
        print(f"Reverse mapping: {reverse_mapping}")
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return "Error", 0.0

def train_pneumonia_model(data_directory, csv_path, validation_split=0.2, epochs=25):
    """Train the pneumonia classification model"""
    X, y, label_mapping, valid_df = prepare_data(data_directory, csv_path)
    print(f"Label mapping: {label_mapping}")
    print(f"Class distribution: {pd.Series(y).value_counts()}")
    
    y_categorical = to_categorical(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=validation_split, random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, fill_mode='nearest')
    
    model = create_model()
    
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=epochs, validation_data=(X_val, y_val))
    
    return model, history, label_mapping

def load_existing_model(model_path):
    """Load a pre-trained model from a saved .h5 file"""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

if __name__ == "__main__":
    train_directory = "D:/IMP/Agile_Avengers/train/archive(1)/chest_xray/train"
    test_image_path = "archive(2)/data/Tuberculosis/Tuberculosis-28.png"
    train_csv_path = "chest_xray_dataset_train.csv"
    
    choice = input("Do you want to train a new model or use an existing one? (train/load): ").strip().lower()
    
    model = None
    label_mapping = {'Normal': 0, 'Pneumonia': 1}  # Consistent label mapping
    
    if choice == "train":
        model, history, label_mapping = train_pneumonia_model(train_directory, train_csv_path)
        model.save('pneumonia_model.h5')
        print("Model trained and saved as pneumonia_model.h5")
    elif choice == "load":
        model_path = input("Enter the path to the saved model (.h5 file): ").strip()
        model = load_existing_model(model_path)
        if model is None:
            print("Failed to load model. Exiting...")
            exit(1)
        print(f"Loaded model from {model_path}")
    
    if model is not None:
        # Predict a single image
        predicted_class, confidence = predict_image(model, test_image_path, label_mapping)
        if predicted_class != "Error":
            print(f"\nResults:")
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.2f}")
        else:
            print("Prediction failed. Please check the error messages above.")