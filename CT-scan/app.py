import gc
import os
from typing import Tuple, List, Dict, Generator
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.layers import (
    Dense, 
    Dropout, 
    GlobalAveragePooling2D, 
    BatchNormalization
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array, 
    load_img
)
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

class MedicalImageClassifier:
    def __init__(
        self,
        base_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 16,
        epochs: int = 50,
        threshold: float = 0.5,
        processing_batch: int = 200
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold = threshold
        self.processing_batch = processing_batch
        
        # Setup paths
        self.base_dir = base_dir
        self.data_path = os.path.join(base_dir, "images_001", "images")
        self.labels_file = os.path.join(base_dir, "CT-scan", "labels", "labels.csv")
        
        # Initialize GPU settings
        self.setup_gpu()
        
    def setup_gpu(self):
        """Configure GPU settings for optimal performance."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU setup completed. Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU setup failed: {e}")

    def prepare_data(self, df: pd.DataFrame, label_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data in batches to manage memory efficiently."""
        images = []
        labels = []
        
        print("Processing images...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                img_path = os.path.join(self.data_path, row['id'])
                if os.path.exists(img_path):
                    img = load_img(img_path, target_size=self.image_size)
                    img_array = img_to_array(img)
                    img_array = img_array.astype('float32') / 255.0
                    
                    label = row[label_columns].values.astype('float32')
                    
                    images.append(img_array)
                    labels.append(label)
                    
                    if len(images) % self.processing_batch == 0:
                        print(f"\nProcessed {len(images)} images...")
                        gc.collect()
                        
            except Exception as e:
                print(f"\nError processing image {row['id']}: {str(e)}")
                continue
        
        if not images:
            raise ValueError(f"No valid images found in {self.data_path}")
        
        return np.array(images), np.array(labels)
    
    @staticmethod
    def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
        """Implementation of Focal Loss for better handling of class imbalance."""
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = alpha * y_true * tf.math.pow((1 - y_pred), gamma)
            focal_loss = weight * cross_entropy
            
            return tf.keras.backend.mean(tf.keras.backend.sum(focal_loss, axis=-1))
        return focal_loss_fixed

    def verify_paths(self) -> bool:
        """Comprehensive path verification with detailed error reporting."""
        errors = []
        
        for path, desc in [
            (self.base_dir, "Base directory"),
            (self.data_path, "Images directory"),
            (self.labels_file, "Labels file")
        ]:
            if not os.path.exists(path):
                errors.append(f"{desc} not found: {path}")
                if os.path.exists(os.path.dirname(path)):
                    print(f"Contents of parent directory:")
                    try:
                        print("\n".join(f"- {item}" for item in os.listdir(os.path.dirname(path))))
                    except PermissionError:
                        print("Permission denied to list directory contents")
            elif not os.access(path, os.R_OK):
                errors.append(f"Cannot read {desc.lower()}: {path}")
        
        if errors:
            print("\nPath verification failed:")
            for error in errors:
                print(f"- {error}")
            return False
        
        print("All paths verified successfully!")
        return True

    def data_generator(
        self, 
        image_files: List[str], 
        labels_df: pd.DataFrame
    ) -> Generator:
        """Memory-efficient data generator with error handling."""
        for i in range(0, len(image_files), self.processing_batch):
            batch_files = image_files[i:i + self.processing_batch]
            batch_images = []
            batch_labels = []
            
            for img_file in batch_files:
                try:
                    img_path = os.path.join(self.data_path, img_file)
                    img = load_img(img_path, target_size=self.image_size)
                    img_array = img_to_array(img) / 255.0  # Normalize while loading
                    
                    img_labels = labels_df.loc[img_file].values
                    
                    batch_images.append(img_array)
                    batch_labels.append(img_labels)
                    
                except Exception as e:
                    print(f"\nError processing {img_file}: {str(e)}")
                    continue
            
            if batch_images:
                yield np.array(batch_images), np.array(batch_labels)
            
            gc.collect()

    def create_model(self, num_classes: int) -> Sequential:
        """Create an enhanced model architecture with additional features."""
        tf.keras.backend.clear_session()
        gc.collect()

        # Use distribution strategy if multiple GPUs are available
        strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) > 1 else None
        
        def build_model():
            base_model = ResNet50(
                weights="imagenet",
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
            
            # Fine-tune the last few layers
            for layer in base_model.layers[-30:]:
                layer.trainable = True
            
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                BatchNormalization(),
                Dense(1024, activation="relu"),
                Dropout(0.5),
                BatchNormalization(),
                Dense(512, activation="relu"),
                Dropout(0.4),
                BatchNormalization(),
                Dense(256, activation="relu"),
                Dropout(0.3),
                Dense(num_classes, activation="sigmoid")
            ])

            model.compile(
                optimizer=Adam(learning_rate=1e-4),
                loss=self.binary_focal_loss(),
                metrics=[
                    'accuracy',
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()
                ]
            )
            return model

        if strategy:
            with strategy.scope():
                return build_model()
        return build_model()

    def train_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        class_weights: Dict = None
    ) -> Tuple[Sequential, dict]:
        """Enhanced training procedure with fixed data generator."""
        model = self.create_model(y_train.shape[1])
        
        # Modify data generator configuration
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.2,
            fill_mode="nearest",
            brightness_range=[0.8, 1.2],
            rescale=None  # Remove rescaling since we already normalized the data
        )

        val_datagen = ImageDataGenerator(rescale=None)  # Remove rescaling

        # Convert class weights to sample weights
        sample_weights = np.ones(len(y_train))
        if class_weights:
            for class_idx, weight in class_weights.items():
                sample_weights[y_train[:, class_idx] == 1] = weight

        # Use fit with proper data flow
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=self.get_callbacks(),
            sample_weight=sample_weights,
            shuffle=True
        )
        
        return model, history

    def get_callbacks(self) -> List:
        """Get model callbacks."""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                "best_model.keras",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1
            ),
            TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_images=True
            )
        ]   

    def evaluate_model(
        self, 
        model: Sequential, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        label_names: List[str]
    ) -> dict:
        """Comprehensive model evaluation with detailed metrics."""
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > self.threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'f1_macro': f1_score(y_test, y_pred_binary, average='macro'),
            'precision_macro': precision_score(y_test, y_pred_binary, average='macro'),
            'recall_macro': recall_score(y_test, y_pred_binary, average='macro'),
            'classification_report': classification_report(
                y_test, 
                y_pred_binary, 
                target_names=label_names,
                zero_division=0
            )
        }
        
        # Print detailed results
        print("\nModel Performance Metrics:")
        print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
        print("\nDetailed Classification Report:")
        print(metrics['classification_report'])
        
        return metrics

def main():
    try:
        # Initialize classifier with proper path
        classifier = MedicalImageClassifier(
            base_dir=os.path.join("D:",os.sep,"IMP", "Agile_Avengers", "train2"),
            image_size=(224, 224),
            batch_size=16,
            epochs=50
        )
        
        if not classifier.verify_paths():
            print("Path verification failed. Please check your data directory structure.")
            return
        
        # Load data
        print("Loading and preparing data...")
        df = pd.read_csv(classifier.labels_file)
        
        if 'id' not in df.columns:
            print("Error: CSV must contain an 'id' column with image filenames")
            return
            
        print(f"Successfully loaded CSV with {len(df)} entries")
        
        # Define label columns
        LABEL_COLUMNS = ['Infiltration', 'Atelectasis', 'Effusion', 'No Finding']
        
        # Prepare data
        X, y = classifier.prepare_data(df, LABEL_COLUMNS)
        
        print(f"\nSuccessfully processed {len(X)} images")
        print(f"Image array shape: {X.shape}")
        print(f"Labels array shape: {y.shape}")
        
        # Calculate class weights
        class_weights = {}
        total_samples = len(y)
        for i, column in enumerate(LABEL_COLUMNS):
            positive_samples = np.sum(y[:, i])
            if positive_samples > 0:
                weight = (total_samples - positive_samples) / positive_samples
                class_weights[i] = min(weight, 100.0)
            else:
                class_weights[i] = 1.0
        
        # Split data with stratification
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y.sum(axis=1)
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp.sum(axis=1)
        )
        
        # Train and evaluate
        model, history = classifier.train_model(
            X_train, y_train, X_val, y_val, class_weights=class_weights
        )
        
        # Plot training history and evaluate
        plot_training_history(history)
        metrics = classifier.evaluate_model(model, X_test, y_test, LABEL_COLUMNS)
        
        # Save results
        model.save("final_model.keras")
        save_results(history, metrics, class_weights)
        
    except Exception as e:
        print(f"\nAn error occurred:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
    
    finally:
        gc.collect()

def plot_training_history(history):
    """Plot training history metrics"""
    import matplotlib.pyplot as plt
    
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def save_results(history, metrics, class_weights):
    """Save training results to file."""
    import json
    
    results = {
        'history': history.history,
        'metrics': metrics,
        'class_weights': class_weights
    }
    with open('training_results.json', 'w') as f:
        json.dump(results, f)
    
    print("\nModel and results saved successfully!")

if __name__ == "__main__":
    main()