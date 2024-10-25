import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from typing import Tuple, Dict, Union

# Constants from original code
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.5
LABEL_COLUMNS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural Thickening", "Pneumonia", "Pneumothorax", "Pneumoperitoneum",
    "Pneumomediastinum", "Subcutaneous Emphysema", "Tortuous Aorta",
    "Calcification of the Aorta", "No Finding"
]

def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = IMAGE_SIZE) -> Union[np.ndarray, None]:
    """
    Load and preprocess a single image for prediction
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image (height, width)
        
    Returns:
        Preprocessed image array or None if loading fails
    """
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def predict_conditions(model, image_path: str, confidence_threshold: float = THRESHOLD) -> Dict[str, float]:
    """
    Predict medical conditions from a single chest X-ray image
    
    Args:
        model: Loaded Keras model
        image_path: Path to the image file
        confidence_threshold: Threshold for positive prediction
        
    Returns:
        Dictionary mapping conditions to their prediction probabilities
    """
    try:
        # Load and preprocess the image
        processed_image = load_and_preprocess_image(image_path)
        if processed_image is None:
            return {}

        # Make prediction
        predictions = model.predict(processed_image, verbose=0)[0]
        
        # Create dictionary of predictions
        prediction_dict = {
            condition: float(prob) 
            for condition, prob in zip(LABEL_COLUMNS, predictions)
            if prob >= confidence_threshold
        }
        
        # If no conditions meet the threshold, return "No Finding" with its probability
        if not prediction_dict and "No Finding" in LABEL_COLUMNS:
            no_finding_index = LABEL_COLUMNS.index("No Finding")
            prediction_dict["No Finding"] = float(predictions[no_finding_index])
            
        return prediction_dict

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {}

def batch_predict(model, image_directory: str, csv_path: str = None) -> pd.DataFrame:
    """
    Perform batch prediction on multiple images
    
    Args:
        model: Loaded Keras model
        image_directory: Directory containing images
        csv_path: Optional path to CSV file with image metadata
        
    Returns:
        DataFrame with predictions for each image
    """
    results = []
    
    # If CSV provided, use it to get image paths
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        image_files = df['id'].tolist()  # Assuming 'id' is the column with image filenames
    else:
        # Otherwise, scan directory for image files
        image_files = [f for f in os.listdir(image_directory) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        if os.path.exists(image_path):
            predictions = predict_conditions(model, image_path)
            predictions['image_file'] = image_file
            results.append(predictions)
        else:
            print(f"Image not found: {image_path}")
    
    return pd.DataFrame(results)

def main():
    """Main function to run the testing script"""
    print("Medical Image Testing Script")
    
    # Get model path from user
    model_path = input("Enter the path to the saved model (.h5 file): ").strip()
    
    # Load the model
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Get test mode from user
    mode = input("Choose test mode (single/batch): ").strip().lower()
    
    if mode == 'single':
        # Test single image
        image_path = input("Enter the path to the test image: ").strip()
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
            
        predictions = predict_conditions(model, image_path)
        
        print("\nPrediction Results:")
        print("-" * 50)
        for condition, probability in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            print(f"{condition}: {probability:.2%}")
            
    elif mode == 'batch':
        # Batch testing
        image_dir = input("Enter the directory containing test images: ").strip()
        csv_path = input("Enter path to CSV file with image metadata (optional, press Enter to skip): ").strip()
        
        if not os.path.exists(image_dir):
            print(f"Directory not found: {image_dir}")
            return
            
        if csv_path and not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return
            
        results_df = batch_predict(model, image_dir, csv_path if csv_path else None)
        
        # Save results
        output_path = "prediction_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        # Display summary
        print("\nPrediction Summary:")
        print("-" * 50)
        for condition in LABEL_COLUMNS:
            if condition in results_df.columns:
                positive_cases = (results_df[condition] >= THRESHOLD).sum()
                print(f"{condition}: {positive_cases} positive predictions")
    
    else:
        print("Invalid mode selected. Please choose 'single' or 'batch'.")

if __name__ == "__main__":
    main()