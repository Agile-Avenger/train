import os
import pandas as pd
from glob import glob

def create_dataset(base_path):
    """
    Create a dataset from images in normal and pneumonia folders (bacterial and viral).
    
    Parameters:
    base_path (str): Base directory containing 'normal' and 'pneumonia' folders
    
    Returns:
    pandas.DataFrame: DataFrame with image paths and labels
    """
    # Initialize lists to store data
    image_paths = []
    labels = []
    detailed_labels = []
    
    # Process normal images
    normal_path = os.path.join(base_path, 'NORMAL', '*')
    normal_images = glob(normal_path)
    image_paths.extend(normal_images)
    labels.extend([0] * len(normal_images))  # 0 for negative cases
    detailed_labels.extend(['normal'] * len(normal_images))
    
    # Process bacterial pneumonia images
    bacterial_path = os.path.join(base_path, 'PNEUMONIA', '*bacteria*')
    bacterial_images = glob(bacterial_path)
    image_paths.extend(bacterial_images)
    labels.extend([1] * len(bacterial_images))  # 1 for positive cases
    detailed_labels.extend(['bacterial'] * len(bacterial_images))
    
    # Process viral pneumonia images
    viral_path = os.path.join(base_path, 'PNEUMONIA', '*virus*')
    viral_images = glob(viral_path)
    image_paths.extend(viral_images)
    labels.extend([1] * len(viral_images))  # 1 for positive cases
    detailed_labels.extend(['viral'] * len(viral_images))
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'binary_label': labels,
        'binary_label_text': ['negative' if l == 0 else 'positive' for l in labels],
        'detailed_label': detailed_labels
    })
    
    # Add numerical detailed label
    label_mapping = {'normal': 0, 'bacterial': 1, 'viral': 2}
    df['detailed_label_num'] = df['detailed_label'].map(label_mapping)
    
    return df

def validate_image_files(base_dir):
    """
    Validate that the directory contains the expected types of image files.
    """
    pneumonia_dir = os.path.join(base_dir, 'PNEUMONIA')
    has_bacterial = any('bacteria' in f.lower() for f in os.listdir(pneumonia_dir))
    has_viral = any('virus' in f.lower() for f in os.listdir(pneumonia_dir))
    
    if not has_bacterial:
        print("Warning: No bacterial pneumonia images found!")
    if not has_viral:
        print("Warning: No viral pneumonia images found!")
    
    return has_bacterial or has_viral

def main():
    # Get base directory path from user
    base_dir = input("Enter the path to your base directory (containing 'normal' and 'pneumonia' folders): ")
    
    # Validate directory structure
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist!")
        return
    
    if not all(os.path.exists(os.path.join(base_dir, folder)) for folder in ['normal', 'pneumonia']):
        print("Error: Make sure both 'normal' and 'pneumonia' folders exist in the base directory!")
        return
    
    try:
        # Validate image files
        if not validate_image_files(base_dir):
            print("Error: No pneumonia images (bacterial or viral) found!")
            return
        
        # Create dataset
        df = create_dataset(base_dir)
        
        # Save to CSV
        csv_path = 'chest_xray_dataset_val.csv'
        df.to_csv(csv_path, index=False)
        
        # Display statistics
        print("\nDataset Statistics:")
        print("-" * 20)
        print(f"Total images: {len(df)}")
        print("\nBinary Classification:")
        print(f"Normal (negative) images: {len(df[df['binary_label'] == 0])}")
        print(f"Pneumonia (positive) images: {len(df[df['binary_label'] == 1])}")
        print("\nDetailed Classification:")
        print(f"Normal images: {len(df[df['detailed_label'] == 'normal'])}")
        print(f"Bacterial pneumonia images: {len(df[df['detailed_label'] == 'bacterial'])}")
        print(f"Viral pneumonia images: {len(df[df['detailed_label'] == 'viral'])}")
        print(f"\nCSV file saved as: {os.path.abspath(csv_path)}")
        
        # Display sample of the dataset
        print("\nFirst few rows of the dataset:")
        print(df.head())
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()