import os
import pandas as pd
from glob import glob

def create_dataset(base_path):
    """
    Create a dataset from images in normal and tuberculosis folders.
    
    Parameters:
    base_path (str): Base directory containing 'normal' and 'tuberculosis' folders
    
    Returns:
    pandas.DataFrame: DataFrame with image paths and labels
    """
    # Initialize lists to store data
    image_paths = []
    labels = []
    
    # Process normal images
    normal_path = os.path.join(base_path, 'Normal', '*')
    normal_images = glob(normal_path)
    image_paths.extend(normal_images)
    labels.extend([0] * len(normal_images))  # 0 for normal cases
    
    # Process tuberculosis images
    tb_path = os.path.join(base_path, 'Tuberculosis', '*')
    tb_images = glob(tb_path)
    image_paths.extend(tb_images)
    labels.extend([1] * len(tb_images))  # 1 for tuberculosis cases
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels,
        'label_text': ['normal' if l == 0 else 'tuberculosis' for l in labels]
    })
    
    return df

def validate_directory(base_dir):
    """
    Validate that the directory contains the expected folder structure.
    """
    required_folders = ['NORMAL', 'TUBERCULOSIS']
    for folder in required_folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Error: '{folder}' folder not found in {base_dir}")
            return False
        if not os.listdir(folder_path):
            print(f"Warning: '{folder}' folder is empty!")
    return True

def main():
    # Get base directory path from user
    base_dir = input("Enter the path to your base directory (containing 'NORMAL' and 'TUBERCULOSIS' folders): ")
    
    # Validate directory structure
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist!")
        return
    
    if not validate_directory(base_dir):
        return
    
    try:
        # Create dataset
        df = create_dataset(base_dir)
        
        # Save to CSV
        csv_path = 'chest_xray_tb_dataset.csv'
        df.to_csv(csv_path, index=False)
        
        # Display statistics
        print("\nDataset Statistics:")
        print("-" * 20)
        print(f"Total images: {len(df)}")
        print(f"Normal images: {len(df[df['label'] == 0])}")
        print(f"Tuberculosis images: {len(df[df['label'] == 1])}")
        print(f"\nCSV file saved as: {os.path.abspath(csv_path)}")
        
        # Display sample of the dataset
        print("\nFirst few rows of the dataset:")
        print(df.head())
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()