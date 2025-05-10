import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import hog

# Function to extract HOG features and compute summary statistics
def extract_hog_summary_features(image):
    # Compute HOG features without multichannel parameter
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)

    # Compute summary statistics
    mean_hog = np.mean(hog_features)
    var_hog = np.var(hog_features)
    max_hog = np.max(hog_features)
    min_hog = np.min(hog_features)

    # Return the summary statistics
    return [mean_hog, var_hog, max_hog, min_hog]

# Function to extract features from the entire dataset and store them in a DataFrame
def extract_features_to_csv(image_paths, labels, physical_features, output_csv):
    features_list = []
    
    for i, image_path in enumerate(image_paths):
        # Load the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, (128, 128))  # Resize to a fixed size
            hog_summary_features = extract_hog_summary_features(image)  # Extract HOG summary features
            
            # Combine HOG summary features with physical features
            combined_features = np.concatenate((hog_summary_features, physical_features[i], [labels[i]]))
            features_list.append(combined_features)

    # Define feature columns
    feature_columns = ['HOG_Mean', 'HOG_Variance', 'HOG_Max', 'HOG_Min'] + \
                      ['BMI', 'Height', 'Weight', 'PulseRate', 'BloodGroup', 
                       'Endometrium', 'RegularExercise', 'BloodPressure', 
                       'Haemoglobin', 'CycleLength', 'NoOfAbortions', 
                       'HairLoss', 'Pimples', 'Label']
    
    # Create a pandas DataFrame
    df = pd.DataFrame(features_list, columns=feature_columns)
    
    # Save the DataFrame to CSV
    try:
        df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")
    except PermissionError:
        print(f"Permission denied: Unable to write to {output_csv}. Please check if the file is open or if you have the required permissions.")

# Example usage
pcos_images_path = "C:\\Users\\sarth\\Downloads\\archive (4)\\dataset\\pcos"
normal_images_path = "C:\\Users\\sarth\\Downloads\\archive (4)\\dataset\\normal"

# Get image paths
pcos_image_paths = [os.path.join(pcos_images_path, img) for img in os.listdir(pcos_images_path)]
normal_image_paths = [os.path.join(normal_images_path, img) for img in os.listdir(normal_images_path)]

# Create labels for the images
pcos_labels = [0] * len(pcos_image_paths)  # Label 0 for PCOS
normal_labels = [1] * len(normal_image_paths)  # Label 1 for Normal

# Combine PCOS and Normal images and labels
all_image_paths = pcos_image_paths + normal_image_paths
all_labels = pcos_labels + normal_labels

# Assuming physical features are stored in this format for each image (BMI, Height, Weight, PulseRate, BloodGroup, Endometrium, RegularExercise, BloodPressure, Haemoglobin, CycleLength, NoOfAbortions, HairLoss, Pimples)
# Example physical features for demonstration; you can customize the values accordingly
physical_features_pcos = [
    [22.5, 165, 68, 80, 'A+', 10, 1, 120, 13.5, 28, 2, 1, 3] for _ in range(len(pcos_image_paths))
]

physical_features_normal = [
    [21.0, 160, 60, 75, 'O+', 8, 1, 115, 14.0, 30, 0, 0, 1] for _ in range(len(normal_image_paths))
]

# Combine the physical features for both PCOS and Normal images
all_physical_features = physical_features_pcos + physical_features_normal

# Output CSV file path
output_csv_file = 'C:\\Users\\sarth\\Downloads\\pcos_image_hog_physical_features.csv'

# Extract features and save to CSV
extract_features_to_csv(all_image_paths, all_labels, all_physical_features, output_csv_file)
