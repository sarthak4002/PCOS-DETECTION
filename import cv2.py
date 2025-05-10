import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import kurtosis, skew
import os

# Feature extraction function
def extract_features(image):
    features = []
    
    # 1. Histogram Equalization (Preprocessing)
    image = cv2.equalizeHist(image)
    
    # 2. Texture Features (GLCM)
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    features.extend([contrast, energy, homogeneity])
    
    # 3. Statistical Features
    mean_val = np.mean(image)
    var_val = np.var(image)
    skewness = skew(image.flatten())
    kurt_val = kurtosis(image.flatten())
    features.extend([mean_val, var_val, skewness, kurt_val])
    
    # 4. Shape Features (Edge detection + Hu Moments)
    edges = cv2.Canny(image, 100, 200)
    moments = cv2.moments(edges)
    hu_moments = cv2.HuMoments(moments).flatten()
    features.extend(hu_moments)
    
    # 5. Local Binary Pattern (LBP)
    lbp = local_binary_pattern(image, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 10), range=(0, 9), density=True)
    features.extend(lbp_hist)
    
    return features

# Function to extract features from the entire dataset and store them in a DataFrame
def extract_features_to_csv(image_paths, labels, physical_features, output_csv):
    features_list = []
    
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if image is not None:
            image = cv2.resize(image, (128, 128))  # Resize to 128x128
            features = extract_features(image)  # Extract features
            
            # Ensure physical features exist for this index
            if i < len(physical_features):
                features.extend(physical_features[i])  # Add the physical features
            
            # Append the label as the last column
            features.append(labels[i])
            features_list.append(features)
    
    # Define feature columns including physical attributes
    feature_columns = ['Contrast', 'Energy', 'Homogeneity', 'Mean', 'Variance', 'Skewness', 'Kurtosis'] + \
                      [f'HuMoment_{i}' for i in range(7)] + \
                      [f'LBP_{i}' for i in range(9)] + \
                      ['BMI', 'Height', 'Weight', 'PulseRate', 'BloodGroup', 'Endometrium', 
                       'RegularExercise', 'BloodPressure', 'Haemoglobin', 'CycleLength', 
                       'NoOfAbortions', 'HairLoss', 'Pimples', 'Label']
    
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
pcos_labels = ['PCOS'] * len(pcos_image_paths)
normal_labels = ['Normal'] * len(normal_image_paths)

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
output_csv_file = 'C:\\Users\\sarth\\Downloads\\pcos_image_physical_features.csv'

# Extract features and save to CSV
extract_features_to_csv(all_image_paths, all_labels, all_physical_features, output_csv_file)
