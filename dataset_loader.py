import os
import pandas as pd
import cv2  # OpenCV for image processing
import numpy as np

# Dictionary mapping dataset types to their respective label filenames & image folders
DATASET_CONFIG = {
    "Training": {"label_file": "training_labels.csv", "image_folder": "training_words"},
    "Testing": {"label_file": "testing_labels.csv", "image_folder": "testing_words"},
    "Validation": {"label_file": "validation_labels.csv", "image_folder": "validation_words"},
}

def load_dataset(base_path, dataset_type):
    """
    Loads images and labels from the dataset directory.

    :param base_path: The root directory of the dataset
    :param dataset_type: The dataset type ('Training', 'Testing', 'Validation')
    :return: List of preprocessed images and labels DataFrame
    """
    # Paths for labels and images
    dataset_path = os.path.join(base_path, dataset_type)
    if dataset_type not in DATASET_CONFIG:
        raise ValueError(f"❌ Invalid dataset type: {dataset_type}")

    label_file = os.path.join(dataset_path, DATASET_CONFIG[dataset_type]["label_file"])
    image_folder = os.path.join(dataset_path, DATASET_CONFIG[dataset_type]["image_folder"])

    # Check if dataset folders exist
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    print(f"Loading {dataset_type} dataset from: {dataset_path}")
    
    # Load CSV file
    df = pd.read_csv(label_file)

    # Store preprocessed images
    images = []

    # Iterate over image filenames in CSV
    for index, row in df.iterrows():
        image_filename = row["IMAGE"]  
        image_path = os.path.join(image_folder, image_filename)  # Use correct folder

        if not os.path.exists(image_path):
            print(f"⚠️ Missing image: {image_path}")
            continue  # Skip missing images

        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        # Preprocessing
        image = cv2.resize(image, (128, 128))  # Resize
        image = cv2.GaussianBlur(image, (5, 5), 0)  # Reduce noise
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)  # Thresholding
        image = image / 255.0  # Normalize

        images.append(np.array(image))

    return np.array(images), df

def load_all_datasets(base_path):
    """
    Loads Training, Testing, and Validation datasets.

    :param base_path: Root directory of the dataset
    :return: Dictionary with images and labels for each dataset
    """
    datasets = {}
    for dataset_type in DATASET_CONFIG.keys():
        try:
            images, labels_df = load_dataset(base_path, dataset_type)
            datasets[dataset_type] = {"images": images, "labels": labels_df}
            print(f"✅ Loaded {len(images)} images from {dataset_type}.")
        except FileNotFoundError as e:
            print(e)

    return datasets

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define dataset root path
dataset_root = os.path.join(script_dir, "Doctor's Handwritten Prescription BD dataset")

# Load all datasets
datasets = load_all_datasets(dataset_root)
