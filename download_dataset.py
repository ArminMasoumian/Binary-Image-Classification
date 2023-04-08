import os
import urllib.request

# URL of the binary image classification dataset
DATASET_URL = "https://example.com/binary-image-classification-dataset.zip"

# Path where the dataset will be saved
DATASET_DIR = "data"

# Create the directory if it doesn't exist
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Download the dataset and save it to the dataset directory
urllib.request.urlretrieve(DATASET_URL, os.path.join(DATASET_DIR, "binary-image-classification-dataset.zip"))

# Unzip the downloaded file
os.system(f"unzip {os.path.join(DATASET_DIR, 'binary-image-classification-dataset.zip')} -d {DATASET_DIR}")

# Delete the downloaded zip file
os.remove(os.path.join(DATASET_DIR, "binary-image-classification-dataset.zip"))
