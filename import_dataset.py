# import_dataset.py
import kagglehub

# Download the latest version of the dataset
path = kagglehub.dataset_download("sovitrath/diabetic-retinopathy-224x224-gaussian-filtered")

print("Path to dataset files:", path)