import torch
import uproot
import os
import numpy as np

# Load model and data
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from networks.example_ParticleTransformer import get_model
from weaver.utils.data.config import DataConfig

# Load data configuration
data_config = DataConfig.load('data/JetClass/JetClass_full.yaml')

# Load model
model, model_info = get_model(data_config)
model.eval()

# Function to load data from ROOT files
def load_data_from_root_files(root_dir):
    data = {}
    root_files = [file for file in os.listdir(root_dir) if file.endswith('.root')]
    for root_file in root_files:
        file_path = os.path.join(root_dir, root_file)
        with uproot.open(file_path) as file:
            tree = file[tree_name]
            for branch_name, branch_array in tree.arrays().items():
                if branch_name not in data:
                    data[branch_name] = []
                data[branch_name].extend(branch_array)
    return data

# Function to preprocess data
def preprocess_data(data):
    cleaned_data = np.array(data)  # Convert to numpy array
    cleaned_data = cleaned_data[~np.isnan(cleaned_data)]  # Remove NaN values
    cleaned_data = np.clip(cleaned_data, a_min=0, a_max=None)  # Clip outliers to a reasonable range
    return cleaned_data

# Path to the directory containing the ROOT files
data_path = "/home/particle/particle_transformer/retrain_test_1/JetClass/Pythia/test_20M/"
data = load_data_from_root_files(data_path)

# Preprocess the data
processed_data = preprocess_data(data)

# Convert the preprocessed test data to PyTorch tensor
test_inputs_tensor = torch.tensor(processed_data, dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_inputs_tensor = test_inputs_tensor.to(device)

# Perform inference
with torch.no_grad():
    outputs = model(test_inputs_tensor)

# Load ground truth labels (assuming they are stored separately)
ground_truth_path = "/home/particle/particle_transformer/retrain_test_1/JetClass/Pythia/val_5M/"
ground_truth_labels = load_data_from_root_files(ground_truth_path)

# Calculate accuracy
_, predicted_labels = torch.max(outputs, 1)
correct_predictions = torch.sum(predicted_labels == torch.tensor(ground_truth_labels)).item()
total_samples = len(ground_truth_labels)
accuracy = correct_predictions / total_samples
print(f"Accuracy: {accuracy}")

