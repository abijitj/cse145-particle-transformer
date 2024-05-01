import os
import numpy as np
import torch
from weaver.utils.data.config import DataConfig
from networks.example_ParticleTransformer import get_model
from dataloader import read_file

# Load data configuration
data_config = DataConfig.load('data/JetClass/JetClass_full.yaml')

# Load model
model, model_info = get_model(data_config)
model.eval()

data_path = "/home/particle/particle_transformer/retrain_test_1/JetClass/Pythia/test_20M/"

# Initialize lists to store data from all files
all_x_particles = []
all_x_jets = []
all_y = []

# Loop through all files in the directory
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith('.root'):
            # Construct the full path to the current ROOT file
            file_path = os.path.join(root, file)
            
            # Read data from the current file
            x_particles, x_jets, y = read_file(file_path)
            
            # Append data from the current file to the lists
            all_x_particles.append(x_particles)
            all_x_jets.append(x_jets)
            all_y.append(y)

# Concatenate data from all files
x_particles = np.concatenate(all_x_particles)
x_jets = np.concatenate(all_x_jets)
y = np.concatenate(all_y)

# Preprocess the data (replace with actual preprocessing steps)
# Here, we assume x_particles and x_jets are already preprocessed based on model_info
# Replace this with your actual preprocessing code
x_particles_normalized = (x_particles - model_info["particle_mean"]) / model_info["particle_std"]
x_jets_normalized = (x_jets - model_info["jet_mean"]) / model_info["jet_std"]

# Convert preprocessed data to PyTorch tensors
x_particles_tensor = torch.tensor(x_particles_normalized, dtype=torch.float32)
x_jets_tensor = torch.tensor(x_jets_normalized, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Move tensors to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_particles_tensor = x_particles_tensor.to(device)
x_jets_tensor = x_jets_tensor.to(device)
y_tensor = y_tensor.to(device)

# Perform inference
with torch.no_grad():
    outputs = model(x_particles_tensor, x_jets_tensor)

# Calculate accuracy
_, predicted_labels = torch.max(outputs, 1)
correct_predictions = torch.sum(predicted_labels == torch.argmax(y_tensor, dim=1)).item()
total_samples = y_tensor.size(0)
accuracy = correct_predictions / total_samples
print(f"Accuracy: {accuracy}")



