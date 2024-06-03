import os
import numpy as np
import torch
from weaver.utils.data.config import DataConfig
from networks.example_ParticleTransformer import get_model
from weaver.utils.data.config import DataConfig

import torchvision.models as models

import uproot
import os

# Load data configuration
data_config = DataConfig.load('data/JetClass/JetClass_full.yaml')

# Load model
model, model_info = get_model(data_config)
print(type(model))
#print(type(ParticleTransformer(test)))

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()




def load_data_from_root_files(root_dir):
    data = []
    root_files = [file for file in os.listdir(root_dir) if file.endswith('.root')]
    for root_file in root_files:
        file_path = os.path.join(root_dir, root_file)
        with uproot.open(file_path) as file:
            # Assuming you have a single tree called "tree" containing your data
            tree = file["tree"]
            # Assuming you have a single branch called "branch" containing your float point numbers
            branch_data = tree.array("branch")
            data.extend(branch_data)
    return data

# Path to the directory containing the ROOT files
data_path = "/home/particle/particle_transformer/retrain_test_1/JetClass/Pythia/test_20M/"

# Load data from ROOT files
data = load_data_from_root_files(data_path)

# Now 'data' contains all the float point numbers from the ROOT files in the specified directory

def preprocess_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()  # Read data from the file

    # Preprocess the data (tokenization, padding, etc.)
    # Example preprocessing steps:
    processed_data = data[~np.isnan(data)]  # Tokenize, pad, etc.

    return processed_data

test_data_path = "/home/particle/particle_transformer/retrain_test_1/JetClass/Pythia/test_20M/*"
test_data = preprocess_data(test_data_path)

# Convert the preprocessed test data to PyTorch tensors
test_inputs_tensor = torch.tensor(test_data)
