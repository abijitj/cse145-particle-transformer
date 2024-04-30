import torch
import yaml
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from networks.example_ParticleTransformer import get_model
from weaver.utils.data.config import DataConfig
from torch.autograd import Variable
from pytorch2keras import pytorch_to_keras
import numpy as np

# THIS SCRIPT DOES NOT WORK SUCCESSFULLY BUT DEMOS THE WORK PUT INTO
# AUTOMATED TRANSLATION

data_config = DataConfig.load('data/JetClass/JetClass_full.yaml')
print(data_config)

pytorch_model, model_info = get_model(data_config, for_inference=True)
print(model_info.keys())
print(model_info['input_names'])
#print(data_config)
input_shape = [shape for shape in model_info['input_shapes'].values()]
print(input_shape)

pf_points_var = Variable(torch.FloatTensor(np.random.uniform(0, 1, (1, 2, 128))))
pf_features_var = Variable(torch.FloatTensor(np.random.uniform(0, 1, (1, 17, 128))))
pf_vectors_var = Variable(torch.FloatTensor(np.random.uniform(0, 1, (1, 4, 128))))
pf_mask_var = Variable(torch.FloatTensor(np.random.uniform(0, 1, (1, 1, 128))))
#input_var = Variable(torch.FloatTensor(input_np))

# we should specify shape of the input tensor

k_model = pytorch_to_keras(pytorch_model,[pf_points_var, pf_features_var, pf_vectors_var, pf_mask_var], input_shapes=input_shape, verbose=True)

#input_np = 
#input_torch = Variable()
#keras_model = pytorch_to_keras(pytorch_model, )
#print(type(model))
#print(type(ParticleTransformer(test)))