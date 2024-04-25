import torch
import yaml
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from networks.example_ParticleTransformer import get_model
from weaver.utils.data.config import DataConfig

# = DataConfig()
#with open(, 'r') as f:
data_config = DataConfig.load('data/JetClass/JetClass_full.yaml')
print(data_config)



model, model_info = get_model(data_config)
print(type(model))
#print(type(ParticleTransformer(test)))