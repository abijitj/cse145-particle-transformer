from glob import glob
import numpy as np
import tensorflow as tf
import keras as k
from tqdm import tqdm
from particle_transformer import ParticleTransformer
from particle_transformer_tagger import ParticleTransformerTagger
from embed import Embed 

from dataloading.dataset import create_tf_dataloader
import os

device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"

tf.random.set_seed(0)
np.random.seed(0)

# Hyperparameters
learning_rate = 3e-4
batch_size = 256

epochs = 5
steps_per_epoch = 1000

validation_split = 0.2

#print('cwd\n\n',os.getcwd())
#file_dict = {'validation':glob('C:/Users/andre/Desktop/UCSD/CSE145/cse145-particle-transformer/dataloading/JetClass_Pythia_val_5M/val_5M/*.root')}
file_dict = {'validation': glob('C:/Users/Abiji/Documents/ABClasses/CSE145/cse145-particle-transformer/final_particle_transformer/dataloading/JetClass_Pythia_val_5M/val_5M/*.root')}
data_config_file = './dataloading/dataconfig.yaml'
dataloader = create_tf_dataloader(file_dict, data_config_file)

# for test in dataloader:
#     print("THIS IS THE DATALOADER!!!")
#     print(len(test)) # 2
#     print("something1", test[0].shape, test[1].shape) # (2, 128), ()
#     break

train_dataset = dataloader.batch(batch_size) 
for test in train_dataset:
    print("THIS IS THE TRAIN DATASET!!!")
    print(len(test)) # 2
    print("something2", test[0].shape, test[1].shape) # (batch_size, 2, 128), (batch_size, )
    break

model = ParticleTransformer((2, 128), num_classes=10)
# model = ParticleTransformerTagger((1, 2, 128), )
# model = Embed([2, 128], [128, 512, 128], activation='gelu')

with tf.device(device):
    loss_function = k.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function
    )
    #validation_split=validation_split, validation_batch_size=batch_size, validation_steps=20
    #TODO validation data shouldn't be the same as training data
    
    model.fit(train_dataset, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
    # model.fit(train_dataset, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch, validation_data=train_dataset)