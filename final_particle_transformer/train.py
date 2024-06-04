  
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


if __name__ == '__main__':
    try:
        device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"


        tf.random.set_seed(0)
        np.random.seed(0)

        # Hyperparameters
        learning_rate = 3e-5
        batch_size = 16 #192

        epochs = 15
        steps_per_epoch = 2000
        validation_steps = 150
        
        print(tf.config.list_physical_devices(), device)

        data_labels = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q', 'TTBar', 'TTBarLep', 'WToQQ', 'ZToQQ', 'ZJetsToNuNu']

        training_data_root = '/home/particle/particle_transformer/retrain_test_1/JetClass/Pythia/train_100M/'
        train_file_dict = {data_label: glob(training_data_root + data_label + "*.root") for data_label in data_labels}
        validation_file_dict = {'validation': glob('/home/particle/particle_transformer/retrain_test_1/JetClass/Pythia/val_5M/*.root')}
        data_config_file = './dataloading/dataconfig.yaml'

        train_dataloader = create_tf_dataloader(train_file_dict, data_config_file)
        validation_dataloader = create_tf_dataloader(validation_file_dict, data_config_file)
        
        # counts= {0: 0, 1: 0, 2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
        # i = 0
        # for test in train_dataloader:
        #     print("THIS IS THE DATALOADER!!!")
        #     #print(len(test)) # 2
        #     #print("something1", test[0].shape, test[1].shape) # (2, 128), ()
        #     print(test[1])
        #     counts[int(test[1])] += 1
        #     i += 1

        #     if i == 100000:
        #         break

        # print('counts', counts)

        def mapping(pf_features, pf_vectors, pf_mask, label):
            return (pf_features, pf_vectors, pf_mask), label 

        train_dataset = train_dataloader.map(mapping).batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        validation_dataset = validation_dataloader.map(mapping).batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)

        model = ParticleTransformer((17, 128), num_classes=10)

        for layer in model.layers:
            print(layer.name, layer.trainable)

        with tf.device(device):
            print("Using ", device)
            loss_function = k.losses.SparseCategoricalCrossentropy(from_logits=True)
            model.compile(
                optimizer=k.optimizers.Adam(learning_rate=learning_rate),
                loss=loss_function,
                metrics=['accuracy']
            )
            #TODO validation data shouldn't be the same as training data
            
            print("1:")
            model.fit(train_dataset, 
                    epochs=epochs, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_data=validation_dataset, 
                    validation_steps=validation_steps, 
                    verbose=1
            )
            print("2:")

    except Exception as e:
        print(e)
        log_file = open('./error.log', 'w')
        log_file.truncate(0)
        log_file.write(str(e))
        log_file.close()
