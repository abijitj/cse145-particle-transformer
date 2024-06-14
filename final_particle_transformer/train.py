  
from glob import glob
import numpy as np
import tensorflow as tf
# from tensorflow import keras as k 

import keras as k
from particle_transformer import ParticleTransformer 
import traceback 

from dataloading.dataset import create_tf_dataloader
import os

# tf.compat.v1.disable_eager_execution()

if __name__ == '__main__':
    try:
        device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"

        tf.random.set_seed(0)
        # np.random.seed(0)

        # Hyperparameters
        learning_rate = 3e-3 #5e-4
        batch_size = 96 #16 #192 #96

        epochs = 15 #2000
        steps_per_epoch = 500 #1 #2000 #500
        validation_steps = 70 #1 #150 #70
        
        print(tf.config.list_physical_devices(), device)

        data_labels = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q', 'TTBar', 'TTBarLep', 'WToQQ', 'ZToQQ', 'ZJetsToNuNu']

        training_data_root = '/home/particle/particle_transformer/retrain_test_1/JetClass/Pythia/train_100M/'
        train_file_dict = {data_label: glob(training_data_root + data_label + "*.root") for data_label in data_labels}
        validation_file_dict = {'validation': glob('/home/particle/particle_transformer/retrain_test_1/JetClass/Pythia/val_5M/*.root')}
        data_config_file = './dataloading/dataconfig.yaml'

        train_dataloader = create_tf_dataloader(train_file_dict, data_config_file)
        validation_dataloader = create_tf_dataloader(validation_file_dict, data_config_file)
        

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

            print("1:")
            model.fit(train_dataset, 
                    epochs=epochs, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_data=validation_dataset, 
                    validation_steps=validation_steps, 
                    verbose=1
            )
            model.summary()
            model.save('keras_ParT.tf', save_format="tf")
            print("2:")
            

    except Exception as e:
        # print(e)
        traceback.print_exc()
        log_file = open('./error.log', 'w')
        log_file.truncate(0)
        log_file.write(str(e))
        log_file.close()
