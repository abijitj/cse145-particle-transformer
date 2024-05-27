import numpy as np
import tensorflow as tf
import keras as k
from tqdm import tqdm
from particle_transformer import ParticleTransformer

device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"

tf.random.set_seed(0)
np.random.seed(0)

# Hyperparameters
learning_rate = 3e-4

model = ParticleTransformer()

with tf.device(device):
    loss_function = k.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function
    )

    #x, y = get_training_data()
    
    model.fit(x, y, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch, validation_split=.2, validation_batch_size=batch_size, validation_steps=10)
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # generate from the model
    context = np.zeros((1, 1), dtype='float_')
    print(decode(model.generate(context, max_new_tokens=500)[0].numpy().tolist()))