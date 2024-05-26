import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Flatten
from tensorflow.keras.models import Model
import os
pip! install keras
from keras.layers import *
from qkeras import *

filepath = './retrain_test_1/JetClass/Pythia/test_20M/'
x_particles = [] 
max_elements = 5
x_jets = []
y = []
long_string = "./retrain_test_1/JetClass/Pythia/test_20M//['HToCC_101.root', 'HToWW4Q_116.root', 'HToBB_112.root', 'TTBarLep_106.root', 'WToQQ_105.root', 'ZToQQ_105.root', 'WToQQ_102.root', 'HToBB_119.root', 'WToQQ_113.root', 'HToGG_103.root', 'TTBarLep_116.root', 'HToBB_114.root', 'ZToQQ_104.root', 'ZJetsToNuNu_119.root', 'TTBar_112.root', 'ZToQQ_101.root', 'WToQQ_117.root', 'HToBB_106.root', 'WToQQ_104.root', 'HToGG_110.root', 'WToQQ_116.root', 'HToWW2Q1L_118.root', 'HToWW4Q_107.root', 'WToQQ_110.root', 'ZJetsToNuNu_105.root', 'HToBB_108.root', 'HToCC_117.root', 'ZJetsToNuNu_113.root', 'TTBar_117.root', 'ZToQQ_112.root', 'HToWW4Q_119.root', 'HToWW2Q1L_100.root', 'HToWW4Q_117.root', 'HToCC_118.root', 'TTBarLep_109.root', 'TTBar_104.root', 'ZJetsToNuNu_112.root', 'HToCC_107.root', 'TTBar_111.root', 'HToWW2Q1L_105.root', 'ZToQQ_114.root', 'TTBarLep_113.root', 'HToCC_109.root', 'HToGG_113.root', 'HToCC_100.root', 'HToWW4Q_104.root', 'TTBar_107.root', 'TTBar_100.root', 'TTBar_106.root', 'ZToQQ_117.root', 'HToCC_106.root', 'WToQQ_118.root', 'TTBarLep_105.root', 'HToWW2Q1L_116.root', 'ZToQQ_103.root', 'TTBarLep_101.root', 'ZJetsToNuNu_116.root', 'HToWW4Q_100.root', 'ZJetsToNuNu_103.root', 'HToBB_115.root', 'TTBarLep_107.root', 'ZJetsToNuNu_104.root', 'HToGG_105.root', 'ZJetsToNuNu_101.root', 'HToWW2Q1L_112.root', 'WToQQ_109.root', 'HToGG_109.root', 'WToQQ_106.root', 'WToQQ_108.root', 'TTBar_109.root', 'HToBB_100.root', 'TTBar_113.root', 'HToBB_111.root', 'WToQQ_111.root', 'WToQQ_101.root', 'HToBB_107.root', 'HToCC_116.root', 'HToGG_115.root', 'ZJetsToNuNu_100.root', 'ZJetsToNuNu_102.root', 'HToWW4Q_105.root', 'ZJetsToNuNu_108.root', 'TTBar_101.root', 'WToQQ_119.root', 'HToGG_100.root', 'TTBar_102.root', 'HToBB_102.root', 'HToBB_118.root', 'TTBar_116.root', 'HToWW2Q1L_113.root', 'HToWW2Q1L_119.root', 'HToWW2Q1L_108.root', 'HToWW4Q_103.root', 'ZJetsToNuNu_115.root', 'HToCC_104.root', 'HToGG_111.root', 'HToWW4Q_114.root', 'TTBar_108.root', 'HToGG_118.root', 'HToWW2Q1L_110.root', 'HToCC_111.root', 'HToWW2Q1L_101.root', 'TTBarLep_119.root', 'HToWW4Q_101.root', 'ZToQQ_113.root', 'TTBar_103.root', 'ZJetsToNuNu_118.root', 'HToBB_109.root', 'TTBarLep_117.root', 'HToGG_107.root', 'HToWW2Q1L_103.root', 'HToWW4Q_110.root', 'HToWW4Q_118.root', 'HToGG_116.root', 'ZToQQ_100.root', 'TTBarLep_114.root', 'HToCC_103.root', 'ZToQQ_107.root', 'HToWW2Q1L_104.root', 'ZToQQ_102.root', 'TTBarLep_115.root', 'HToGG_101.root', 'WToQQ_103.root', 'ZJetsToNuNu_107.root', 'ZJetsToNuNu_109.root', 'TTBarLep_102.root', 'HToWW2Q1L_115.root', 'HToWW2Q1L_102.root', 'HToGG_117.root', 'HToWW4Q_115.root', 'HToGG_119.root', 'HToBB_117.root', 'TTBar_110.root', 'ZToQQ_108.root', 'HToWW2Q1L_107.root', 'HToGG_114.root', 'HToBB_101.root', 'HToWW2Q1L_111.root', 'HToCC_108.root', 'TTBar_115.root', 'HToWW4Q_111.root', 'HToCC_105.root', 'HToGG_108.root', 'HToWW4Q_102.root', 'ZJetsToNuNu_110.root', 'HToWW4Q_106.root', 'ZToQQ_115.root', 'HToGG_104.root', 'TTBar_118.root', 'TTBarLep_118.root', 'ZToQQ_116.root', 'HToWW2Q1L_114.root', 'ZJetsToNuNu_111.root', 'HToWW4Q_109.root', 'ZToQQ_110.root', 'HToBB_103.root', 'HToBB_116.root', 'HToCC_115.root', 'HToCC_102.root', 'HToGG_106.root', 'ZToQQ_106.root', 'HToWW2Q1L_117.root', 'TTBar_105.root', 'HToBB_105.root', 'TTBarLep_103.root', 'HToWW2Q1L_106.root', 'TTBarLep_111.root', 'WToQQ_112.root', 'ZJetsToNuNu_117.root', 'TTBarLep_108.root', 'TTBarLep_110.root', 'ZToQQ_119.root', 'ZJetsToNuNu_106.root', 'TTBarLep_104.root', 'ZJetsToNuNu_114.root', 'WToQQ_115.root', 'WToQQ_114.root', 'HToBB_113.root', 'HToCC_119.root', 'TTBarLep_100.root', 'ZToQQ_111.root', 'HToGG_102.root', 'HToWW4Q_113.root', 'HToBB_110.root', 'HToCC_113.root', 'HToBB_104.root', 'WToQQ_107.root', 'HToWW4Q_108.root', 'ZToQQ_118.root', 'HToCC_110.root', 'TTBar_114.root', 'TTBar_119.root', 'HToCC_114.root', 'ZToQQ_109.root', 'TTBarLep_112.root', 'HToGG_112.root', 'WToQQ_100.root', 'HToWW2Q1L_109.root', 'HToCC_112.root', 'HToWW4Q_112.root']"

def read_file(
        filepath,
        max_num_particles=128,
        particle_features=['part_pt', 'part_eta', 'part_phi', 'part_energy'],
        jet_features=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'],
        labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']):
    def _pad(a, maxlen, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x
    
    table = uproot.open(filepath)['tree'].arrays()

    p4 = vector.zip({'px': table['part_px'],
                     'py': table['part_py'],
                     'pz': table['part_pz'],
                     'energy': table['part_energy']})
    table['part_pt'] = p4.pt
    table['part_eta'] = p4.eta
    table['part_phi'] = p4.phi

    temp_x_particles = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in particle_features], axis=1)
    temp_x_jets = np.stack([ak.to_numpy(table[n]).astype('float32') for n in jet_features], axis=1)
    temp_y = np.stack([ak.to_numpy(table[n]).astype('int') for n in labels], axis=1)
    return temp_x_particles, temp_x_jets, temp_y

try:
    filenames_list = eval(long_string.split("//")[1])  # Extract the list of fi>
    for filename in filenames_list:
        if len(x_particles) >= max_elements:
            print(f"Reached the limit of {max_elements} elements. Stopping further processing.")
            break
        print("Loading file: ", filename)
        temp_x_particles, temp_x_jets, temp_y = read_file(f"{filepath}/{filename}")
        x_particles.append(temp_x_particles)
        x_jets.append(temp_x_jets)
        y.append(temp_y)
except Exception as e:
    print("Error occurred:", e)

# print("x_particles shape:", x_particles)
# print("x_jets shape:", x_jets)  
# print("y shape:", y) 
print(len(x_particles))
print(len(x_jets))
print(len(y))
x_particles = np.array(x_particles)
x_jets = np.array(x_jets)
y = np.concatenate(y, axis=0)
# print("all_x_particles shape:", x_particles.shape)  # Should be (num_jets, num_particle_features, max_num_particles)
# print("all_x_jets shape:", x_jets.shape)  # Should be (num_jets, num_jet_features)
# print("all_y shape:", y.shape)  # Should be (num_jets, num_classes)


num_particles = 128
num_particle_features = 4
num_interaction_features = 4

# It uses a particle embedding of a dimension d = 128, encoded from the input particle features using a 3-layer MLP with (128, 512, 128) nodes each layer with GELU nonlinearity, and LN is used in between for normalization. 
d = 128
d_prime = 8

num_particles = x_particles.shape[0]
num_jets = x_particles.shape[1]
index3 = x_particles.shape[3]


def preprocess_particles(x_particles, num_particles, num_particle_features):
    # Reshape x_particles to (num_jets * num_particles, num_particle_features) (N,C)
    particles = x_particles.reshape(num_jets * max_elements * index3, num_particle_features)
    return particles

def preprocess_interaction(particles_matrix, num_particle_features): # U matrix (N, N, C'), # pairwise interactions between all particles in the jet and feeding numebrs
    # initalize the shape with all 0
    N = num_jets * max_elements * index3
    C_prime = 2 * num_particle_features
    interaction = np.zeros((N, N, C_prime))
    # pairwise interactions between all particles in the jet and feeding numebrs
    matrix1 = particles_matrix
    matrix2 = particles_matrix
    for i in range(N):
        for j in range(N):
            interaction[i, j] = np.concatenate((matrix1[i], matrix2[j]))
    return interaction

# Get particles matrix (N,C)
particles_matrix = preprocess_particles(x_particles, num_particles, num_particle_features)
print(particles_matrix.shape)
# Get interaction matrix (N, N, C')
interaction_matrix = preprocess_interaction(particles_matrix, num_particle_features)

# Check shapes
print("Particle Matrix shape:", particles_matrix.shape)
print("Interaction Matrix shape:", interaction_matrix.shape)

particle_input = layers.Input(shape=(None, num_particle_features))  # Shape: (None, None, num_particle_features)
interaction_input = layers.Input(shape=(None, None, num_interaction_features))  # Shape: (None, None, None, num_interaction_features)

####### should we use quantlization here? - yes
# 3-layer MLP with (128, 512, 128) nodes each layer with GELU nonlinearity
# particle_embedding = Dense(128, activation='gelu')(particle_input)
# particle_embedding = Dense(512, activation='gelu')(particle_embedding)
# particle_embedding = Dense(d, activation='gelu')(particle_embedding)
# particle_embedding = LayerNormalization()(particle_embedding)
#### compiler error!!!! not able to import qkeras
particle_embedding = QDense(128,
                            kernel_quantizer=quantized_bits(4),
                            bias_quantizer=quantized_bits(4))(particle_input)
particle_embedding = QActivation("gelu", name="gelu")(particle_embedding)
particle_embedding = QDense(512,
                            kernel_quantizer=quantized_bits(4),
                            bias_quantizer=quantized_bits(4))(particle_embedding)
particle_embedding = QActivation("gelu", name="gelu")(particle_embedding)
particle_embedding = QDense(d,
                            kernel_quantizer=quantized_bits(4),
                            bias_quantizer=quantized_bits(4))(particle_embedding)
particle_embedding = QActivation("gelu", name="gelu")(particle_embedding)
particle_embedding = LayerNormalization()(particle_embedding)

# using a 4-layer pointwise 1D convolution with (64, 64, 64, 8) channels with GELU nonlinearity and batch normalization in between to yield a d′ = 8 dimensional interaction matrix.

#### compiler error!!!! not able to import qkeras
# interaction_embedding = tf.keras.layers.Conv1D(64, 1, activation='gelu')(interaction_input)
# interaction_embedding = tf.keras.layers.Conv1D(64, 1, activation='gelu')(interaction_embedding)
# interaction_embedding = tf.keras.layers.Conv1D(64, 1, activation='gelu')(interaction_embedding)
# interaction_embedding = tf.keras.layers.Conv1D(d_prime, 1, activation='gelu')(interaction_embedding)
# interaction_embedding = tf.keras.layers.BatchNormalization()(interaction_embedding)
interaction_embedding = QConv1D(64, 1, 
                                kernel_quantizer=quantized_bits(4),
                                bias_quantizer=quantized_bits(4))(interaction_input)
interaction_embedding = QActivation("gelu", name="gelu")(interaction_embedding)
interaction_embedding = QConv1D(64, 1, 
                                kernel_quantizer=quantized_bits(4),
                                bias_quantizer=quantized_bits(4))(interaction_embedding)
interaction_embedding = QActivation("gelu", name="gelu")(interaction_embedding)
interaction_embedding = QConv1D(64, 1, 
                                kernel_quantizer=quantized_bits(4),
                                bias_quantizer=quantized_bits(4))(interaction_embedding)
interaction_embedding = QActivation("gelu", name="gelu")(interaction_embedding)
interaction_embedding = QConv1D(64, 1, 
                                kernel_quantizer=quantized_bits(4),
                                bias_quantizer=quantized_bits(4))(interaction_embedding)
interaction_embedding = QActivation("gelu", name="gelu")(interaction_embedding)
interaction_embedding = tf.keras.layers.BatchNormalization()(interaction_embedding)

# num_attention_blocks = 8
# for _ in range(8):
#     particle_embedding = particle_attention_block(particle_embedding, interaction_embedding, d, d_prime, num_heads=8)


    
