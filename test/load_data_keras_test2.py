import os
import numpy as np
#import awkward as ak # type: ignore

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers

# using dataloader to get np array
from dataloader import read_file
filepath = 'PATH TO TEST DATASET'
x_particles = [] 
x_jets = []
y = []

long_string = "./retrain_test_1/JetClass/Pythia/test_20M//['HToCC_101.root', 'HToWW4Q_116.root', 'HToBB_112.root', 'TTBarLep_106.root', 'WToQQ_105.root', 'ZToQQ_105.root', 'WToQQ_102.root', 'HToBB_119.root', 'WToQQ_113.root', 'HToGG_103.root', 'TTBarLep_116.root', 'HToBB_114.root', 'ZToQQ_104.root', 'ZJetsToNuNu_119.root', 'TTBar_112.root', 'ZToQQ_101.root', 'WToQQ_117.root', 'HToBB_106.root', 'WToQQ_104.root', 'HToGG_110.root', 'WToQQ_116.root', 'HToWW2Q1L_118.root', 'HToWW4Q_107.root', 'WToQQ_110.root', 'ZJetsToNuNu_105.root', 'HToBB_108.root', 'HToCC_117.root', 'ZJetsToNuNu_113.root', 'TTBar_117.root', 'ZToQQ_112.root', 'HToWW4Q_119.root', 'HToWW2Q1L_100.root', 'HToWW4Q_117.root', 'HToCC_118.root', 'TTBarLep_109.root', 'TTBar_104.root', 'ZJetsToNuNu_112.root', 'HToCC_107.root', 'TTBar_111.root', 'HToWW2Q1L_105.root', 'ZToQQ_114.root', 'TTBarLep_113.root', 'HToCC_109.root', 'HToGG_113.root', 'HToCC_100.root', 'HToWW4Q_104.root', 'TTBar_107.root', 'TTBar_100.root', 'TTBar_106.root', 'ZToQQ_117.root', 'HToCC_106.root', 'WToQQ_118.root', 'TTBarLep_105.root', 'HToWW2Q1L_116.root', 'ZToQQ_103.root', 'TTBarLep_101.root', 'ZJetsToNuNu_116.root', 'HToWW4Q_100.root', 'ZJetsToNuNu_103.root', 'HToBB_115.root', 'TTBarLep_107.root', 'ZJetsToNuNu_104.root', 'HToGG_105.root', 'ZJetsToNuNu_101.root', 'HToWW2Q1L_112.root', 'WToQQ_109.root', 'HToGG_109.root', 'WToQQ_106.root', 'WToQQ_108.root', 'TTBar_109.root', 'HToBB_100.root', 'TTBar_113.root', 'HToBB_111.root', 'WToQQ_111.root', 'WToQQ_101.root', 'HToBB_107.root', 'HToCC_116.root', 'HToGG_115.root', 'ZJetsToNuNu_100.root', 'ZJetsToNuNu_102.root', 'HToWW4Q_105.root', 'ZJetsToNuNu_108.root', 'TTBar_101.root', 'WToQQ_119.root', 'HToGG_100.root', 'TTBar_102.root', 'HToBB_102.root', 'HToBB_118.root', 'TTBar_116.root', 'HToWW2Q1L_113.root', 'HToWW2Q1L_119.root', 'HToWW2Q1L_108.root', 'HToWW4Q_103.root', 'ZJetsToNuNu_115.root', 'HToCC_104.root', 'HToGG_111.root', 'HToWW4Q_114.root', 'TTBar_108.root', 'HToGG_118.root', 'HToWW2Q1L_110.root', 'HToCC_111.root', 'HToWW2Q1L_101.root', 'TTBarLep_119.root', 'HToWW4Q_101.root', 'ZToQQ_113.root', 'TTBar_103.root', 'ZJetsToNuNu_118.root', 'HToBB_109.root', 'TTBarLep_117.root', 'HToGG_107.root', 'HToWW2Q1L_103.root', 'HToWW4Q_110.root', 'HToWW4Q_118.root', 'HToGG_116.root', 'ZToQQ_100.root', 'TTBarLep_114.root', 'HToCC_103.root', 'ZToQQ_107.root', 'HToWW2Q1L_104.root', 'ZToQQ_102.root', 'TTBarLep_115.root', 'HToGG_101.root', 'WToQQ_103.root', 'ZJetsToNuNu_107.root', 'ZJetsToNuNu_109.root', 'TTBarLep_102.root', 'HToWW2Q1L_115.root', 'HToWW2Q1L_102.root', 'HToGG_117.root', 'HToWW4Q_115.root', 'HToGG_119.root', 'HToBB_117.root', 'TTBar_110.root', 'ZToQQ_108.root', 'HToWW2Q1L_107.root', 'HToGG_114.root', 'HToBB_101.root', 'HToWW2Q1L_111.root', 'HToCC_108.root', 'TTBar_115.root', 'HToWW4Q_111.root', 'HToCC_105.root', 'HToGG_108.root', 'HToWW4Q_102.root', 'ZJetsToNuNu_110.root', 'HToWW4Q_106.root', 'ZToQQ_115.root', 'HToGG_104.root', 'TTBar_118.root', 'TTBarLep_118.root', 'ZToQQ_116.root', 'HToWW2Q1L_114.root', 'ZJetsToNuNu_111.root', 'HToWW4Q_109.root', 'ZToQQ_110.root', 'HToBB_103.root', 'HToBB_116.root', 'HToCC_115.root', 'HToCC_102.root', 'HToGG_106.root', 'ZToQQ_106.root', 'HToWW2Q1L_117.root', 'TTBar_105.root', 'HToBB_105.root', 'TTBarLep_103.root', 'HToWW2Q1L_106.root', 'TTBarLep_111.root', 'WToQQ_112.root', 'ZJetsToNuNu_117.root', 'TTBarLep_108.root', 'TTBarLep_110.root', 'ZToQQ_119.root', 'ZJetsToNuNu_106.root', 'TTBarLep_104.root', 'ZJetsToNuNu_114.root', 'WToQQ_115.root', 'WToQQ_114.root', 'HToBB_113.root', 'HToCC_119.root', 'TTBarLep_100.root', 'ZToQQ_111.root', 'HToGG_102.root', 'HToWW4Q_113.root', 'HToBB_110.root', 'HToCC_113.root', 'HToBB_104.root', 'WToQQ_107.root', 'HToWW4Q_108.root', 'ZToQQ_118.root', 'HToCC_110.root', 'TTBar_114.root', 'TTBar_119.root', 'HToCC_114.root', 'ZToQQ_109.root', 'TTBarLep_112.root', 'HToGG_112.root', 'WToQQ_100.root', 'HToWW2Q1L_109.root', 'HToCC_112.root', 'HToWW4Q_112.root']"
try:
    filenames_list = eval(long_string.split("//")[1])  # Extract the list of filenames
    for filename in filenames_list:
        temp_x_particles, temp_x_jets, temp_y = read_file(f"{filepath}/{filename}")
        x_particles.append(temp_x_particles)
        x_jets.append(temp_x_jets)
        y.append(temp_y)
except Exception as e:
    print("Error occurred:", e)

try:
    filenames_list = eval(long_string.split("//")[1])  # Extract the list of fi>
    for filename in filenames_list:
        temp_x_particles, temp_x_jets, temp_y = read_file(f"{filepath}/{filename}")
        x_particles.append(temp_x_particles)
        x_jets.append(temp_x_jets)
        y.append(temp_y)
        print(x_particles)
        print(x_jets)
        print(y)
except Exception as e:
    print("Error occurred:", e)

# for root, dirs, files in os.walk(filepath):
#     if files.endswith('.root'):
#         temp_x_particles, temp_x_jets, temp_y = read_file(f"{filepath}/{filenames}")
#         x_particles.append(temp_x_particles)
#         x_jets.append(temp_x_jets)
#         y.append(temp_y)

x_particles = np.array(x_particles)
x_jets = np.array(x_jets)
y = np.array(y)

# x_particles, x_jets, y = read_file(f"{filepath}/HToBB_100.root")
# Convert labels to categorical (one-hot encoding)
y = to_categorical(y)



"""
Particle Input:
Shape: (N, C)
Each row corresponds to the features of a particle in a jet.

Interaction Input:
Shape: (N, N, C')
Represents the interactions between pairs of particles.
"""

num_particles = x_particles.shape[2]
num_particle_features = x_particles.shape[1]
num_interaction_features = num_particle_features * 2  

def preprocess_particles(x_particles, num_particles, num_particle_features):
    # Reshape x_particles to (num_jets * num_particles, num_particle_features) (N,C)
    particles = x_particles.reshape(-1, num_particle_features)
    return particles

def preprocess_interaction(x_particles, num_particle_features): # U matrix (N, N, C')
    # initalize the shape with all 0
    N = x_particles.shape[0]
    interaction = np.zeros((N, N, num_particle_features))
    # pairwise interactions between all particles in the jet and feeding numebrs
    for i in range(num_particles):
        for j in range(num_particles):
            interaction[i, j] = np.concatenate((x_particles[i], x_particles[j]))
    return interaction

# Get particles matrix (N,C)
particles_matrix = preprocess_particles(x_particles, num_particles, num_particle_features)

# Get interaction matrix (N, N, C')
interaction_matrix = preprocess_interaction(particles_matrix, num_particle_features)

# Check shapes
print("Particle Matrix shape:", particles_matrix.shape)
print("Interaction Matrix shape:", interaction_matrix.shape)

particle_input = layers.Input(shape=(None, num_particle_features))  # Shape: (None, None, num_particle_features)
interaction_input = layers.Input(shape=(None, None, num_interaction_features))  # Shape: (None, None, None, num_interaction_features)
    





