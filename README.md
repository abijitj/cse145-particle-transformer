![quantized-particle-transformer-high-resolution-logo-white](https://github.com/abijitj/cse145-particle-transformer/assets/79886525/6307c880-e92c-45e6-9447-d61ca67482e0)

This is a project that is working towards building a quantized and more efficient version of the Particle Transformer (ParT) machine learning model described in this [paper](https://arxiv.org/abs/2202.03772) and its associated [repo](https://github.com/jet-universe/particle_transformer). 

# Team Members
 - Abijit Jayachandran [ajayachandran@ucsd.edu](ajayachandran@ucsd.edu)
 - Andrew Masek [amasek@ucsd.edu](amasek@ucsd.edu)
 - Juan Yin [j9yin@ucsd.edu](j9yin@ucsd.edu)

# Abstract & Introduction
Running large machine learning models on the edge at the Large Hadron Collider (LHC) is a challenging task because of the limited compute available and the time constraints. Existing models are often too large and therefore take a lot of memory and time to process the data. One method to get around this problem is to quantize existing models and use FPGAs (as opposed to general-purpose GPUs) for faster and more specialized processing. Our project aims to quantize an existing Particle Transformer model from PyTorch to QKeras. This new quantized model can then be implemented on an FPGA using the DeepSoCFlow library. We hope to maintain similar accuracy levels but achieve faster inference time. 

# File Organization
All documentation is contained in the `cse145-particle-transformer/Documentation` directory. The directory contents are:
- `presentations/`: all presentations of the project 
- `reports/`: all reports of the project
- `CITATION.cff`: citation file for ParT
- `requirements.txt`: a list containing different libraries necessary for the project.

All test files are contained in the `cse145-particle-transformer/test` directory, which records our previous attempts at loading data from the dataset. 

Our keras implementation of nanoGPT is in the `cse145-particle-transformer/nanogpt` directory.

Our Keras implementation of the ParT is contained in the  `cse145-particle-transformer/final_particle_transformer/` directory. You can use the `train.py` file to train this model. Please note, most of the other files are from the original [ParT repo](https://github.com/jet-universe/particle_transformer). As such you can also run the original ParT here by following the instructions in their repo.  

# How to run Quantized Particle Transformer
## Setting up
Quantized Particle Transformer needs to run in particular environments. To do this we have included a requirements.txt file that contains the required environment for running the Keras ParT. You can download these dependencies using: 
```
pip install -r ./Documentation/requirements.txt
```

## Downloading dataset
After making sure the environment is correct, you download the JetClass dataset using the command below. Please note that this is a very large dataset that will download well over 100GB of files. If you would like to download a subset of the data, please modify `get_datasets.py` or download directly from [this website](https://zenodo.org/records/6619768). 
```
./get_datasets.py JetClass -d [DATA_DIR]
```

## Training with GPU
Make sure you have CUDA installed in your computer. For our project, we need [cuda version 10.1](https://www.tensorflow.org/install/source#gpu:~:text=11.0-,tensorflow-2.3.0,10.1,-tensorflow-2.2.0) since that is compatible with our TensorFlow version (2.10.0)

Our model is implemented in Tensorflow instead of Pytorch, so the training is not based on the weaver framework for dataset loading and transformation. After you get CUDA installed, make sure you've downloaded the entire dataset and changed the training and validation data paths in the `final_particle_transformer/train.py` file:

`train.py` datapath:
```
# line 35
training_data_root = '/home/particle/particle_transformer/retrain_test_1/JetClass/Pythia/train_100M/'
```
to: 
```
# line 35
training_data_root = '[USER_DIR]/[DATA_DIR]/JetClass/Pythia/train_100M/'
```

Finally, make sure you are in the `cse145-particle-transformer/final_particle_transformer/` directory and run the following command.
```
python3 train.py
```

# Technical Materials

## Difference between PT and QPT
![image](https://github.com/abijitj/cse145-particle-transformer/assets/79886525/1b1f33ae-2de7-4095-a43d-c180361121eb)
*image from this [website](https://medium.com/@jan_marcel_kezmann/master-the-art-of-quantization-a-practical-guide-e74d7aad24f9)

#### Original Particle Transformer:

1. The original Parallel Transformer operates with the default precision, which is 32-bit floating point numbers (FP32). The high precision results in more accurate computations, but it requires more memory and computational resources.

2. It is suitable for training and used to generate high accuracy where there are sufficient resources to handle the computational load.
However, it is more memory-intensive and slower due to the high precision of calculations.

#### Quantized Particle Transformer:

1. The quantized ParT operates with reduced precision. Lower precision leads to faster computations and reduced memory usage but with a potential trade-off in accuracy.

2. It is faster for training and ideal for inference scenarios where speed and resource efficiency are prioritized, such as deployment on edge devices or in real-time applications (such as at the LHC). It uses less memory and faster computations due to lower precision. And the size of the model is smaller because weights and activations are stored with reduced precision.

#### Key Differences in four aspects:
1. Precision and Accuracy: Original ParT uses high precision (FP32), leading to higher accuracy, whereas quantized ParT uses lower precision (INT8 or smaller).
2. Performance and Efficiency: Quantized ParT offers better performance and efficiency in speed and memory usage.
3. Use Cases: Original ParT is used where accuracy is important, while quantized ParT is used for efficient inference, especially in production conditions. And in order to make it compatible to be implemented in FPGA, quantization is needed.
4. Model Size: The quantized ParT model is smaller and requires fewer computational resources, making it faster to run, whereas the original ParT model is larger and takes more time to do the computation.

## Dataset 
We have a large particle jets dataset used for training and testing by downloading from [website](https://zenodo.org/records/6619768), it is a large dataset with more than 100GB of data.

**[JetClass](https://zenodo.org/record/6619768)** is a new large-scale jet tagging dataset proposed in "[Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772)". It consists of 100M jets for training, 5M for validation and 20M for testing. The dataset contains 10 classes of jets, simulated with [MadGraph](https://launchpad.net/mg5amcnlo) + [Pythia](https://pythia.org/) + [Delphes](https://cp3.irmp.ucl.ac.be/projects/delphes):

![dataset](figures/dataset.png)


## Reference Models Used
- [Particle Transformer](https://github.com/jet-universe/particle_transformer)
- [weaver-core](https://github.com/hqucms/weaver-core)
- [Nano-GPT](https://github.com/karpathy/nanoGPT)

## Deliverables

#### 1. Train original ParT
In this section, we download the large particle jets dataset to train the original ParT model in PyTorch. 

 - Result 1 - final ACC = 0.834:
   epoch = 2
   ![image](https://github.com/abijitj/cse145-particle-transformer/assets/79886525/1ff3ad89-5ea9-44ce-849b-adb590289140)
   ![image](https://github.com/abijitj/cse145-particle-transformer/assets/79886525/907b584e-fb72-485d-a78c-0d4f50966415)
   
 - Result 3 - final ACC = 0.849:
   epoch = 5
   <img width="700" alt="截屏2024-04-26 14 42 24" src="https://github.com/abijitj/cse145-particle-transformer/assets/79886525/ac7edbd8-6834-44ea-b61b-d2fc964becc9">
   <img width="730" alt="截屏2024-04-27 15 53 44" src="https://github.com/abijitj/cse145-particle-transformer/assets/79886525/726c80b4-f6c9-44c8-a3f7-a63ec25a670a">
   <img width="742" alt="截屏2024-04-27 17 59 54" src="https://github.com/abijitj/cse145-particle-transformer/assets/79886525/5fb0f8cd-4a70-41c1-9340-be49b3f5b934">
   <img width="762" alt="截屏2024-04-27 18 54 18" src="https://github.com/abijitj/cse145-particle-transformer/assets/79886525/b5537d2d-8199-4178-b0b3-0b2dc9c01481">
   <img width="731" alt="截屏2024-04-27 19 43 21" src="https://github.com/abijitj/cse145-particle-transformer/assets/79886525/e6e00183-33e8-4739-b8c0-40924f4aefb1">


#### 2. Practice translation from PyTorch to TensorFlow
In this section, we try to get familiar with the transformer model, so we reimplement a transformer model called [nanoGPT](https://github.com/karpathy/ng-video-lecture).

First of all, we followed instructions to rewrite the transformer architecture in order to get a sense of how to create a transformer and get familiar with all the components like multi-head attention blocks and embedded layers with different activation functions. 

After simulating and training the NanoGPT model, we translated the nanoGPT from Pytorch to Tensorflow to verify that the translation won't affect the performance of the model.
![image](https://github.com/abijitj/cse145-particle-transformer/assets/79886525/8f722ab2-d1e0-416b-a6ce-94885af8b189)
![image](https://github.com/abijitj/cse145-particle-transformer/assets/79886525/30695dc4-ad30-4fba-80bc-6acc8572d825)

 - [Abijit's nanoPGT](https://github.com/abijitj/nanoGPT)
 - [Juan's nanoGPT](https://github.com/JuanYin1/smallGPT_keras)
 - [Andrew's nanoGPT](https://github.com/portoaj/NanoGPT-Fork)


### 3. Apply Transformation on Building Particle Transformer Model
In this section we begin to do the translation on the particle transformer model from [weaver-core repo](https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py). We implementing the data loadar, embedded layers, particle multi-head attention blocks, and class attention blocks. The attention portions of the model are similar to the NanoGPT that we completed in section 2, and adding the particle transformer features. According to the [Particle Transformer paper](https://arxiv.org/pdf/2202.03772) the feature matrix `U` is a pair matrix that has three dimensions `(N, N, C')`, and instead of feeding one embedded layer to each particle multi-head attention blocks, we feed two embedded layers: particle layer and interactive U layer.

We first make sure the Keras works on the model, then continue translate them to QKeras and try to maintain the accuracy of training.

Finally we got our model running:
[Add images when finished training]
[images]


#### Pytorch to Keras and QKeras translations


# References
- [Particle Transformer Paper](https://arxiv.org/pdf/2202.03772)
- [NanoGPT Model Guide](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4298s)
- [NanoGPT Documentation](https://github.com/abarajithan11/nanoGPT)
- [TensorFlow Input Pipline](https://www.tensorflow.org/guide/data)
- [QKeras Documentation](https://github.com/google/qkeras/blob/v0.9.0/notebook/QKerasTutorial.ipynb)
- [Quantization Helps Reduce MAC Paper](https://arxiv.org/pdf/2106.08295)


# Citation
[Citation file](https://github.com/abijitj/cse145-particle-transformer/blob/main/CITATION.cff)



   

   

