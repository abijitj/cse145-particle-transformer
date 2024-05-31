![quantized-particle-transformer-high-resolution-logo-white](https://github.com/abijitj/cse145-particle-transformer/assets/79886525/6307c880-e92c-45e6-9447-d61ca67482e0)

This repo contains a project that is working towards building a quantized and more efficient version of the Particle Transformer (ParT) machine learning model described in this [paper](https://arxiv.org/abs/2202.03772) and its associated [repo](https://github.com/jet-universe/particle_transformer). 

# Team Members
 - Abijit Jayachandran [ajayachandran@ucsd.edu](ajayachandran@ucsd.edu)
 - Andrew Masek [amasek@ucsd.edu](amasek@ucsd.edu)
 - Juan Yin [j9yin@ucsd.edu](j9yin@ucsd.edu)

#  Abstract & Introduction
Running large machine learning models on the edge at the Large Hadron Collider is a challenging task because of the limited computing available and the time constraints. Existing models are often too large and therefore take a lot of memory to process and time to process the data. One method to get around this problem is to quantize existing models and use FPGAs (as opposed to general-purpose GPUs) for faster and more specialized processing. Our project aims to quantize an existing Particle Transformer model from PyTorch to QKeras. This new quantized model can then be implemented on an FPGA using the DeepSoCFlow library. We hope to maintain similar accuracy levels but achieve faster inference time. 

# How to run Quantized Particle Transformer
## Setting up
Quantized Particle Transformer need to run in particular environments. To do this we have included a requirements.txt file that contains the required environment that can be used for preprocessing and postprocessing for your Quantized Particle Transformer run.
The run syntax is as follows:
```
pip install -r requirements.txt
```

## Downloading dataset
After making sure the environment is correct, we should download our large dataset:
```
./get_datasets.py JetClass -d [DATA_DIR]
```

## Training
The QPT model are implemented in Tensorflow instead of Pytorch, so the training is not based on the weaver framework for dataset loading and transformation. To run the training on the JetClass dataset:
```
python3 train.py
```

## Multi-gpu support:
Training dataset faster using GPU:
```
```






# Technical Materials

## Dataset 
We have a large particle jets dataset used for training and testing by downloading from [website](https://zenodo.org/records/6619768), it is a large dataset with more than 200GB Jets data which takes 7-12 hours to download.

## Reference Models Used
- [Partile Transformer](https://github.com/jet-universe/particle_transformer)
- [weaver-core](https://github.com/hqucms/weaver-core)
- [Nano-GPT](https://github.com/karpathy/nanoGPT)

## Methods

#### 1. Retrain the model
In this section, we download the large particle jets dataset to train the Particle Transformer model that was implemented using PyTorch...

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
In this section, we try to get familiar with the transformer model, so we reimplement a transformer model called [nanoGPT](https://github.com/karpathy/nanoGPT).
First of all, we followed instructions to rewrite the transformer architecture in order to get a sence of how to create a transformer and get familiar with all the components like multi-head attention blocks and embedded layers with different activation functions. 

After simulating and training the NanoGPT model, we translate the nanoGPT from Pytorch to Tensorflow to verify that the translation won't affect the performance of the model.
![image](https://github.com/abijitj/cse145-particle-transformer/assets/79886525/8f722ab2-d1e0-416b-a6ce-94885af8b189)
![image](https://github.com/abijitj/cse145-particle-transformer/assets/79886525/30695dc4-ad30-4fba-80bc-6acc8572d825)


Then we begin to do the translation on the particle transformer model from [weaver-core repo](https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py). 
...


#### Pytorch to Keras and QKeras translations
 - [nanoPGT translation](https://github.com/abijitj/nanoGPT)
 - [bigram translation](https://github.com/JuanYin1/smallGPT_keras)
 - [Pytorch nanoGPT & bigram model](https://github.com/portoaj/NanoGPT-Fork)


## REFERENCES
- [Quantization helps reduce MAC](https://arxiv.org/pdf/2106.08295)
- 

   





   

   

