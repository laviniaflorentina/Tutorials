# About Artificial Inteligence :thought_balloon:

# :construction: ... Work in Progress ... :construction:

**Artificial Inteligence (AI)** is a field in Computer Science where the code is _dynamic_, being able to learn by itself, “observing” (word substitute for “applying mathematical magic beasts”) other similar situations. Those special types of codes are named Artificial Neural Networks. They aim to keep up with this ever-changing world by self-adapting and learning on the fly.  

If you want to apply some of the exciting models that creates false images or videos of celebrities or drive self-driving cars, or if you just wonder about what Artificial Inteligence is about, you might find this repository useful.

This repository exposes all the terminology used in Neural Networks, as well as it presents and explains the most common used algorithms that you need to know in order to understand how to build your own model.

**NOTE.** This tutorial is only for educational purpose. It is not an academic paper. All references are listed at the end of the file.

----------------------------------------------- 

# What's next in this cool repo? :octocat:

## 1. [Introduction](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/Introduction.md). What I cover:

 - What is an Artificial Neural Network?
 - What are the main components and why do we need each of them?
   - Weights, Bias and Layers
   - Activation Function: **Linear Activation Function** and **Non-linear Activation Function** (_Sigmoid, Tanh_ & _ReLU_)
   - Derivatives
 
## 2. [Artificial Neural Network Architectures](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md). What I cover:

- [Introduction](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#introduction)
- Feedforward Neural Networks (FF):
  - [Singe-layer Perceptron (Perceptron)](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#11singe-layer-perceptron-perceptron---coursera)
  - [Multi-Layer Perceptron (MLP)](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#12multi-layer-perceptron-mlp---coursera)
- [Radial Basis Function (RBF) Neural Network ](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#2radial-basis-function-rbf-neural-network)
- [Deep Feedforward Network (DFF)](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#3deep-feedforward-network-dff---coursera)
- Recurrent Networks:
  - [Recurrent Neural Networks (RNN)](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#41-recurrent-neural-networks-rnn--coursera)
  - [Long-Short Term Memory (LSTM) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#42long-short-term-memory-lstm-neural-networks---coursera)
  - [Gated Recurrent Unit (GRU) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#43gated-recurrent-unit-gru-neural-networks---coursera)
- Auto-Encoder Networks:
  - [Auto-Encoder (AE) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#51-auto-encoder-ae-neural-networks---coursera)
  - [Variational Auto-Encoder (VAE) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#52-variational-auto-encoder-vae-neural-networks---coursera)
  - [Denoising Auto-Encoder (DAE) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#53denoising-auto-encoder-dae-neural-networks---coursera)
  - [Sparse Auto-Encoder (SAE) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#54sparse-auto-encoder-sae-neural-networks)
- [Markov Chain (MC) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#6markov-chain-mc-neural-networks---coursera)
- [Hopfield Network (HN) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#7hopfield-network-hn-neural-networks)
- Boltzmann Networks:
  - [Boltzmann Machine (BM) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#81-boltzmann-machine-bm-neural-networks)
  - [Restricted Boltzmann Machine (RBM) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#82restricted-boltzmann-machine-rbm-neural-networks---coursera)
- [Deep Belief Network (DBN) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#9deep-belief-network-dbn-neural-networks)
- Convolutional Networks:
  - [Deep Convolutional Network (DCN) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#101-deep-convolutional-network-dcn-neural-networks---coursera)
  - [Deconvolutional Neural Network (DNN) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#102-deconvolutional-neural-network-dnn-neural-networks)
  - [Deep Convolutional Inverse Graphics Network (DCIGN)](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#103-deep-convolutional-inverse-graphics-network-dcign)
- [Generative Adversarial Neural Networks (GAN)](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#11generative-adversarial-neural-networks-gan---coursera)
- [Liquid State Machine (LSM) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#12liquid-state-machine-lsm-neural-networks)
- [Extreme Learning Machine (ELM) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#13extreme-learning-machine-elm-neural-networks)
- [Echo State Neural Networks (ESN)](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#14echo-state-neural-networks-esn---youtube)
- [Deep Residual Neural Networks (DRN)](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#15deep-residual-neural-networks-drn---coursera)
- [Kohonen Neural Networks (KN)](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#16kohonen-neural-networks-kn)
- [Support Vector Machine (SVM) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#17support-vector-machine-svm-neural-networks---coursera)
- [Neural Turing Machine (NTM) Neural Networks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/architectures.md#18neural-turing-machine-ntm-neural-networks)

## 3. [Machine Learning](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#what-is-machine-learning-ml). What I cover:
 
 - Introduction
 - Machine Learning Categories by the level of human supervision:
   - Supervised Learning in different areas: Text - Natural Language Processing (NLP), Image - Computer Vision, Audio - Automatic Speech Recognition (ASR).
   - Unsupervised Learning in different areas: Text - Natural Language Processing (NLP), Image - Computer Vision, Audio - Automatic Speech Recognition (ASR).
   - Semi-Supervised Learning in different areas: Text - Natural Language Processing (NLP), Image - Computer Vision, Audio - Automatic Speech Recognition (ASR).
   - Self-Supervised Learning in different areas: Text - Natural Language Processing (NLP), Image - Computer Vision, Audio - Automatic Speech Recognition (ASR).
   - Reinforcement Learning in different areas: Text - Natural Language Processing (NLP), Image - Computer Vision), Audio - Automatic Speech Recognition (ASR).
 
 ## 3. [Algorithms by tasks](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/algorithms.md). What I cover:
 
 - Classification: 
 - Regression:
 - Clustering: K - Means, K - Nearest Neighbors.
 - Prediction:
 
 ## 4. [Training](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/training.md). What I cover:
 
 -
 -
 
 ## 5. [Evaluation](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/evaluation.md). What I cover:
 
 - Training Evaluation Methods
 - Model Evaluation Methods
 
 ## 6. [Hardware Parts for Training](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/hardware_for_training.md#nut_and_bolt-hardware-parts-for-training-wrench). What I cover:
 
 - Why bother?

 - About **CPU, GPU, TPU, FPGA**

 - How to use these resources in your code?

   - In Browser (Google Colab or Jupyter Notebooks)

   - On HPC (or other High Performance Computing) 
   
-----------------------------------------------
