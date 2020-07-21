# Artificial Neural Networks Architectures

----------------------------

Content:

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

--------------------

## Introduction

The neural network architecture represents the way in which the elements of the network are binned together. The architecture will define the behavior of the network. Considering their architecture, Artificial Neural Networks can be classified as follows:

# 1.	Feedforward Neural Networks (FF)

Feedforward Neural Networks were the first type of Artificial Neural Network. These kinds of nets have a large variance but there are mainly two pioneers:

## 1.1.	Singe-layer Perceptron (Perceptron) - [Coursera](https://www.coursera.org/lecture/mind-machine-computational-vision/perceptrons-4AT1O)

The **perceptron** is the simplest type of feedforward neural network, compost of only one neuron, where:  

  -	It takes some inputs, <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/xi.png" alt="inpits" width="25" height="25">, and each of them is multiplied with their related weight, <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/wi.png" alt="inpits" width="25" height="25"> : 

 ![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann3.png)

  - Then, it sums them up: <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/sum_n.png" alt="inpits" width="50" height="50">, particularly for the example above, we will have: <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/sum_5.png" alt="inpits" width="50" height="50">.

  -	The **activation function** (usually this is a _logistic function_) squashes the above summation and passes it to the output layer.

**Where do we use Perceptron?**

Perceptron is good for classification and decision-making systems because of the logistic function properties. That being the case, it is usually used to classify the data into two parts and it is also known as a Linear Binary Classifier. 

The issue with individual neurons rapidly arises when trying to solve every day - life problems due to the real-world complexity. Determined to address this challenge, researchers observed that by combining neurons together, their decision is basically combined getting as a result insight from more elaborated data. 

And so, the creation of an Artificial Neural Network with more neurons seems to be the answer.

## 1.2.	Multi-Layer Perceptron (MLP) - [Coursera](https://www.coursera.org/lecture/intro-to-deep-learning/multilayer-perceptron-mlp-yy1NV)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann4.png" width="250" height="200" > 

The **multi-layer perceptron** is a type of feedforward neural network that introduces the _multiple neurons_ design, where:

  - all nodes are fully connected (each node connects all the nodes from the next layer);
  - information only travels forward in the network (no loops);
  - there is one hidden layer (if present).

**Where do we use Multi-Layer Perceptron?**

Contrasting single layer perceptrons, MLPs are capable to manipulate non-linearly separable data. 

Therefore, these nets are used in many applications, but not by themselves. Most of the time they stand as a pillar of support in the construction of the next neural networks’ architecture. 

# 2.	Radial Basis Function (RBF) Neural Network 

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann5.png" width="280" height="180" > 

**Radial Basis Function Networks** are feedforward nets _with a different activation function_ in place of the logistic function, named the **radial basis function (RBF)**. 

Intuitively, RBF answers the question _“How far is the target from where we are?”_. More technically, it is a real-valued function defined as the difference between the input and some fixed point, called the **center**. 

The RBF chosen is usually a Gaussian, and it behaves like in the following example: 

<img src="https://www.digitalvidya.com/wp-content/uploads/2019/01/Image-3.gif" width="300" height="200" > 

RBN is strictly limited to have **exactly one hidden layer** (green dots in the related figure). Here, this hidden layer is known as a **feature vector**.

Moreover, Wikipedia says:

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann5-1.png)

**Where do we use Radial Basis Function Network?**

RBF nets are used in [function approximation](https://github.com/andrewdyates/Radial-Basis-Function-Neural-Network/blob/master/CSE779LabReport2.pdf) ([paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.6019&rep=rep1&type=pdf) & [code](https://github.com/thomberg1/UniversalFunctionApproximation)), time series prediction ([paper](https://dl.acm.org/doi/pdf/10.1145/3305160.3305187) & [code](https://www.mathworks.com/matlabcentral/fileexchange/66216-mackey-glass-time-series-prediction-using-radial-basis-function-rbf-neural-network)), and machine/system control (for example as a replacement of Partial Integral Derivative controllers). 

# 3.	Deep Feedforward Network (DFF) - [Coursera](https://www.coursera.org/lecture/ai/deep-feed-forward-neural-networks-kfTED)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann6.png" width="200" height="180" > 

**Deep Feedforward Neural Networks** are feedforward nets with _more than one hidden layer_.

It follows the rules:
  - all nodes are fully connected;
  - activation flows from input layer to output, without back loops;
  - there is **more than one layer** between input and output (hidden layers – green dots).

When training the traditional FF model, only a small amount of error information passes to the next layer. With more layers, DFF is able to learn more about errors; however, it becomes impractical as the amount of training time required increases.

Nowadays, a series of effective methods for training DFF have been developed, which have formed the core of modern machine learning systems and enable the functionality of feedforward neural networks.

**Where do we use Deep Feed-Forward Network?**

DFF is being used for automatic language identification (the process of automatically identifying the language spoken or written: [paper](https://static.googleusercontent.com/media/research.google.com/en/pubs/archive/42538.pdf), [paper](https://arxiv.org/pdf/1708.04811.pdf) & [code](https://github.com/HPI-DeepLearning/crnn-lid)), acoustic modeling for speech recognition ([thesis](https://mi.eng.cam.ac.uk/~mjfg/thesis_cw564.pdf), [paper](https://arxiv.org/pdf/1809.02108.pdf) & [code](https://github.com/amitai1992/AutomatedLipReading)), and other.

# 4.	Recurrent Networks 

## 4.1. Recurrent Neural Networks (RNN) – [Coursera](https://www.coursera.org/lecture/nlp-sequence-models/recurrent-neural-network-model-ftkzt) 

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann7.png" width="280" height="180" > 

The **Recurrent Neural Networks (RNN)** introduce the _recurrent cells_, a special type of cells located in the hidden layer (blue dots) and responsible of receiving its own output with a fixed delay — for one or more iterations creating loops. Apart from that, this network is like a usual FF net and so it follows similar rules:

  - all nodes are fully connected;
  - activation flows from input layer to output, with back loops;
  - there is more than one layer between input and output (hidden layers).

**Where do we use Recurrent Neural Networks?**

RNN is mainly used when the context is important — when decisions from past iterations or samples can influence the current ones, such as sentiment analysis ([paper](https://arxiv.org/pdf/1902.09314v2.pdf) & [code](https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/SentimentAnalysisProject)).

The most common examples of such contexts are texts — a word can be analyzed only in the context of previous words or sentences. RNNs can process texts by “keeping in mind” ten previous words.

[More on RNN](https://github.com/kjw0612/awesome-rnn) - A curated list of resources dedicated to recurrent neural networks.

## 4.2.	Long-Short Term Memory (LSTM) Neural Networks - [Coursera](https://www.coursera.org/lecture/tensorflow-sequences-time-series-and-prediction/lstm-5Iebr)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann8.png" width="280" height="180" > 

**LSTM** is a subcategory of RNN. They introduce a _memory cell_ – a special cell that can store and recall facts from time dependent data sequences. (tutorial like [paper](https://arxiv.org/pdf/1909.09586.pdf))

Memory cells are actually composed of a couple of elements — called gates, that are recurrent and control how information is being remembered and forgotten. 

  -	The input Gate determines how much of the last sample is stored in memory; 
  -	The output gate adjusts the amount of data transferred to the next level; 
  -	The Forget Gate controls the rate at which memory is stored.

Note that there are **no activation functions** between blocks.

**Where do we use Long-Short Term Memory Network?**

LSTM networks are used when we have timeseries data, such as: video frame processing ([paper](https://arxiv.org/pdf/1909.05622.pdf) & [code](https://github.com/matinhosseiny/Inception-inspired-LSTM-for-Video-frame-Prediction)), writing generator ([article w/ code](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/), [article w/ code](https://medium.com/towards-artificial-intelligence/sentence-prediction-using-word-level-lstm-text-generator-language-modeling-using-rnn-a80c4cda5b40)) as it can “keep in mind” something that happened many frames/ sentences ago. 

## 4.3.	Gated Recurrent Unit (GRU) Neural Networks - [Coursera](https://www.coursera.org/lecture/nlp-sequence-models/gated-recurrent-unit-gru-agZiL)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann9.png" width="250" height="180" > 

**GRU** is a subcategory of RNN. GRUs are similar with LSTMs, but with a different type of gates. The lack of output gate makes it easier to repeat the same output for a concrete input multiple time and, therefore, they are less resource consuming than LSTMs and have similar performance.

**Where do we use Gated Recurrent Unit?**

They are currently used in similar applications as LSTMs.

# 5.	Auto-Encoder Networks

## 5.1. Auto-Encoder (AE) Neural Networks - [Coursera](https://www.coursera.org/projects/dimensionality-reduction-autoencoder-python)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann10.png" width="145" height="200"> 

**Auto-encoder networks** can be trained without supervision – covered in the next chapter!
Their structure with a number of hidden cells smaller than the number of input cells (and number of output cells equals number of input cells). 

The fact that AE is trained such that the output is as close as possible to the input, forces AEs to generalize data and search for common patterns.

**Where do we use Auto-Encoders?**

Auto-encoders can only answer questions like: "How do we summarize the data?", so they are used for classification, clustering and feature compression in problems like face recognition and acquiring semantic meaning of words.  

## 5.2. Variational Auto-Encoder (VAE) Neural Networks - [Coursera](https://www.coursera.org/projects/image-compression-generation-vae)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann11.png" width="150" height="200" > 

**VAE** is a category of AE. While the Auto-Encoder compresses features, Variational Auto-Encoders compresses the probability. 
This change makes a VAE answer questions like _"How strong is the connection between the two things?”_, _“Should we divide in two parts or are they completely independent?"_.
 
In neural net language, a VAE consists of an encoder, a decoder, and a loss function. 
In probability model terms, the variational autoencoder refers to approximate inference in a latent Gaussian model where the approximate posterior and model likelihood are parametrized by neural nets (the inference and generative networks).

•	Encoder: 
  -	in the neural net world, the encoder is a neural network that outputs a representation z of data x. 
  -	In probability model terms, the inference network parametrizes the approximate posterior of the latent variables z. The inference network outputs parameters to the distribution q(z∣x).
  
•	Decoder: 
  -	in deep learning, the decoder is a neural net that learns to reconstruct the data x given a representation z. 
  -	In terms of probability models, the likelihood of the data x given latent variables z is parametrized by a generative network. The generative network outputs parameters to the likelihood distribution p(x∣z).
  
•	Loss function: 
  -	in neural net language, we think of loss functions. Training means minimizing these loss functions. But in variational inference, we maximize the ELBO (which is not a loss function). This leads to awkwardness like calling optimizer.minimize(-elbo) as optimizers in neural net frameworks only support minimization.

**Where do we use Variational Auto-Encoder?**

They are powerful generative models with vast applications, including generating fake human faces, and purely music composition.

Other paper & code.

## 5.3.	Denoising Auto-Encoder (DAE) Neural Networks - [Coursera](https://www.coursera.org/projects/autoencoders-image-denoising)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann12.png" width="150" height="200" > 

**DAE** is a category of AE. Auto-encoders sometimes fail to find the most proper features but rather adapts to the input data (example of over-fitting). 

The Noise Reduction Auto-Encoder (DAE) adds some noise to the input unit - changing data by random bits, arbitrarily shifting bits in the input, and so on. 
By doing this, a forced noise reduction auto-encoder reconstructs the output from a somewhat noisy input, making it more generic, forcing the selection of more common features.

**Where do we use Denoising Auto-Encoders?**

They are important for feature selection and extraction and the main usage of this network is to recover a clean input from a corrupted version, such as image denoising (super resolution) for medical purposes.

## 5.4.	Sparse Auto-Encoder (SAE) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann13.png" width="150" height="200" > 

**Sparse Auto-Encoder** is another form of auto-encoding that sometimes pulls out some hidden aspects from the data. 

Here, the number of hidden cells is greater than the number of input or output cells and this constraint forces the model to respond to the unique statistical features of the input data.

**Where do we use Sparse Auto-Encoders?**

This type of auto-encoders can be used in popularity prediction (as this paper studied the prediction of Instagram posts popularity), and machine translation.

# 6.	Markov Chain (MC) Neural Networks - [Coursera](https://www.coursera.org/lecture/bayesian-methods-in-machine-learning/markov-chains-uGsUe)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann14.png" width="180" height="200" > 

**Markov Chain** is an old chart concept. It is not a typical neural networks. 
Each of its endpoints is assigned with a certain probability. 

**Where do we use Markov Chain?**

In the past, it’s been used to construct a text structure like "dear" appears after "Hello" with a probability of 0.0053%.
They can be used as **probability-based categories** (like Bayesian filtering), for **clustering** and also as finite state machines.

# 7.	Hopfield Network (HN) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann15.png" width="180" height="200" > 

**Hopfield network** is initially trained to store a number of patterns or memories.
 
It is then able to recognize any of the learned patterns by exposure to only partial or even some corrupted information about that pattern 
_i.e._ it eventually settles down and returns the closest pattern or the best guess. 

Like the human brain memory, the Hopfield network provides similar pattern recognition. 
Each cell serves as input cell before training, as hidden cell during training and as output cell when used.

**Where do we use Hopfield Network?**

As HNs are able to discern the information even if corrupted, they can be used for denoising and restoring inputs. Given a half of learned picture or sequence, they will return a full object.

# 8.	Boltzmann Networks

## 8.1. Boltzmann Machine (BM) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann16.png" width="180" height="190" > 

**Boltzmann Machine**, also known as **Stochastic Hopfield Network**, is a network of symmetrically connected neurons. It is the first network topology that successfully preserves the simulated annealing approach.

It is named after the Boltzmann distribution (also known as Gibbs Distribution) which is an integral part of Statistical Mechanics. 
BM are non-deterministic (or stochastic) generative Deep Learning models, very similar to Hopfield Network, with only two types of nodes — hidden and input. 
There are no output nodes! 

In training, BM updates units one by one instead of in parallel. When the hidden unit updates its status, the input unit becomes the output unit. This is what gives them non-deterministic feature. 

They don’t have the typical 1 or 0 type output through which patterns are learned and optimized using Stochastic Gradient Descent. They learn patterns without that capability, and this is what makes them special.

**Where do we use Boltzmann Machine?**

Multi-layered Boltzmann machines can be used for so-called Deep Belief Networks.

## 8.2.	Restricted Boltzmann Machine (RBM) Neural Networks - [Coursera](https://www.coursera.org/lecture/building-deep-learning-models-with-tensorflow/introduction-to-restricted-boltzmann-machines-XEYUx)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann17.png" width="150" height="200" > 

The **Restricted Boltzmann Machines** are very similar to BMs in structure, but constrained RBMs are allowed to be trained back-propagating like FFs (the only difference is that the RBM will go through the input layer once before data is backpropagated). 

**Where do we use Restricted Boltzmann Machine?**

Restricted Boltzmann machine is an algorithm useful for dimensionality reduction, collaborative filtering, feature learning and topic modeling with practical application, for example, in speech recognition. 

# 9.	Deep Belief Network (DBN) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann18.png" width="280" height="190" > 

**Deep Belief Network** is actually a number of Boltzmann Machines surrounded by VAE together. They can be linked together (when one neural network is training another), and data can be generated using patterns learned. 

**Where do we use Deep Belief Network?**

Deep belief networks can be used for feature detection and extraction.

# 10.	Convolutional Networks

## 10.1. Deep Convolutional Network (DCN) Neural Networks - [Coursera](https://www.coursera.org/learn/convolutional-neural-networks)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann19.png" width="280" height="190" > 

**Deep Convolutional Network** has convolutional units (or pools) and kernels, each for a different purpose. Convolution kernels are actually used to process input data, and pooling layers are used to reduce unnecessary features.

**Where do we use Deep Convolutional Network?**

They are usually used for image recognition, running on a small part of the image (~ 20x20 pixels). 

There is a small window sliding over along the image, analyzing pixel by pixel. The data then flows to the convolution layer, which forms a funnel (compression of the identified features). 

In terms of image recognition, the first layer identifies the gradient, the second layer identifies the line, and the third layer identifies the shape, and so on, up to the level of a particular object. DFF is usually attached to the end of the convolution layer for future data processing.

## 10.2. Deconvolutional Neural Network (DNN) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann20.png"  width="200" height="210" > 

The **Deconvolution Neural Network** is the inverted version of Deep Convolutional Network. 

DNN can generate the vector as: [dog: 0, lizard: 0, horse: 0, cat: 1] after capturing the cat's picture, while DCN can draw a cat after getting this vector. 

<img align="left" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann20-1.png"  width="200" height="180"> 

**Where do we use Deconvolutional Network?**

You could tell the network “cat” and it will try to project it’s understanding of the features of a “cat”. 

DNN by itself is not entirely powerful, but when used in conjunction with some other structures, it can become very useful.

## 10.3. Deep Convolutional Inverse Graphics Network (DCIGN)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann21.png" width="280" height="200" > 

**Deep Convolutional Inverse Graphics Network** is a structure that connects a Convolutional Neural Network with a Deconvolutional Neural Network.
It might be confusing to call it a network when it is actually more of a Variational Auto-encoder (VAE).

**Where do we use Deep Convolutional Inverse Graphics Network?**

Most of these networks can be used in image processing and can process images that they have not been trained on before. 
They can be used to remove something from a picture, redraw it, or replace a horse with a zebra like the famous CycleGAN.
There are also many other types such as atrous convolutions and separable convolutions, that you can learn more about here.

# 11.	Generative Adversarial Neural Networks (GAN) - [Coursera](https://www.coursera.org/projects/generative-adversarial-networks-keras)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann22.png"  width="300" height="200" > 

The **Generative Adversarial Network** represents a dual network consisting of generators and differentiator. Imagine two networks in competition, each trying to outsmart the other. The generator tries to generate some data, and the differentiator tries to discern which are the samples and which ones are generated (code). 

As long as you can maintain the balance between the training of the two neural networks, this architecture can generate the actual image.

**Where do we use Generative Adversarial Network?**

They are used in text to **image generation** ([paper](), [code]()), **image to image translation** ([paper](), [code]()), **increasing image resolution** ([paper](), [code]()) and **predicting next video frame** ([paper](), [code]()).

# 12.	Liquid State Machine (LSM) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann23.png"  width="240" height="200" > 

**Liquid State Machines** are sparse neural networks whose activation functions are replaced (not all connected) by thresholds. When the threshold is reached, the cell accumulates the value information from the successive samples and the output freed, then again sets the internal copy to zero. 

**Where do we use Liquid State Machine?**

These neural networks are widely used in computer vision, speech recognition systems, but has no major breakthrough.

# 13.	Extreme Learning Machine (ELM) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann24.png"  width="240" height="200"> 

**Extreme Learning Machines** reduce the complexity behind a feedforward network by creating a sparse, random connection of hidden layers. 

They require less computer power, and the actual efficiency depends very much on tasks and data.

**Where do we use Extreme Learning Machine?**

It is widely used in batch learning, sequential learning, and incremental learning because of its fast and efficient learning speed, fast convergence, good generalization ability, and ease of implementation.
However, due to its memory-residency, and high space and time complexity, the traditional ELM is not able to train big data fast and efficiently.

# 14.	Echo State Neural Networks (ESN) - [Youtube](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj0waLEiN_qAhVIeawKHbpeBcMQtwIwBHoECAoQAQ&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DT12mA9h1VRs&usg=AOvVaw2OVyOK0knDdlfhBhOrAT8F)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann25.png" width="220" height="200"> 

**Echo status network** is a subdivision of a repeating network. Data passes through the input, and if multiple iterations are monitored, only the weight between hidden layers is updated after that.

**Where do we use Echo State Network?**

Besides multiple theoretical benchmarks, there is not any practical use of this Network. 

# 15.	Deep Residual Neural Networks (DRN) - [Coursera](https://www.coursera.org/lecture/convolutional-neural-networks/resnets-HAhz9)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann26.png"  width="380" height="180"> 

**Deep Residual Network** (or **ResNet**) passes parts of input values to the next level. This feature makes it possible to reach many layers (up to 300), but they are actually recurrent neural network without a clear delay. 

**Where do we use Deep Residual Network?**

As the Microsoft Research study proves, Deep Residual Networks can be used with a significantly importance in image recognition.

# 16.	Kohonen Neural Networks (KN)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann27.png"  width="220" height="200"> 

**Kohonen Network**, also known as Self-Organizing Map (SOM), introduces the "cell distance" feature. This network tries to adjust its cells to make the most probable response to a particular input. When a cell is updated, the closest cells are also updated.

They are not always considered "real" neural networks. 

**Where do we use Kohonen Network?**

<img align="left" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann27-1.png"  width="220" height="200"> 

Kohonen Network produces a low-dimensional (typically two-dimensional) representation of the input space, called a map, and is therefore a method to do dimensionality reduction. 

A [practical example](https://en.wikipedia.org/wiki/Self-organizing_map#/media/File:Synapse_Self-Organizing_Map.png) of appliance is this map (by Original uploader Denoir at en.wikipedia) showing U.S. Congress voting patterns. 

The input data was a table with a row for each member of Congress, and columns for certain votes containing each member's yes/no/abstain vote. 

The SOM algorithm arranged these members in a two-dimensional grid placing similar members closer together. 

# 17.	Support Vector Machine (SVM) Neural Networks - [Coursera](https://www.coursera.org/lecture/machine-learning/using-an-svm-sKQoJ)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann28.png"  width="220" height="200"> 

**Support Vector Machines** are used for binary categorical work and the result will be "yes" or "no" regardless of how many dimensions or inputs the network processes.

It uses a technique called the kernel trick to do some extremely complex data transformations, then figures out how to separate input data based on the defined output labels. 

**Where do we use Support Vector Machine?**

SVMs should be the first choice for any classification task, because is one of the most robust and accurate algorithm among the other classification algorithms.

# 18.	Neural Turing Machine (NTM) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann29.png"  width="220" height="200"> 

Neural networks are like black boxes - we can train them, get results, enhance them, but most of the actual decision paths are not visible to us.

The **Neurological Turing Machine (NTM)** is trying to solve this problem - it is an FF after extracting memory cells. Some authors also say that it is an abstract version of LSTM.

Memory is content based; this network can read memory based on the status quo, write memory, and also represents the Turing complete neural network.

**Where do we use Neural Turing Machine?**

A Neurological Turing Machine with a long short-term memory (LSTM) network controller can infer simple algorithms such as copying, sorting, and associative recall from examples alone.

Other open source implementations of NTMs exist but are not sufficiently stable for production use.
