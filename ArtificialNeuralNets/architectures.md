# Artificial Neural Networks Architectures

# :construction: ... Work in Progress ... :construction:

Content

- [Introduction]()
- [Feedforward Neural Networks (FF)]():
  - [Singe-layer Perceptron (Perceptron)]()
  - [Multi-Layer Perceptron (MLP)]()
- [Radial Basis Function (RBF) Neural Network ]()
- [Deep Feedforward Network (DFF)]()
- [Recurrent Neural Networks (RNN)]():
  - [Long-Short Term Memory (LSTM) Neural Networks]()
  - [Gated Recurrent Unit (GRU) Neural Networks]()
  
--------------------

## Introduction

The neural network architecture represents the way in which the elements of the network are binned together. The architecture will define the behavior of the network. Considering their architecture, Artificial Neural Networks can be classified as follows:

# 1.	Feedforward Neural Networks (FF)

Feedforward Neural Networks were the first type of Artificial Neural Network. These kinds of nets have a large variance but there are mainly two pioneers:

## 1.1.	Singe-layer Perceptron (Perceptron)

The **perceptron** is the simplest type of feedforward neural network, compost of only one neuron, where:  

  -	It takes some inputs, <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/xi.png" alt="inpits" width="25" height="25">, and each of them is multiplied with their related weight, <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/wi.png" alt="inpits" width="25" height="25"> : 

 ![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann3.png)

  - Then, it sums them up: <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/sum_n.png" alt="inpits" width="50" height="50">, particularly for the example above, we will have: <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/sum_5.png" alt="inpits" width="50" height="50">.

  -	The **activation function** (usually this is a _logistic function_) squashes the above summation and passes it to the output layer.

**Where do we use Perceptron?**

Perceptron is good for classification and decision-making systems because of the logistic function properties. That being the case, it is usually used to classify the data into two parts and it is also known as a Linear Binary Classifier. 

The issue with individual neurons rapidly arises when trying to solve every day - life problems due to the real-world complexity. Determined to address this challenge, researchers observed that by combining neurons together, their decision is basically combined getting as a result insight from more elaborated data. 

And so, the creation of an Artificial Neural Network with more neurons seems to be the answer.

## 1.2.	Multi-Layer Perceptron (MLP)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann4.png"> 

The **multi-layer perceptron** is a type of feedforward neural network that introduces the _multiple neurons_ design, where:

  - all nodes are fully connected (each node connects all the nodes from the next layer);
  - information only travels forward in the network (no loops);
  - there is one hidden layer (if present).

**Where do we use Multi-Layer Perceptron?**

Contrasting single layer perceptrons, MLPs are capable to manipulate non-linearly separable data. 

Therefore, these nets are used in many applications, but not by themselves. Most of the time they stand as a pillar of support in the construction of the next neural networks’ architecture. 

# 2.	Radial Basis Function (RBF) Neural Network 

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann5.png"> 

**Radial Basis Function Networks** are feedforward nets _with a different activation function_ in place of the logistic function, named the **radial basis function (RBF)**. 

Intuitively, RBF answers the question _“How far is the target from where we are?”_. More technically, it is a real-valued function defined as the difference between the input and some fixed point, called the **center**. 

The RBF chosen is usually a Gaussian, and it behaves like in the following example: 

![gif](https://www.digitalvidya.com/wp-content/uploads/2019/01/Image-3.gif)

RBN is strictly limited to have **exactly one hidden layer** (green dots in the related figure). Here, this hidden layer is known as a **feature vector**.

Moreover, Wikipedia says:

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann5-1.png)

**Where do we use Radial Basis Function Network?**

RBF nets are used in [function approximation](https://github.com/andrewdyates/Radial-Basis-Function-Neural-Network/blob/master/CSE779LabReport2.pdf) ([paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.6019&rep=rep1&type=pdf) & [code](https://github.com/thomberg1/UniversalFunctionApproximation)), time series prediction ([paper](https://dl.acm.org/doi/pdf/10.1145/3305160.3305187) & [code](https://www.mathworks.com/matlabcentral/fileexchange/66216-mackey-glass-time-series-prediction-using-radial-basis-function-rbf-neural-network)), and machine/system control (for example as a replacement of Partial Integral Derivative controllers). 

# 3.	Deep Feedforward Network (DFF) - [Coursera](https://www.coursera.org/lecture/ai/deep-feed-forward-neural-networks-kfTED)

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann6.png"> 

**Deep Feedforward Neural Networks** are feedforward nets with _more than one hidden layer_.

It follows the rules:
  - all nodes are fully connected;
  - activation flows from input layer to output, without back loops;
  - there is **more than one layer** between input and output (hidden layers – green dots).

When training the traditional FF model, only a small amount of error information passes to the next layer. With more layers, DFF is able to learn more about errors; however, it becomes impractical as the amount of training time required increases.

Nowadays, a series of effective methods for training DFF have been developed, which have formed the core of modern machine learning systems and enable the functionality of feedforward neural networks.

**Where do we use Deep Feed-Forward Network?**

DFF is being used for automatic language identification (the process of automatically identifying the language spoken or written: [paper](https://static.googleusercontent.com/media/research.google.com/en/pubs/archive/42538.pdf), [paper](https://arxiv.org/pdf/1708.04811.pdf) & [code](https://github.com/HPI-DeepLearning/crnn-lid)), acoustic modeling for speech recognition ([thesis](https://mi.eng.cam.ac.uk/~mjfg/thesis_cw564.pdf), [paper](https://arxiv.org/pdf/1809.02108.pdf) & [code](https://github.com/amitai1992/AutomatedLipReading)), and other.

# 4.	Recurrent Neural Networks (RNN) – [YouTube](https://www.youtube.com/watch?v=QciIcRxJvsM) 

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann7.png"> 

The **Recurrent Neural Networks (RNN)** introduce the _recurrent cells_, a special type of cells located in the hidden layer (blue dots) and responsible of receiving its own output with a fixed delay — for one or more iterations creating loops. Apart from that, this network is like a usual FF net and so it follows similar rules:

  - all nodes are fully connected;
  - activation flows from input layer to output, with back loops;
  - there is more than one layer between input and output (hidden layers).

**Where do we use Recurrent Neural Networks?**

RNN is mainly used when the context is important — when decisions from past iterations or samples can influence the current ones, such as sentiment analysis ([paper](https://arxiv.org/pdf/1902.09314v2.pdf) & [code](https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/SentimentAnalysisProject)).

The most common examples of such contexts are texts — a word can be analyzed only in the context of previous words or sentences. RNNs can process texts by “keeping in mind” ten previous words.

[More on RNN](https://github.com/kjw0612/awesome-rnn) - A curated list of resources dedicated to recurrent neural networks.

## 4.1.	Long-Short Term Memory (LSTM) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann8.png"> 

**LSTM** is a subcategory of RNN. They introduce a _memory cell_ – a special cell that can store and recall facts from time dependent data sequences. (tutorial like [paper](https://arxiv.org/pdf/1909.09586.pdf))

Memory cells are actually composed of a couple of elements — called gates, that are recurrent and control how information is being remembered and forgotten. 

  -	The input Gate determines how much of the last sample is stored in memory; 
  -	The output gate adjusts the amount of data transferred to the next level; 
  -	The Forget Gate controls the rate at which memory is stored.

Note that there are **no activation functions** between blocks.

**Where do we use Long-Short Term Memory Network?**

LSTM networks are used when we have timeseries data, such as: video frame processing ([paper](https://arxiv.org/pdf/1909.05622.pdf) & [code](https://github.com/matinhosseiny/Inception-inspired-LSTM-for-Video-frame-Prediction)), writing generator ([article w/ code](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/), [article w/ code](https://medium.com/towards-artificial-intelligence/sentence-prediction-using-word-level-lstm-text-generator-language-modeling-using-rnn-a80c4cda5b40)) as it can “keep in mind” something that happened many frames/ sentences ago. 

## 4.2.	Gated Recurrent Unit (GRU) Neural Networks

<img align="right" src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann9.png"> 

**GRU** is a subcategory of RNN. GRUs are similar with LSTMs, but with a different type of gates. The lack of output gate makes it easier to repeat the same output for a concrete input multiple time and, therefore, they are less resource consuming than LSTMs and have similar performance.

**Where do we use Gated Recurrent Unit?**

They are currently used in similar applications as LSTMs.

