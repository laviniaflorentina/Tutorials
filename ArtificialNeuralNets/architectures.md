# Artificial Neural Networks Architectures

# :construction: ... Work in Progress ... :construction:

Content


--------------------

## Introduction

The neural network architecture represents the way in which the elements of the network are binned together. The architecture will define the behavior of the network. Considering their architecture, Artificial Neural Networks can be classified as follows:

# 1.	Feedforward Neural Networks (FF)

Feedforward Neural Networks were the first type of Artificial Neural Network. These kinds of nets have a large variance but there are mainly two pioneers:

## 1.1.	Singe-layer Perceptron (Perceptron)

The perceptron is the simplest type of feedforward neural network, compost of only one neuron, where:  

•	It takes some inputs, <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/xi.png" alt="inpits" width="25" height="25">, and each of them is multiplied with their related weight, <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/wi.png" alt="inpits" width="25" height="25"> :

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann3.png)

•		Then, it sums them up: <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/sum_n.png" alt="inpits" width="50" height="50">, particularly for the example above, we will have: <img src="https://github.com/laviniaflorentina/Tutorials/blob/master/images/sum_5.png" alt="inpits" width="50" height="50">.

•	The **activation function** (usually this is a _logistic function_) squashes the above summation and passes it to the output layer.

**Where do we use Perceptron?**

Perceptron is good for classification and decision-making systems because of the logistic function properties. That being the case, it is usually used to classify the data into two parts and it is also known as a Linear Binary Classifier. 

The issue with individual neurons rapidly arises when trying to solve every day - life problems due to the real-world complexity. Determined to address this challenge, researchers observed that by combining neurons together, their decision is basically combined getting as a result insight from more elaborated data. 

And so, the creation of an Artificial Neural Network with more neurons seems to be the answer.

## 1.2.	Multi-Layer Perceptron (MLP)

The multi-layer perceptron is a type of feedforward neural network that introduces the multiple neurons design, where:

•	 all nodes are fully connected (each node connects all the nodes from the next layer);

•	 information only travels forward in the network (no loops);

•	 there is one hidden layer (if present).

![](https://github.com/laviniaflorentina/Tutorials/blob/master/images/ann4.png)

**Where do we use Multi-Layer Perceptron?**

Contrasting single layer perceptrons, MLPs are capable to manipulate non-linearly separable data. 

Therefore, these nets are used in many applications, but not by themselves. Most of the time they stand as a pillar of support in the construction of the next neural networks’ architecture. 

# 2.	Radial Basis Function (RBF) Neural Network 

**Radial Basis Function Networks** are feedforward nets _with a different activation function_ in place of the logistic function, named the **radial basis function (RBF)**. The RBF chosen is usually a Gaussian, and it behaves like in the following image: 

![](https://www.digitalvidya.com/wp-content/uploads/2019/01/Image-3.gif) 

Intuitively, RBF answers the question _“How far is the target from where we are?”_. More technically, it is a real-valued function defined as the difference between the input and some fixed point, called the center.


