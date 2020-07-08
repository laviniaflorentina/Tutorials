# Introduction

## :construction: ... Work in Progress ... :construction:

Content:

- [What is an Artificial Neural Network?]()
    - [What are the main components and why do we need each of them?]()
        - Weights, Bias and Layers
        - Activation Function: Linear Activation Function and Non-linear Activation Function (Sigmoid, Tanh & ReLU)
        - Derivatives
    - [Architectures of Artificial Neural Network]()

## What is an Artificial Neural Network?
**Artificial Neural networks (ANN)** are a set of algorithms, modeled in a similar way the human brain works, developed to recognize and predict patterns. They interpret given data through a machine perception, using labeling or collecting raw input. The patterns they recognize are numerical, expressed as vectors, and so is the output before having assigned a meaning (check this [video](https://www.youtube.com/watch?v=aircAruvnKk) explanation). 
Therefore, it is essential to convert the real-world input data, like images, sounds or text, into numerical values.

Most of the existing neural networks architectures are shown in the following picture:
![ANNs](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/ann1.png)

Deep learning is the name we use for _“stratified neural networks”_ = **networks composed of several layers**.

The layers are represented by a column vector and the elements of the vector can be thought of as _nodes_ (also named _neurons_ or _units_).

A **node (neuron)** is just **a place where computation happens**. It receives input from some other nodes, or from an external source and computes the outcome. Each input has an associated **weight (w)**, which is assigned on the basis of its relative importance to other inputs. Then, the output is computed by combining a set of **coefficients**, or **weights**, that either _amplify_ or _reduce_ that input depending on its importance. 

The input-weights are multiplied between each other and summed up. The sum is passed through a node’s so-called **activation function**, to determine whether or to what extent that signal should progress further through the network to affect the ultimate outcome. If the signal passes through, the neuron has been **activated**.

### What are the main components and why do we need each of them?

#### 1. Why do we need Weights, Bias and Layers?

**Weight** shows the strength of the particular node. In other words, the weight is the assigned significance of an input in comparison with the relative importance of other inputs.

A **bias** value allows you to shift the activation function curve up or down.

A neural network can usually consist of three types of nodes:

   - **Input Nodes** – they provide information from the outside world to the network and are referred to as the “Input Layer”. No computation is performed in any of the Input nodes – they just pass on the information to the hidden nodes.

   - **Hidden Nodes** – they have no direct connection with the outside world and form a so called “Hidden Layer”. They perform computations and transfer information from the input nodes to the output nodes.

   - **Output Nodes** – they are collectively referred to as the “Output Layer” and are responsible for computations and mapping information from the network to the outside world.

#### 2. Why do we need Activation Function?

Also known as Transfer Function, it is used to determine the output of neural network like yes or no. It maps the resulting values in between (0, 1) or (-1, 1) etc. (depending upon the type of function).

The Activation Functions can be basically divided into 2 types:

a. **Linear Activation Function**

The function is a line or linear. Therefore, the output of the function will not be restricted between any range.

       Equation: f(x) = x.              Range: (-∞,∞).

Not helpful with the complexity of data that is fed to the neural networks. Doesn’t matter how many neurons we link together the behavior will be the same.

b. **Non-Linear Activation Function**

The Nonlinear Activation Functions are the most used activation functions. They make it easy for the model to generalize and adapt to the variations of the data and to better categorize the output.

The most common non-linear activation functions are:

- Sigmoid or Logistic Activation Function 
- Tanh or hyperbolic tangent Activation Function
- ReLU (Rectified Linear Unit) Activation Function 

#### 3. Why the derivative/differentiation is being used?

We use differentiation in almost every part of Machine Learning and Deep Learning, because when updating the curve, we need to know in which direction and how much to change or update the curve depending upon the slope. 

In the following table it is a clear distinction and classification of some functions and their derivates.

![ANNs](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/ann2.png)

## Architectures of Artificial Neural Network
