# :nut_and_bolt: Hardware Parts for Training :wrench:
# :construction: ... Work in Progress ... :construction:
Content

- [Why bother? ]() 

- [About **CPU, GPU, TPU, FPGA**]()

- [How to use these resources in your code?]()

  - In Browser (Google Colab or Jupyter Notebooks)

  - On HPC (or other High Performance Computing) 

----------------------------
## Why bother? 

When dealing with an AI model it is not enough to know training techniques and algorithms. It is also necessary to become aware of the hardware parts involved in order to allocate enough resources for the training. 

The modern AI problems created complex tasks, and so a neural network might be processing huge dimensions of data, sometimes implying hundreds of thousands of neurons. This kind of process is very time and memory consuming, which is why the hardware supplies are an important factor.

## About CPU, GPU, TPU, FPGA

**CPU** (stands for **Central Processing Unit**) or **Multi-core processors**, are general-purpose processors with several cores and multithreading, having sufficient power for organizing and computing high-performance calculations. This architecture is not a specifically designed solution for Neural Networks, but it can be used in training small AI models.

**GPU** (stands for **Graphical Processor Unit**), also known as videoboards, it is originally developed by NVIDIA for 3D graphics applications such as games. GPUs have many specialized processor units that computes mathematical operations in parallel and have high-speed memory to store the results. 
Moreover, the company created a library in 2007, called CUDA, making possible to process general computations on a GPU as well. In this way, running models on GPUs improve the training speed of a neural network is thousand and thousand times faster compared to regular CPU.

**TPU** (stands for **Tensor Processing Unit**) is developed by Google specifically for increasing the neural network processing performance. The company releases the first TPU in 2016, and now TensorFlow APIs and graph operators are available for everyone. This architecture needs fewer resources to make a huge number of computations. 

**FPGA** (stands for **Field-Programmable Gate Arrays**) makes it easier to develop hardware, frameworks, and software for building effective neural network systems. The advantage of this approach is that it gives high performance and allows you to change the architecture of the neural network for a specific task.

**Special processors.** Many companies are now working on creating effective solutions for training neural networks using specialized processors. This makes it possible to realize the necessary computing power for a certain class of tasks, such as: _voice recognition, auto control, image, and video recognition_. 

With less power consumption, such specialized solutions will give better results compared to GPU and TPU, and these solutions are compact for placement in a small form factor. 

## How to use these resources in your code?

CPU is being used as default everytime unless otherwise specified by user. 

### In Browser (Google Colab or Jupyter Notebooks)

Google Colaboratory and Jupyter Notebook are the easiest to use online environments for running Python code, and additionally they give you free access to GPUs and TPUs. 

Check-out [this tutorial](https://github.com/laviniaflorentina/Tutorials/blob/master/Python/online_in_browser.md#runtime-environment-options) for more details on how to set up GPU/TPU for your code.

### On HPC (or other High Performance Computing)

The good thing about having access to an HPC is that they have huge numbers of core processors that you can play with. Obvoiusly, the first thing would be to check how many such processors are available.

For University of North Texas students: [check available resources here](https://hpc.unt.edu/userguide#Compiling_Software).

