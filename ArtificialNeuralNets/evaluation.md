# Evaluation

# :construction: ... Work in Progress ... :construction:

<div align="right">
<p> :calendar: Summer, 2020
:bust_in_silhouette: Author <a href="https://github.com/laviniaflorentina"> Lavinia Florentina </a> </p>
</div>

In order to evaluate the abilities of a machine learning system, we must design a quantitative measure of its performance. Usually this performance measure is specific to the task and it is being carried out by the system.

# Training Evaluation Methods

# Model Evaluation Methods

For tasks such as _classification_ we often measure **the accuracy** of the model, or **the error rate**. 

**Accuracy** is just the proportion of examples for which the model produces the correct output. **The error rate**, on the other hand, is the proportion of examples for which the model produces an incorrect output. We often refer to the error rate as the expected 0-1 loss. The 0-1 loss on a particular example is **0 if it is correctly classified** and **1 if it is not**. 

For tasks such as _density estimation_, we must use a different performance metric that gives the model a continuous-valued score for each example. The most common approach is to report the average log-probability the model assigns to some examples.



--------------------------

<img align="centre" src="https://media.giphy.com/media/4T1Sf6UvSXYyLJ5tUS/giphy.gif" width="400" height="400">

<div align="right">
<b> NEXT:   </b> 
<a href="https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/hardware_for_training.md#nut_and_bolt-hardware-parts-for-training-wrench" ><i> Hardware parts for training</i></a> 
</div>  

