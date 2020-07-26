# Algorithms by tasks

# :construction: ... Work in Progress ... :construction:

<div align="right">
<p> :calendar: Summer, 2020
:bust_in_silhouette: Author <a href="https://github.com/laviniaflorentina"> Lavinia Florentina </a> </p>
</div>

Content:

--------------------------

## Classification

Classification is the task for identifying similarity among items in order to group them while having a name for that group (a label). There are different kinds of classification:

<img align="right" src="https://miro.medium.com/max/2800/1*PM4dqcAe6N7kWRpXKwgWag.png" width=580 height=300>

<div align="right">
<p> <i>:copyrights: Image found on <a href="https://towardsdatascience.com/machine-learning-classifiers-a5cc4e1b0623" > towardsdatascience.com</a> </i> </p>
</div> 


-	**Binary Classification**: when there are two possible outcomes. _**Example**_: Gender classification (Male/Female).
-	**Multi-class Classification**: when there are more than two possible outcomes. Each sample is assigned to one and only one target group. _**Example**_: An animal can be _a cat_ or _dog_ but not both at the same time. 
-	**Multi-label Classification**: when each sample is mapped to a set of target labels (more than one class). _**Example**_: A text can be about fashion, a person, and location at the same time.

**When do we use Classification Algorithms?**

These algorithms are used for problems such as:

-	Email spam classification.
-	Sentiment analysis. 
- Facial key-points detection.
-	Pedestrian detection in automotive car driving.

### Classification Algorithms

Classification Algorithms can be grouped as the following:

Binary Classification:

- **Perceptron**. In Perceptron, we take weighted linear combination of input features and pass it through a thresholding function which outputs 1 or 0. 

-	**Logistic regression**. In Logistic regression, we take weighted linear combination of input features and pass it through a sigmoid function which outputs a number between 1 and 0. Unlike perceptron, which just tells us which side of the plane the point lies on, logistic regression gives a probability of a point lying on a particular side of the plane. 

-	**Naive Bayes**.

Multi-class classification:

-	**Support Vector Machines (SVM)**. There can be multiple hyperplanes that separate linearly separable data. SVM calculates the optimal separating hyperplane using concepts of geometry.
  -	Least squares support vector machines
-	**Kernel estimation**
  -	k-nearest neighbor 
-	**Decision trees**
  -	Random forests


--------------------------

<img align="centre" src="https://media.giphy.com/media/4T1Sf6UvSXYyLJ5tUS/giphy.gif" width="400" height="400">

<div align="right">
<b> NEXT:  </b> 
<a href="https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/training.md#training" ><i> Training</i></a> 
</div>  

