# Algorithms by tasks

# :construction: ... Work in Progress ... :construction:

<div align="right">
<p> :calendar: Summer, 2020
:bust_in_silhouette: Author <a href="https://github.com/laviniaflorentina"> Lavinia Florentina </a> </p>
</div>

Content:

--------------------------

## Classification

Classification is the task for identifying similarity among items in order to group them while having a name for that group (a label). 

<div align="right">
<img align="right" src="https://miro.medium.com/max/2886/1*xpRoFLy0BHsCr62Uf1FfTQ.png" width=500 height=300>
  
<p> <i>:copyright: Image found on <a href="https://miro.medium.com/max/2886/1*xpRoFLy0BHsCr62Uf1FfTQ.png"> medium </a> </i> </p>
</div> 

**When do we use Classification Algorithms?**

These algorithms are used for problems such as:
-	Email spam classification.
-	Sentiment analysis. 
- Facial key-points detection.
-	Pedestrian detection in automotive car driving.

There are different kinds of classification:
-	**Binary Classification**: when there are two possible outcomes. _**Example**_: Gender classification (Male/Female).
-	**Multi-class Classification**: when there are more than two possible outcomes. Each sample is assigned to one and only one target group. _**Example**_: An animal can be _a cat_ or _dog_ but not both at the same time. 
-	**Multi-label Classification**: when each sample is mapped to a set of target labels (more than one class). _**Example**_: A text can be about fashion, a person, and location at the same time.

### Classification Algorithms

Classification Algorithms can be grouped as the following:

Binary Classification:

- **Perceptron**. In perceptron, we take weighted linear combination of input features and pass it through a thresholding function which outputs 1 or 0. 

``` python



```

-	**Logistic regression**. In logistic regression, we take weighted linear combination of input features and pass it through a sigmoid function which outputs a number between 1 and 0. Moreover, unlike perceptron, which just tells us which side of the plane the point lies on, logistic regression gives a probability of a point lying on a particular side of the plane. 

``` python



```

-	**Naive Bayes Classification**.

``` python



```

Multi-class classification:

-	**Support Vector Machines (SVM)**. There can be multiple hyperplanes that separate linearly separable data. SVM calculates the optimal separating hyperplane using concepts of geometry.
  -	Least squares support vector machines
``` python

# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

# loading the iris dataset 
iris = datasets.load_iris() 

# X -> features, y -> label 
X = iris.data 
y = iris.target 

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 

# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 

# model accuracy for X_test 
accuracy = svm_model_linear.score(X_test, y_test) 

# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 

```
-	**Kernel estimation**
  -	k-nearest neighbor 
``` python

# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

# loading the iris dataset 
iris = datasets.load_iris() 

# X -> features, y -> label 
X = iris.data 
y = iris.target 

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 

# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 

# accuracy on X_test 
accuracy = knn.score(X_test, y_test) 
print accuracy 

# creating a confusion matrix 
knn_predictions = knn.predict(X_test) 
cm = confusion_matrix(y_test, knn_predictions) 

```
-	**Decision trees**
  -	Random forests
``` python

# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

# loading the iris dataset 
iris = datasets.load_iris() 

# X -> features, y -> label 
X = iris.data 
y = iris.target 

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 

# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 

# creating a confusion matrix 
cm = confusion_matrix(y_test, dtree_predictions) 

```

--------------------------

<img align="centre" src="https://media.giphy.com/media/4T1Sf6UvSXYyLJ5tUS/giphy.gif" width="400" height="400">

<div align="right">
<b> NEXT:  </b> 
<a href="https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/training.md#training" ><i> Training</i></a> 
</div>  

