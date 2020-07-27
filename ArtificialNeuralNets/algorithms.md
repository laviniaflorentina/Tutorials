# Algorithms by tasks

# :construction: ... Work in Progress ... :construction:

<div align="right">
<p> :calendar: Summer, 2020
:bust_in_silhouette: Author <a href="https://github.com/laviniaflorentina"> Lavinia Florentina </a> </p>
</div>

Content:

- [Classification]():
  - Perceptron
  - Logistic Regression
  - Naive Bayes Classification
  - Support Vector Machines (SVM)
  - k-nearest neighbor (KNN)
  - Decision trees 

- [Regression]():
  - Linear Regression 
  - Polynomial Regression 
  - Exponential Regression 
  - Logistic Regression 
  - Logarithmic Regression 
  
- [Clustering]():
  - k-Means
  
- [Dimensionality reduction]()

- [Visualization]():
  - PCA
--------------------------

## All these tasks can be classified by the ML category they belong to as following:
  
<div align="center">
<img align="center" src="https://miro.medium.com/max/1400/0*botktOR526S9maYd" width=700 height=300>
<p> <i>:copyright: Image found on <a href="https://medium.com/@priyalwalpita/types-of-machine-learning-556529ad6a23"> medium </a> </i> </p>
</div>  

## Classification - [RealPython Tutorial](https://realpython.com/logistic-regression-python/)

Classification is the task for identifying similarity among items in order to group them while having a name for that group (a label). 

There are several general steps you’ll take when you’re preparing your classification models:

1. Import packages, functions, and classes
2. Get data to work with and, if appropriate, transform it
3. Create a classification model and train (or fit) it with your existing data
4. Evaluate your model to see if its performance is satisfactory

A sufficiently good model that you define can be used to make further predictions related to new, unseen data. The above procedure is the same for classification and regression.

**When do we use Classification Algorithms?**

These algorithms are used for problems such as:
-	Email spam classification.
-	Sentiment analysis. 
- Facial key-points detection.
-	Pedestrian detection in automotive car driving.

There are different kinds of classification:
-	**Binary (Binomial) Classification**: when there are two possible outcomes. _**Example**_: Gender classification (Male/Female).
-	**Multi-class (Multinomial) Classification**: when there are more than two possible outcomes. Each sample is assigned to one and only one target group. _**Example**_: An animal can be _a cat_ or _dog_ but not both at the same time. 
-	**Multi-label Classification**: when each sample is mapped to a set of target labels (more than one class). _**Example**_: A text can be about fashion, a person, and location at the same time.

### Classification Algorithms

Classification Algorithms can be grouped as the following:

Binary Classification:

- **Perceptron**. In perceptron, we take weighted linear combination of input features and pass it through a thresholding function which outputs 1 or 0. 

<!---
``` python
import numpy as np
class Perceptron:    
    def fit(self, X, y, n_iter=100):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        # Add 1 for the bias term
        self.weights = np.zeros((n_features+1,))
        # Add column of 1s
        X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        for i in range(n_iter):
            for j in range(n_samples):
                if y[j]*np.dot(self.weights, X[j, :]) <= 0:
                    self.weights += y[j]*X[j, :]
    def predict(self, X):
        if not hasattr(self, 'weights'):
            print('The model is not trained yet!')
            return
        n_samples = X.shape[0]
        # Add column of 1s
        X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        y = np.matmul(X, self.weights)
        y = np.vectorize(lambda val: 1 if val > 0 else -1)(y)
        return y
    def score(self, X, y):
        pred_y = self.predict(X)
        return np.mean(y == pred_y)
--->

-	**Logistic regression**. In logistic regression, we take weighted linear combination of input features and pass it through a sigmoid function which outputs a number between 1 and 0. Moreover, unlike perceptron, which just tells us which side of the plane the point lies on, logistic regression gives a probability of a point lying on a particular side of the plane. 

<!---
``` python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
model = LogisticRegression(solver='liblinear', random_state=0)
model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)
model.predict_proba(x)
model.score(x, y)
```
--->

-	**Naive Bayes Classification**: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.05-Naive-Bayes.ipynb#scrollTo=olqAAQnoMtIR)

Multi-class classification:

-	**Support Vector Machines (SVM)**. There can be multiple hyperplanes that separate linearly separable data. SVM calculates the optimal separating hyperplane using concepts of geometry. 
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
-	**Kernel estimation**: k-Nearest Neighbor (KNN)
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

## Regression

Regression task is also known as a prediction task. It uses given sample datapoints of a situation in order to anticipate future outcomes of that situation.  

**When do we use Regression Algorithms?**

These algorithms are used for problems such as:
-	House price prediction
-	Forecast prediction
-	Stock predictions

### Regression Algorithms
Regression Algorithms can be grouped as the following:

-	**Linear Regression**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
-	**Polynomial Regression**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
-	**Exponential Regression**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

## Clustering

Clustering is the task for identifying similarity among items in order to group them – without having a name for that group (a label).

Determining the optimal number of clusters in a dataset is a fundamental issue in pursuing with a K-Means or K-medoids clustering algorithms. Other algorithms such as DBSCAN and OPTICS do not require this parameter, while hierarchical clustering avoids the problem altogether.

The optimal number of clusters can be identified from data by applying direct methods (such as elbow and silhouette) or statistical testing methods (such as gap statistic). There are many other methods that have been published for identifying the optimal number of clusters, but these are the most commons used ones. 

**When do we use Clustering Algorithms?**

These algorithms are used for problems such as:

### Clustering Algorithms

- **K-Means**: 

K-means clustering algorithm is also called flat clustering algorithm. It is classifying n objects into k groups (called clusters) of greatest possible distinction. A value is assigned from the beginning to k, meaning that you need to have a first guess on the number of clusters in your data. Then, each object goes to a specific cluster depending on its distance to a point that is declared to be the center of that cluster. 

In this algorithm, the data points are assigned to a cluster in such a manner that the sum of the squared distance between the data points and centroid would be minimum. It is to be understood that less variation within the clusters will lead to more similar data points within same cluster.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.11-K-Means.ipynb)  

- **K-Nearest Neighbor (KNN)**:  

This algorithm is used to solve the classification model problems. K-nearest neighbor or K-NN algorithm basically creates an imaginary boundary to classify the data. When new data points come in, the algorithm will try to predict that to the nearest of the boundary line.

Therefore, larger k value means smother curves of separation resulting in less complex models. Whereas, smaller k value tends to overfit the data and resulting in complex models.

**Note**: It’s very important to have the right k-value when analyzing the dataset to avoid overfitting and underfitting of the dataset.

Using the k-nearest neighbor algorithm we fit the historical data (or train the model) and predict the future.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1My8UggN12Opt_gscK3tl4VLhZkHiQSyX) and [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nholmber/google-colab-cs231n/blob/master/assignment1/knn.ipynb)  

- **DBSCAN**:

DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. This technique is one of the most common clustering algorithms which works based on density of object. The whole idea is that if a particular point belongs to a cluster, it should be near to lots of other points in that cluster.

It works based on two parameters: _Epsilon_ and _Minimum Points_.

**Epsilon** determine a specified radius that if includes enough number of points within, we call it dense area.

**Minimum Samples** determine the minimum number of data points we want in a neighborhood to define a cluster.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gurubux/CognitiveClass-ML/blob/master/Course_MachineLearningWithPython/5-Clustering/ML0101EN-Clus-DBSCN-weather-py-v1.ipynb#scrollTo=SvJqeKgGBruX)   

- **Hierarchical Cluster Analysis (HCA)**: 

Hierarchical clustering works by first putting each data point in their own cluster and then merging clusters based on some rule, until there are only the wanted number of clusters remaining. For this to work, there needs to be a distance measure between the data points. With this distance measure `d`, we can define another distance measure between the **clusters** U and V.

At each iteration of the algorithm two clusters that are closest to each other are merged. After this the distance between the clusters are recomputed, and then it continues to the next iteration.

Below is an example with a botanical dataset with 150 samples from three species. Each species appears in the dataset 50 times. Each sample point has 4 features, which are basically dimensions of the "leaves" of the flower.

We use the [seaborn](https://seaborn.pydata.org/index.html) library to both to compute the clustering and to visualize the result. The visualization consists of two parts: the *heatmap*, whose rows and/or columns may be reordered so as to have the elements of the same cluster next to each other; and the *dendrogram*, which shows the way the clusters were merged. The colors give the length of the corresponding features.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jttoivon/data-analysis-with-python-spring-2019/blob/master/clustering.ipynb#scrollTo=78nax17B9FGa)  

## Dimensionality reduction 

  - **PCA**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.09-Principal-Component-Analysis.ipynb#scrollTo=8gU1tEBi8vGM)
  
## Visualization  

  - **PCA**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.09-Principal-Component-Analysis.ipynb#scrollTo=8gU1tEBi8vGM)  

--------------------------

<img align="centre" src="https://media.giphy.com/media/4T1Sf6UvSXYyLJ5tUS/giphy.gif" width="400" height="400">

<div align="right">
<b> NEXT:  </b> 
<a href="https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/training.md#training" ><i> Training</i></a> 
</div>  

