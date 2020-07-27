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
  - PCA
--------------------------

## Classification - [RealPython Tutorial](https://realpython.com/logistic-regression-python/)

Classification is the task for identifying similarity among items in order to group them while having a name for that group (a label). 

<!---
<div align="right">
<img align="right" src="https://miro.medium.com/max/2886/1*xpRoFLy0BHsCr62Uf1FfTQ.png" width=500 height=300>
<p> <i>:copyright: Image found on <a href="https://miro.medium.com/max/2886/1*xpRoFLy0BHsCr62Uf1FfTQ.png"> medium </a> </i> </p>
</div>  
--->

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

```

-	**Logistic regression**. In logistic regression, we take weighted linear combination of input features and pass it through a sigmoid function which outputs a number between 1 and 0. Moreover, unlike perceptron, which just tells us which side of the plane the point lies on, logistic regression gives a probability of a point lying on a particular side of the plane. 

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

-	**Naive Bayes Classification**.

![[Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.05-Naive-Bayes.ipynb#scrollTo=olqAAQnoMtIR)

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
-	**Kernel estimation**: k-nearest neighbor (KNN)
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
•	House price prediction
•	Forecast prediction
•	Stock predictions


### Regression Algorithms
Regression Algorithms can be grouped as the following:

-	Linear Regression 

``` python
# Import required libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
# Read the CSV file :
data = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv")
data.head()
# Let's select some features to explore more :
data = data[["ENGINESIZE","CO2EMISSIONS"]]
# ENGINESIZE vs CO2EMISSIONS:
plt.scatter(data["ENGINESIZE"] , data["CO2EMISSIONS"] , color="blue")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()
# Generating training and testing data from our data:
# We are using 80% data for training.
train = data[:(int((len(data)*0.8)))]
test = data[(int((len(data)*0.8))):]
# Modeling:
# Using sklearn package to model data :
regr = linear_model.LinearRegression()
train_x = np.array(train[["ENGINESIZE"]])
train_y = np.array(train[["CO2EMISSIONS"]])
regr.fit(train_x,train_y)
# The coefficients:
print ("coefficients : ",regr.coef_) #Slope
print ("Intercept : ",regr.intercept_) #Intercept
# Plotting the regression line:
plt.scatter(train["ENGINESIZE"], train["CO2EMISSIONS"], color='blue')
plt.plot(train_x, regr.coef_*train_x + regr.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
# Predicting values:
# Function for predicting future values :
def get_regression_predictions(input_features,intercept,slope):
 predicted_values = input_features*slope + intercept
 return predicted_values
# Predicting emission for future car:
my_engine_size = 3.5
estimatd_emission = get_regression_predictions(my_engine_size,regr.intercept_[0],regr.coef_[0][0])
print ("Estimated Emission :",estimatd_emission)
# Checking various accuracy:
from sklearn.metrics import r2_score
test_x = np.array(test[['ENGINESIZE']])
test_y = np.array(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Mean sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
```
-	Polynomial Regression
```python
# Import required libraries:
import numpy as np
import matplotlib.pyplot as plt
# Generate datapoints:
x = np.arange(-5,5,0.1)
y_noise = 20 * np.random.normal(size = len(x))
y = 1*(x**3) + 1*(x**2) + 1*x + 3+y_noise
plt.scatter(x,y)
# Make polynomial data:
x1 = x
x2 = np.power(x1,2)
x3 = np.power(x1,3)
n = len(x1)
# Reshaping data:
x1_new = np.reshape(x1,(n,1))
x2_new = np.reshape(x2,(n,1))
x3_new = np.reshape(x3,(n,1))
# First column of matrix X:
x_bias = np.ones((n,1))
# Form the complete x matrix:
x_new = np.append(x_bias,x1_new,axis=1)
x_new = np.append(x_new,x2_new,axis=1)
x_new = np.append(x_new,x3_new,axis=1)
# Finding transpose:
x_new_transpose = np.transpose(x_new)
# Finding dot product of original and transposed matrix :
x_new_transpose_dot_x_new = x_new_transpose.dot(x_new)
# Finding Inverse:
temp_1 = np.linalg.inv(x_new_transpose_dot_x_new)# Finding the dot product of transposed x and y :
temp_2 = x_new_transpose.dot(y)
# Finding coefficients:
theta = temp_1.dot(temp_2)
theta
# Store coefficient values in different variables:
beta_0 = theta[0]
beta_1 = theta[1]
beta_2 = theta[2]
beta_3 = theta[3]
# Plot the polynomial curve:
plt.scatter(x,y)
plt.plot(x,beta_0 + beta_1*x1 + beta_2*x2 + beta_3*x3,c="red")
# Prediction function:
def prediction(x1,x2,x3,beta_0,beta_1,beta_2,beta_3):
 y_pred = beta_0 + beta_1*x1 + beta_2*x2 + beta_3*x3
 return y_pred
 
# Making predictions:
pred = prediction(x1,x2,x3,beta_0,beta_1,beta_2,beta_3)
 
# Calculate accuracy of model:
def err(y_pred,y):
 var = (y - y_pred)
 var = var*var
 n = len(var)
 MSE = var.sum()
 MSE = MSE/n
 
 return MSE
# Calculating the error:
error = err(pred,y)
error
```
-	Exponential Regression
```python
# Import required libraries:
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Dataset values :
day = np.arange(0,8)
weight = np.array([251,209,157,129,103,81,66,49])
# Exponential Function :
def expo_func(x, a, b):
 return a * b ** x
#popt :Optimal values for the parameters
#pcov :The estimated covariance of popt
popt, pcov = curve_fit(expo_func, day, weight)
weight_pred = expo_func(day,popt[0],popt[1])
# Plotting the data
plt.plot(day, weight_pred, 'r-')
plt.scatter(day,weight,label='Day vs Weight')
plt.title("Day vs Weight a*b^x")
plt.xlabel('Day')
plt.ylabel('Weight')
plt.legend()
plt.show()
# Equation
a=popt[0].round(4)
b=popt[1].round(4)
print(f'The equation of regression line is y={a}*{b}^x')
```
-	Logistic Regression
```python
# Import required libraries:
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
# Generating dataset:
# Y = A*sin(B(X + C)) + D
# A = Amplitude
# Period = 2*pi/B
# Period = Length of One Cycle
# C = Phase Shift (In Radian)
# D = Vertical Shift
X = np.linspace(0,1,100) #(Start,End,Points)
# Here…
# A = 1
# B= 2*pi
# B = 2*pi/Period
# Period = 1
# C = 0
# D = 0
Y = 1*np.sin(2*np.pi*X)
# Adding some Noise :
Noise = 0.4*np.random.normal(size=100)
Y_data = Y + Noise
plt.scatter(X,Y_data,c="r")
# Calculate the value:
def calc_sine(x,a,b,c,d):
 return a * np.sin(b* ( x + np.radians(c))) + d
# Finding optimal parameters :
popt,pcov = curve_fit(calc_sine,X,Y_data)
# Plot the main data :
plt.scatter(X,Y_data)# Plot the best fit curve :
plt.plot(X,calc_sine(X,*popt),c="r")
plt.show()
# Check the accuracy :
Accuracy =r2_score(Y_data,calc_sine(X,*popt))
print (Accuracy)
# Function to calculate the value :
def calc_line(X,m,b):
 return b + X*m
# It returns optimized parametes for our function :
# popt stores optimal parameters
# pcov stores the covarience between each parameters.
popt,pcov = curve_fit(calc_line,X,Y_data)
# Plot the main data :
plt.scatter(X,Y_data)
# Plot the best fit line :
plt.plot(X,calc_line(X,*popt),c="r")
plt.show()
# Check the accuracy of model :
Accuracy =r2_score(Y_data,calc_line(X,*popt))
print ("Accuracy of Linear Model : ",Accuracy)
```
-	Logarithmic Regression
```python
# Import required libraries:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
# Dataset:
# Y = a + b*ln(X)
X = np.arange(1,50,0.5)
Y = 10 + 2*np.log(X)
#Adding some noise to calculate error!
Y_noise = np.random.rand(len(Y))
Y = Y +Y_noise
plt.scatter(X,Y)
# 1st column of our X matrix should be 1:
n = len(X)
x_bias = np.ones((n,1))
print (X.shape)
print (x_bias.shape)
# Reshaping X :
X = np.reshape(X,(n,1))
print (X.shape)
# Going with the formula:
# Y = a + b*ln(X)
X_log = np.log(X)
# Append the X_log to X_bias:
x_new = np.append(x_bias,X_log,axis=1)
# Transpose of a matrix:
x_new_transpose = np.transpose(x_new)
# Matrix multiplication:
x_new_transpose_dot_x_new = x_new_transpose.dot(x_new)
# Find inverse:
temp_1 = np.linalg.inv(x_new_transpose_dot_x_new)
# Matrix Multiplication:
temp_2 = x_new_transpose.dot(Y)
# Find the coefficient values:
theta = temp_1.dot(temp_2)
# Plot the data:
a = theta[0]
b = theta[1]
Y_plot = a + b*np.log(X)
plt.scatter(X,Y)
plt.plot(X,Y_plot,c="r")
# Check the accuracy:
Accuracy = r2_score(Y,Y_plot)
print (Accuracy)
```

--------------------------

<img align="centre" src="https://media.giphy.com/media/4T1Sf6UvSXYyLJ5tUS/giphy.gif" width="400" height="400">

<div align="right">
<b> NEXT:  </b> 
<a href="https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/training.md#training" ><i> Training</i></a> 
</div>  

