# Python code structure 

Content:
- [Introduction](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/Python%20code%20structure.md#introduction)
- [Python Input](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/Python%20code%20structure.md#python-input)
- [Python Output](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/Python%20code%20structure.md#python-output)
- [Python Imports: Useful libraries/modules to import](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/Python%20code%20structure.md#python-import-useful-librariesmodules-to-import)

## Introduction 

Here is a basic python program that plots the graph of the function f: R → R , where  f(x)= √x

![Example of a basic py code](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/code_structure.png)

As shown in the example above, the file can contain _imports, defined functions, built-in functions_, and so on. Almost any code you'll write will have an input, output and imports, along with the main commands and functions.

In order to become a python user you need to be aware of the integrated tools available for you to apply in your 
personal/school/work projects (Data-Analysis, Data processing, etc. …). 
 
## Python Input

Until now, the value of variables was defined. To allow flexibility in the program, sometimes we might want to take the input from the user. In Python, the input() function allows this. 
The syntax for input() is:

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/code_structure2.png)

Where ‘Your name is: ‘ can be replaced with what you need from the user. For example:

``` python 
input ("Insert a number: ")
```

The entered value is automatically taken by the program as a string. 

It is important to know what kind of input you are expecting from the user. 

If you need a string – the above method works, but if you need an integer or a float to proceed with further calculations, you have to encapsulate the input() into _int(input( … ))_ or _float(input( … ))_. 

**Bonus** you can directly calculate a string operation using _eval()_ on the input() as in the example below: 

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/code_structure3.png)

## Python Output
The _print()_ function is used to output data to the screen. We can also output [data to a file](https://www.programiz.com/python-programming/file-operation) (useful when run the code on HPC). 

``` python 
print ("My first print using Py")
print ('My send print')
```

This example shows that you can use both “ ” and ‘ ’ to print a string.
The following example shows how to print a description of a variable together with that variable.

``` python 
x = 2
print ('The value of the variable is:', x)
```

**Output formatting.** If you want a more rigorous output you can do this by using _str.format()_ method.
![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/code_structure1.png)

## Python Import. Useful libraries/modules to import

Some commands might run without any imported libraries (for example “print(‘Hello World!’)”, or some basic calculations _a+b, a*b_) but most of the time you will need to use specific packages called libraries.

Here are some of the most common used libraries (click on the name to access their official documentation):

### 1. [Matplotlib](https://matplotlib.org) is a Python library used to write 2-dimensional graphs and plots.

-	Often, **mathematic or scientific** applications require more than single axes in a representation.   
-	This library helps us to **build multiple plots at a time**. 
-	You can, however, use Matplotlib to **manipulate different characteristics of figures** as well (like shown in the example: _linewidth, marker type, color, etc._)

How to call it: 
``` python 
import matplotlib.pyplot as plt
```

How to use it:
``` python 
# Create an empty xOy figure
fig = plt.figure()  # an empty figure with no Axes
fig, ax = plt.subplots()  # a figure with a single Axes

# Calculate and plot x^2 for the first 100 natural consecutive numbers:
for x in range(1,101):
  ax.plot(x, x*x, ‘bo’)

# Create the graph labels
plt.xlabel('x label')
plt.ylabel('x^2 label')
plt.title("Plot Title")

# Print the final plot
plt.show()
```
**Note**: you can change the highlighted part depending on your needs.

### 2. [Numpy](https://numpy.org) provides good support for different dimensional array objects as well as for matrices.

-	Not only confined to **provide arrays**, but it also provides a variety of tools to **manage these arrays**. 
-	It is fast, efficient, and really **good for managing matrices and arrays**.
-	Numpy provides such functionalities that are comparable to MATLAB. They both allow users to get **faster with operations**.

How to call it:
``` python 
import numpy as np
```

How to use it ([more examples](https://www.geeksforgeeks.org/numpy-in-python-set-1-introduction/)):
![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/code_structure4.png)
**Note**: .shape, .size, .ndim are features of the numpy package.

### 3. [Scipy]() is a python library that is used for mathematics, science, and engineering computation.

-	It can operate on an array of NumPy library.
-	Very suitable **for machine learning**, because contains a variety of sub-packages which help to solve the most common issue related to Scientific Computation.
-	It contains the following sub-packages:

    -	File input/output - **[scipy.io](https://docs.scipy.org/doc/scipy/reference/io.html)**
    -	Special Function - **[scipy.special]()**
    -	Linear Algebra Operation - **[scipy.linalg]()**
    -	Interpolation - **[scipy.interpolate]()**
    -	Optimization and fit - **[scipy.optimize]()**
    -	Statistics and random numbers - **[scipy.stats]()**
    -	Numerical Integration - **[scipy.integrate]()**
    -	Fast Fourier transforms - **[scipy.fftpack]()**
    -	Signal Processing - **[scipy.signal]()**   
    -	Image manipulation – **[scipy.ndimage]()**
    
    **Note**: SciPy sub-packages need to be imported separately.

For example, SciPy special function includes _Cubic Root, Exponential, Log sum Exponential, Lambert, Permutation and Combinations, Gamma, Bessel, hypergeometric, Kelvin, beta, parabolic cylinder, Relative Error Exponential,_ etc. ...

How to call and use it:

a). the special functions sub-package:
``` python 
from scipy.special import cbrt # for the Cubic Function
from scipy.special import exp10 # for the Exponential Function

# Calculate the cubic root of a given number
a = int(cbrt(27))
print("The cubic root is: ",a)

# Calculate the exponential
e = int(exp10(3))
print("The exponential is: ",e)
```
![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/code_structure5.png)

b). features of the library for image processing: 
``` python 
from scipy import ndimage, misc
from matplotlib import pyplot as plt

# Get face image of panda from misc package
panda = misc.face()

# Rotation function of SciPy for image at 45 degree
panda_rotated = ndimage.rotate(panda, 45)

# Show the original image and the rotated image
fig, (ax1,ax2) = plt.subplots(1, 2)
ax1.imshow(panda)
ax2.imshow(panda_rotated)
```

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/code_structure6.png)

### 4. [Panda](https://pandas.pydata.org/docs/user_guide/index.html) is a fast, demonstrative package that can be used to easily manipulate datasets.

-	Provides us with many Series and DataFrames. 
-	You can easily **organize, explore, represent, and manipulate data**.
-	Pandas can support **Excel, CSV, JSON, HDF5**, and many other formats. 
-	In fact, **it allows you to merge different databases** at a time.
-	It has a collection of built-in tools that allows you to both **read and write data** in databases, as well. 

How to call and use it:
``` python 
import pandas as pd
import pandas.util.testing as tm
import seaborn as sns

# Different ways to download and read your file 
# 1). from a public link
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# 2). from your DropBox account
!wget https://www.dropbox.com/blalbapath/file.csv -q -nc
data = pd.read_csv('file.csv')

# 3). if it is a sample dataset (for this you need to import seaborn package)
data = sns.load_dataset('iris')

# List the columns' names of the .csv file
print ("These are the columns: \n", data.columns)

# Print the beginning of the table to check how it looks like
data.head() # you can indicate between () the number of rows to be shown
```
![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/code_structure7.png)
**Note**: Use only one of the 3 methods. I wrote all the commands just to give an example for each of them.

### Other modules and libraries can be found [here](https://docs.python.org/3/library/). 

### Other Python Tutorials:

[DataCamp](https://www.datacamp.com/?utm_source=learnpython_com&utm_campaign=learnpython_tutorials) has tons of great interactive [Python Tutorials](https://www.datacamp.com/courses/?utm_source=learnpython_com&utm_campaign=learnpython_tutorials) covering data manipulation, data visualization, statistics, machine learning, and more;

Read [Python Tutorials and References](https://www.afterhoursprogramming.com/?article=181) course from After Hours Programming.

