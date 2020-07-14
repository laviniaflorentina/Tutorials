# What is Machine Learning (ML)?

Content:

[Description](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#description)

[Machine Learning Categories:](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#machine-learning-categories)
  1. [Supervised Learning](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#1supervised-learning)
  2. [Unsupervised Learning](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#2unsupervised-learning)
  3. [Semi-Supervised Learning](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#3semi-supervised-learning)
  4. [Self-Supervised Learning](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#4self-supervised-learning)
  5. [Reinforcement Learning](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#5reinforcement-learning-rl)

----------------------------------------
# Description 

**Machine Learning (ML)** is the computer programming field where the machine is led to learn from data. In contrast with traditional programming approaches where the developer had to continuously improve the code, machine learning aims to keep up with this ever-changing world by self-adapting and learning on the fly.  

It is been around for decades and nowadays it is present in so many shapes that became unnoticeable and yet indispensable for our daily life. From call center robots to a simple Google search, as well as Amazon or Netflix recommendations, they all have a machine learning algorithm working behind it. 

Every such application uses a specific learning system and we can categorize these systems by different criteria. We call them _supervised, unsupervised, semi-supervised, self-supervised_ or _reinforcement learning_ by the level of human supervision, _online_ or _batch learning_ depending on weather they are pretrained or learn on-the-spot and _instance-based_ or _model-based learning_ if it compares receiving data to known data points, or if otherwise detects patterns in the training data and builds a predictive model.

----------------------------------------
# Machine Learning Categories
## 1.	Supervised Learning

Supervised learning is the most common method because of its advantage of using known target to correct itself. 

Inspired by how students are supervised by their teacher who provides them the right answer to a problem, similarly this technique uses pre-matching input-output pairs. In this way, explicit examples make the learning process easy and straightforward. 

**When do we use Supervised Learning?** 

Whenever the problem lies in one of the two subcategories: **regression** or **classification**. 

**Regression** is the task of estimating or predicting continuous data (unstable values), such as: popularity ([paper](https://arxiv.org/pdf/1907.01985.pdf) & [code](https://github.com/dingkeyan93/Intrinsic-Image-Popularity))/ population growth/ weather ([article](https://stackabuse.com/using-machine-learning-to-predict-the-weather-part-1/) & [code](https://github.com/MichaelE919/machine-learning-predict-weather))/ stock prices ([code & details](https://github.com/dduemig/Stanford-Project-Predicting-stock-prices-using-a-LSTM-Network/blob/master/Final%20Project.ipynb)), etc. using algorithms like linear regression (because it outputs a probabilistic value, ex.: 40% chance of rain), non-linear regression or [Bayesian linear regression](https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7).

- If the model is unable to provide accurate results, backward propagation (detailed in the next chapter) is used to repeat the whole function until it receives satisfactory results.

**Classification** is the task of estimating or predicting discrete values (static values), processes which assigns meaning to items (tags, annotations, topics, etc.), having applications such as: image classification, spam detection, etc. using algorithms like _linear regression, Decision Tree_ and _Random Forests_. 

## 2.	Unsupervised Learning

Unsupervised learning, on the other hand, is dealing with unlabeled datasets, being forced to find patterns on its own by extracting useful features from provided data and analyzing its structure.

**When do we use Unsupervised Learning?**

Unsupervised learning is applied when the dataset doesn’t come with labels, as well as when the labels are available, but you seek for more interesting hidden patterns in your data. 

This learning method is being used for tasks such as: _clustering, data visualization, dimentionality reduction_ and _anomaly detection._

**Clustering**: is the task for identifying similarity among items in order to group them – without having a name for that group (a label).
   Popular algorithms for this task: **K-Mean, KNN, DBSCAN, Hierarchical Cluster Analysis (HCA)**

**Visualization**: is the task for identifying and providing qualitative understanding of your dataset, like: trends, outliers, and patterns.
   Popular algorithms for this task: **Principal Component Analysis (PCA), Kernel PCA, Locally Linear Embedding, t-Distributed Stochastic Neighbor Embedding**; They conserve as much structure as they can by keeping separate classes in the input to prevent overlapping in the visualization, so that you can identify unusual patterns, if present. 
 
**Dimensionality reduction** (essential in meaningful compression and structure discovery) has the goal to simplify the input data without losing too much information. A solution is to merge several similar features into one. For example, a movie’s director may be strongly correlated with its actors, so the dimensionality reduction algorithm will merge them into one feature that represents the movie staff. This is called _feature extraction_. 

   - a dimensionality reduction algorithm used before a learning method will allow a much faster running, occupy less memory space and, sometimes, might perform better. 

**Anomaly Detection** has the goal to detect any unusual activity or presence in your data. Such algorithms detect credit card frauds, sustain the system health monitoring, etc. Even if you don’t have such a complex application, you can still run an anomaly detection algorithm to make sure the training set is not misleading. 

   - An anomaly detection algorithm used before a learning method will eliminate possible outliers, improving the dataset quality.

Some specific areas include recommender systems, targeted marketing and customer segmentation, big data visualization, etc.

## 3.	Semi-supervised Learning

Semi-supervised learning is the method used when the training dataset has both labeled and unlabeled data.

An example from everyday life where we meet this kind of machine learning is in the photo-storage cloud-based services. You might have noticed that once you upload your photos to a cloud service, it automatically makes a distinction between different people in your pictures. Furthermore, it asks you to add a tag for each person (which will represent the labeled data) so that it can learn to name them in other untagged pictures uploaded (which is the unlabeled data).

**When do we use Semi-Supervised Learning?**

Semi-Supervised learning is used when have both labeled and unlabeled data.

Some Semi-Supervised Algorithms include: **self-training, generative methods, mixture models, graph-based methods, co-training, semi-supervised SVM** and many others. 

## 4.	Self-supervised Learning

Self-supervised learning is the method with a level of supervision similar with the fully supervised method, but the labels here are automatically generated, typically by a heuristic algorithm, from the input data. After the label extraction, the following steps are similar as in the supervised learning algorithm.

Supervised learning is a safe bet, but it is limited. Unsupervised and Self-supervised learning are more flexible options and bring considerable value to the data.

**When do we use Self-Supervised Learning?**

Self-Supervised Learning is mostly use for motion-object detection as in this [paper](https://arxiv.org/pdf/1905.11137.pdf) & [code](https://people.eecs.berkeley.edu/~pathak/unsupervised_video/). Here is [a list of other papers](https://github.com/jason718/awesome-self-supervised-learning) using self-supervised learning.


## 5.	Reinforcement Learning (RL) 

Reinforcement learning has no kind of human supervision. It is a completely different approach, where the machine – called the agent – learns by observing the environment: it selects actions from a list of possibilities and acts accordingly, getting a reward if the result is good or a penalty if the result is bad. 
 
**When do we use Reinforcement Learning?**

Reinforcement learning is used in Games ([DeepMind’s AlphaGo](https://deepmind.com/research/case-studies/alphago-the-story-so-far) – from Google: [paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ), the official code hasn’t been released, but here’s an [alternative](https://github.com/tensorflow/minigo)), Real-Time Decisions (Traffic Light Control – [paper](http://web.eecs.utk.edu/~ielhanan/Papers/IET_ITS_2010.pdf), [paper](https://arxiv.org/pdf/1903.04527.pdf) & [code](https://github.com/cts198859/deeprl_network/blob/master/README.md)), Robot Navigation ([MuJoCo](http://www.mujoco.org/book/index.html) – physics simulator), Skill Acquisition (Self-Driving Car – [paper](https://arxiv.org/pdf/1801.02805.pdf) & [code](https://github.com/lexfridman/deeptraffic)). 
