# What is Machine Learning (ML)?

Content:

[Description](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#description)

[Machine Learning Categories by the level of human supervision:](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#machine-learning-categories)
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
# Machine Learning Categories by the level of human supervision
## 1.	Supervised Learning

Supervised learning is the most common method because of its advantage of using known target to correct itself. 

Inspired by how students are supervised by their teacher who provides them the right answer to a problem, similarly this technique uses pre-matching input-output pairs. In this way, explicit examples make the learning process easy and straightforward. 

**When do we use Supervised Learning?** 

Whenever the problem lies in one of the two subcategories: **regression** or **classification**. 

**Regression** is the task of estimating or predicting continuous data (unstable values), such as: popularity ([paper](https://arxiv.org/pdf/1907.01985.pdf) & [code](https://github.com/dingkeyan93/Intrinsic-Image-Popularity))/ population growth/ weather ([article](https://stackabuse.com/using-machine-learning-to-predict-the-weather-part-1/) & [code](https://github.com/MichaelE919/machine-learning-predict-weather))/ stock prices ([code & details](https://github.com/dduemig/Stanford-Project-Predicting-stock-prices-using-a-LSTM-Network/blob/master/Final%20Project.ipynb)), etc. using algorithms like linear regression (because it outputs a probabilistic value, ex.: 40% chance of rain), non-linear regression or [Bayesian linear regression](https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7).

- If the model is unable to provide accurate results, backward propagation (detailed in the next chapter) is used to repeat the whole function until it receives satisfactory results.

**Classification** is the task of estimating or predicting discrete values (static values), processes which assigns meaning to items (tags, annotations, topics, etc.), having applications such as: image classification, spam detection, etc. using algorithms like _linear regression, Decision Tree_ and _Random Forests_. 

### Supervised Learning in different areas:

### Text (Natural Language Processing - NLP)

- **Language Modelling**:
- **Machine Translation**: 
- **Text Classification**:
- **Natural Language Inference**:
- **Question Answering**:
- **Named entity recognition**:
- **Abstractive Summarization**:
- **Dependency Parsing**: 

### Image (Computer Vision)

- **Semantic Segmentation**: ResNeSt: Split-Attention Networks - [paper](https://arxiv.org/pdf/2004.08955v1.pdf) & code [Tensorflow](https://github.com/dmlc/gluon-cv)/[PyTorch](https://github.com/zhanghang1989/ResNeSt).
- **Image Classification**: 
- **Visual Question Answering**:
- **Person Re-identification**:

### Audio (Automatic Speech Recognition - ASR)

- **Speech to Text/ Text to Speech**: Speech to text and text to speech recognition systems-Areview - [paper](https://www.iosrjournals.org/iosr-jce/papers/Vol20-issue2/Version-1/E2002013643.pdf).

- **Speech recognition**: Deep Speech 2: End-to-End Speech Recognition in English and Mandarin - [paper](https://arxiv.org/pdf/1512.02595v1.pdf) & [code](https://github.com/tensorflow/models/tree/master/research/deep_speech), Real-Time Voice Cloning - [code](https://github.com/CorentinJ/Real-Time-Voice-Cloning).

- **Speech Synthesis**: Natural TTS Synthesis by conditioning wavenet on MEL spectogram predictions - [paper](https://arxiv.org/pdf/1712.05884v2.pdf) & [code](https://github.com/NVIDIA/tacotron2) & [explained](https://github.com/codetendolkar/tacotron-2-explained)(using Tacotron 2 method); Other method: [WaveNet](https://github.com/r9y9/wavenet_vocoder) - [paper](https://arxiv.org/pdf/1609.03499v2.pdf) & [code](https://github.com/maciejkula/spotlight).

- **Speeche Enhancement**: Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression - [paper](https://arxiv.org/pdf/2005.07551.pdf) & [code](https://github.com/breizhn/DTLN).

- **Speaker Verification**: Text Independant Speaker Verification - [code](https://github.com/Suhee05/Text-Independent-Speaker-Verification).

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

### Unsupervised Learning in different areas:

### Text (Natural Language Processing - NLP)

- **Language Modelling**: Improving Language Understanding by Generative Pre-Training - [paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) & [code](https://github.com/openai/finetune-transformer-lm).
- **Machine Translation**: Unsupervised Neural Machine Translation with Weight Sharing - [paper](https://arxiv.org/pdf/1804.09057.pdf) & [code](https://github.com/facebookresearch/UnsupervisedMT).
- **Text Classification**: Unsupervised Text Classification for Natural Language Interactive Narratives - [paper](https://people.ict.usc.edu/~gordon/publications/INT17A.PDF) & code not provided.
- **Question Answering**: Language Models are Unsupervised Multitask Learner - [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) & [code](https://github.com/hanchuanchuan/gpt-2).
- **Abstractive Summarization**: Centroid-based Text Summarization through Compositionality of Word Embeddings - [paper](https://www.aclweb.org/anthology/W17-1003.pdf) & [code](https://github.com/gaetangate/text-summarizer).

### Image (Computer Vision)

- **Semantic Segmentation**:  - [paper]() & code .
- **Image Classification**:  - [paper]() & [code]().
- **Object Recognition**:  - [paper]() & [code]().
- **Person Re-identification**:  - [paper]() & [code]().

### Audio (Automatic Speech Recognition - ASR)

- **Speech to Text/ Text to Speech**:  - [paper]().
- **Speech recognition**:  - [paper]() & [code]().
- **Speech Synthesis**:  - [paper]() & [code]().
- **Speeche Enhancement**:  - [paper]() & [code]().
- **Speaker Verification**:  - [code]().

## 3.	Semi-supervised Learning

Semi-supervised learning is the method used when the training dataset has both labeled and unlabeled data.

An example from everyday life where we meet this kind of machine learning is in the photo-storage cloud-based services. You might have noticed that once you upload your photos to a cloud service, it automatically makes a distinction between different people in your pictures. Furthermore, it asks you to add a tag for each person (which will represent the labeled data) so that it can learn to name them in other untagged pictures uploaded (which is the unlabeled data).

**When do we use Semi-Supervised Learning?**

Semi-Supervised learning is used when have both labeled and unlabeled data.

Some Semi-Supervised Algorithms include: **self-training, generative methods, mixture models, graph-based methods, co-training, semi-supervised SVM** and many others. 

### Semi-supervised Learning in different areas:

### Text (Natural Language Processing - NLP)

- **Language Modelling**:  - [paper]() & [code]().
- **Machine Translation**:  - [paper]() & [code]().
- **Text Classification**:  - [paper]() & [code]().
- **Question Answering**:  - [paper]() & [code]().
- **Abstractive Summarization**:  - [paper]() & [code]().

### Image (Computer Vision)

- **Semantic Segmentation**:  - [paper]() & code .
- **Image Classification**: Fixing the train-test resolution discrepancy - [paper](https://arxiv.org/pdf/2003.08237v4.pdf) & [code](https://github.com/facebookresearch/FixRes).
- **Object Recognition**:  - [paper]() & [code]().
- **Person Re-identification**:  - [paper]() & [code]().

### Audio (Automatic Speech Recognition - ASR)

- **Speech to Text/ Text to Speech**:  - [paper]().
- **Speech recognition**:  - [paper]() & [code]().
- **Speech Synthesis**:  - [paper]() & [code]().
- **Speeche Enhancement**:  - [paper]() & [code]().
- **Speaker Verification**:  - [code]().

## 4.	Self-supervised Learning

Self-supervised learning is the method with a level of supervision similar with the fully supervised method, but the labels here are automatically generated, typically by a heuristic algorithm, from the input data. After the label extraction, the following steps are similar as in the supervised learning algorithm.

Supervised learning is a safe bet, but it is limited. Unsupervised and Self-supervised learning are more flexible options and bring considerable value to the data.

**When do we use Self-Supervised Learning?**

Self-Supervised Learning is mostly use for motion-object detection as in this [paper](https://arxiv.org/pdf/1905.11137.pdf) & [code](https://people.eecs.berkeley.edu/~pathak/unsupervised_video/). Here is [a list of other papers](https://github.com/jason718/awesome-self-supervised-learning) using self-supervised learning.

### Self-supervised Learning in different areas:

### Text (Natural Language Processing - NLP)

- **Language Modelling**:  - [paper]() & [code]().
- **Machine Translation**:  - [paper]() & [code]().
- **Text Classification**:  - [paper]() & [code]().
- **Question Answering**:  - [paper]() & [code]().
- **Abstractive Summarization**:  - [paper]() & [code]().

### Image (Computer Vision)

- **Semantic Segmentation**:  - [paper]() & code .
- **Image Classification**:  - [paper]() & [code]().
- **Object Recognition**:  - [paper]() & [code]().
- **Person Re-identification**:  - [paper]() & [code]().

### Audio (Automatic Speech Recognition - ASR)

- **Speech to Text/ Text to Speech**:  - [paper]().
- **Speech recognition**:  - [paper]() & [code]().
- **Speech Synthesis**:  - [paper]() & [code]().
- **Speeche Enhancement**:  - [paper]() & [code]().
- **Speaker Verification**:  - [code]().

## 5.	Reinforcement Learning (RL) 

Reinforcement learning has no kind of human supervision. It is a completely different approach, where the machine – called the agent – learns by observing the environment: it selects actions from a list of possibilities and acts accordingly, getting a reward if the result is good or a penalty if the result is bad. 
 
**When do we use Reinforcement Learning?**

Reinforcement learning is used in Games ([DeepMind’s AlphaGo](https://deepmind.com/research/case-studies/alphago-the-story-so-far) – from Google: [paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ), the official code hasn’t been released, but here’s an [alternative](https://github.com/tensorflow/minigo)), Real-Time Decisions (Traffic Light Control – [paper](http://web.eecs.utk.edu/~ielhanan/Papers/IET_ITS_2010.pdf), [paper](https://arxiv.org/pdf/1903.04527.pdf) & [code](https://github.com/cts198859/deeprl_network/blob/master/README.md)), Robot Navigation ([MuJoCo](http://www.mujoco.org/book/index.html) – physics simulator), Skill Acquisition (Self-Driving Car – [paper](https://arxiv.org/pdf/1801.02805.pdf) & [code](https://github.com/lexfridman/deeptraffic)). 
