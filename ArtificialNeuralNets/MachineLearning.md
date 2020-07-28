# What is Machine Learning (ML)?

<div align="right">
<p> :calendar: Summer, 2020
:bust_in_silhouette: Author <a href="https://github.com/laviniaflorentina"> Lavinia Florentina </a> </p>
</div>

Content:

- [Introduction](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#introduction)
- [Machine Learning Categories by the level of human supervision:](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#machine-learning-categories)
  - [Supervised Learning](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#1supervised-learning)
  - [Unsupervised Learning](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#2unsupervised-learning)
  - [Semi-Supervised Learning](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#3semi-supervised-learning)
  - [Self-Supervised Learning](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#4self-supervised-learning)
  - [Reinforcement Learning](https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/MachineLearning.md#5reinforcement-learning-rl)

----------------------------------------
# Introduction 

**Machine Learning (ML)** is the AI branch dealing with how a machine is led to learn from data. In contrast with traditional programming approaches where the developer had to continuously improve the code, machine learning aims to keep up with this ever-changing world by self-adapting and learning on the fly.  

It is been around for decades and nowadays it is present in so many shapes that became unnoticeable and yet indispensable for our daily life. From call center robots to a simple Google search, as well as Amazon or Netflix recommendations, they all have a machine learning algorithm working behind it. 

Every such application uses a specific learning system and we can categorize these systems by different criteria. 

<!---
<img align="right" src="https://cdn2.hubspot.net/hubfs/202339/machine%20learning.png" width="600" height="340">
-->

Depending on the level of human supervision, we call them:
  - supervised 
  - unsupervised 
  - semi-supervised 
  - self-supervised 
  - reinforcement learning. 

Depending on weather they are pretrained or learn on-the-spot, we have:  
  - online 
  - batch learning.  
  
And if it compares receiving data to known data points, or if otherwise detects patterns in the training data and builds a predictive model, we call them:  
  - instance-based
  - model-based learning.

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

### :abc: Text (Natural Language Processing - NLP)

- **Machine Translation**: Syntactically Supervised Transformers for Faster Neural Machine Translation - [paper](https://arxiv.org/pdf/1906.02780v1.pdf) & [code](https://github.com/dojoteef/synst).
- **Named entity recognition**: Distantly Supervised Named Entity Recognition using Positive-Unlabeled Learning - [paper](https://arxiv.org/pdf/1906.01378v2.pdf) & [code](https://github.com/v-mipeng/LexiconNER).
- **Text Summarization**: Iterative Document Representation Learning Towards Summarization with Polishing - [paper](https://arxiv.org/pdf/1809.10324v2.pdf) & [code](https://github.com/yingtaomj/Iterative-Document-Representation-Learning-Towards-Summarization-with-Polishing).

### :cinema: Image (Computer Vision)

- **Semantic Segmentation**: ResNeSt: Split-Attention Networks - [paper](https://arxiv.org/pdf/2004.08955v1.pdf) & code [Tensorflow](https://github.com/dmlc/gluon-cv)/[PyTorch](https://github.com/zhanghang1989/ResNeSt); Invisibility Cloak -[Tutorial](https://lnkd.in/gSpeKx7) & [code](https://lnkd.in/gYq_nku), [another Code](https://lnkd.in/ggSmHwB).
- **Image Classification**: Dynamic Routing Between Capsules - [paper](https://arxiv.org/pdf/1710.09829.pdf) & [code](https://github.com/Sarasra/models/tree/master/research/capsules).
- **Visual Question Answering**: Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning - [paper](https://arxiv.org/pdf/1703.06585v2.pdf) & [code](https://github.com/batra-mlp-lab/visdial-rl).
- **Person Re-identification**: Weakly supervised discriminative feature learning with state information for person identification - [paper](https://arxiv.org/pdf/2002.11939v1.pdf) & [code](https://github.com/KovenYu/state-information).

### :sound: Audio (Automatic Speech Recognition - ASR)

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

### :abc: Text (Natural Language Processing - NLP)

- **Language Modelling**: Improving Language Understanding by Generative Pre-Training - [paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) & [code](https://github.com/openai/finetune-transformer-lm).
- **Machine Translation**: Unsupervised Neural Machine Translation with Weight Sharing - [paper](https://arxiv.org/pdf/1804.09057.pdf) & [code](https://github.com/facebookresearch/UnsupervisedMT).
- **Text Classification**: Unsupervised Text Classification for Natural Language Interactive Narratives - [paper](https://people.ict.usc.edu/~gordon/publications/INT17A.PDF) & code not provided.
- **Question Answering**: Language Models are Unsupervised Multitask Learner - [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) & [code](https://github.com/hanchuanchuan/gpt-2).
- **Abstractive Summarization**: Centroid-based Text Summarization through Compositionality of Word Embeddings - [paper](https://www.aclweb.org/anthology/W17-1003.pdf) & [code](https://github.com/gaetangate/text-summarizer).

### :cinema: Image (Computer Vision)

- **Semantic Segmentation**: Invariant Information Clustering for Unsupervised Image Classification and Segmentation - [paper](https://arxiv.org/pdf/1807.06653.pdf) & [code](https://github.com/xu-ji/IIC).
- **Image Classification**: SCAN: Learning to Classify Images without Labels - [paper](https://arxiv.org/pdf/2005.12320v2.pdf) & [code](https://github.com/wvangansbeke/Unsupervised-Classification).
- **Object Recognition**: Unsupervised Domain Adaptation through Inter-modal Rotation for RGB-D Object Recognition - [paper](https://arxiv.org/pdf/2004.10016v1.pdf) & [code](https://github.com/MRLoghmani/relative-rotation).
- **Person Re-identification**: Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification - [paper](https://arxiv.org/pdf/1811.10144v3.pdf) & [code](https://github.com/SHI-Labs/Self-Similarity-Grouping).

### :sound: Audio (Automatic Speech Recognition - ASR)

- **Speech to Text/ Text to Speech**: Representation Learning with Contrastive Predictive Coding - [paper](https://arxiv.org/pdf/1807.03748v2.pdf) & [code](https://github.com/davidtellez/contrastive-predictive-coding).
- **Speech recognition**: A segmental framework for fully-unsupervised large-vocabulary speech recognition - [paper](https://arxiv.org/pdf/1606.06950v2.pdf) & [code](https://github.com/kamperh/recipe_bucktsong_awe).
- **Speech Synthesis**: Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis - [paper](https://arxiv.org/pdf/1803.09017v1.pdf) & [code](https://github.com/syang1993/gst-tacotron).
- **Speeche Enhancement**: Supervised and Unsupervised Speech Enhancement
Using Nonnegative Matrix Factorization - [paper](https://arxiv.org/pdf/1709.05362v1.pdf) & [code](https://github.com/mohammadiha/bnmf).
- **Speaker Verification**: An Unsupervised Autoregressive Model for Speech Representation Learning - [paper](https://arxiv.org/pdf/1904.03240v2.pdf) & [code](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding).

## 3.	Semi-supervised Learning

Semi-supervised learning is the method used when the training dataset has both labeled and unlabeled data.

An example from everyday life where we meet this kind of machine learning is in the photo-storage cloud-based services. You might have noticed that once you upload your photos to a cloud service, it automatically makes a distinction between different people in your pictures. Furthermore, it asks you to add a tag for each person (which will represent the labeled data) so that it can learn to name them in other untagged pictures uploaded (which is the unlabeled data).

**When do we use Semi-Supervised Learning?**

Semi-Supervised learning is used when have both labeled and unlabeled data.

Some Semi-Supervised Algorithms include: **self-training, generative methods, mixture models, graph-based methods, co-training, semi-supervised SVM** and many others. 

### Semi-supervised Learning in different areas:

### :abc: Text (Natural Language Processing - NLP)

- **Language Modelling**: Semi-supervised sequence tagging with bidirectional language models - [paper](https://arxiv.org/pdf/1705.00108v1.pdf) & [code](https://github.com/flairNLP/flair).
- **Machine Translation**: A Simple Baseline to Semi-Supervised Domain Adaptation for Machine Translation - [paper](https://arxiv.org/pdf/2001.08140v2.pdf) & [code](https://github.com/jind11/DAMT).
- **Text Classification**: Variational Pretraining for Semi-supervised Text Classification - [paper](https://arxiv.org/pdf/1906.02242v1.pdf) & [code](https://github.com/allenai/vampire).
- **Question Answering**: Addressing Semantic Drift in Question Generation for Semi-Supervised Question Answering - [paper](https://arxiv.org/pdf/1909.06356v1.pdf) & [code](https://github.com/ZhangShiyue/QGforQA).
- **Abstractive Summarization**: Abstractive and Extractive Text Summarization using Document Context Vector and Recurrent Neural Networks - [paper](https://arxiv.org/pdf/1807.08000.pdf) & code not provided.

### :cinema: Image (Computer Vision)

- **Semantic Segmentation**: Semi-supervised semantic segmentation needs strong, varied perturbations - [paper](https://arxiv.org/pdf/1906.01916v4.pdf) & [code](https://github.com/Britefury/cutmix-semisup-seg).
- **Image Classification**: Fixing the train-test resolution discrepancy - [paper](https://arxiv.org/pdf/2003.08237v4.pdf) & [code](https://github.com/facebookresearch/FixRes).
- **Object Recognition**: Data Distillation: Towards Omni-Supervised Learning - [paper](https://arxiv.org/pdf/1712.04440v1.pdf) & [code](https://github.com/facebookresearch/detectron) & [code](Data Distillation: Towards Omni-Supervised Learning).
- **Person Re-identification**: Sparse Label Smoothing Regularization for Person Re-Identification - [paper](https://arxiv.org/pdf/1809.04976v3.pdf) & [code](https://github.com/jpainam/SLS_ReID).

### :sound: Audio (Automatic Speech Recognition - ASR)

- **Speech to Text/ Text to Speech**: Libri-Light: A Benchmark for ASR with Limited or No Supervision - [paper]() & [code](https://github.com/facebookresearch/libri-light).
- **Speech recognition**: Semi-Supervised Speech Recognition via Local Prior Matching - [paper](https://arxiv.org/pdf/2002.10336v1.pdf) & [code](https://github.com/facebookresearch/wav2letter).
- **Speech Synthesis**: Semi-Supervised Generative Modeling for Controllable Speech Synthesis - [paper](https://arxiv.org/pdf/1910.01709v1.pdf) & code not provided.
- **Speeche Enhancement**: Semi-Supervised Multichannel Speech Enhancement With a Deep Speech Prior - [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8861142) & [code](https://github.com/sekiguchi92/SpeechEnhancement).
- **Speaker Verification**: Learning Speaker Representations with Mutual Information - [paper](https://arxiv.org/pdf/1812.00271v2.pdf) & [code](https://github.com/Js-Mim/rl_singing_voice).

## 4.	Self-supervised Learning

Self-supervised learning is the method with a level of supervision similar with the fully supervised method, but the labels here are automatically generated, typically by a heuristic algorithm, from the input data. After the label extraction, the following steps are similar as in the supervised learning algorithm.

Supervised learning is a safe bet, but it is limited. Unsupervised and Self-supervised learning are more flexible options and bring considerable value to the data.

**When do we use Self-Supervised Learning?**

Self-Supervised Learning is mostly use for motion-object detection as in this [paper](https://arxiv.org/pdf/1905.11137.pdf) & [code](https://people.eecs.berkeley.edu/~pathak/unsupervised_video/). Here is [a list of other papers](https://github.com/jason718/awesome-self-supervised-learning) using self-supervised learning.

### Self-supervised Learning in different areas:

### :abc: Text (Natural Language Processing - NLP)

- **Language Modelling**: ALBERT: A Lite BERT for Self-supervised Learning of Language Representations - [paper](https://arxiv.org/pdf/1909.11942v6.pdf) & [code](https://github.com/tensorflow/models/tree/master/official/nlp/albert).
- **Machine Translation**: Self-Supervised Neural Machine Translation - [paper](https://www.aclweb.org/anthology/P19-1178.pdf) & code not provided.
- **Text Classification**: Supervised Multimodal Bitransformers for Classifying Images and Text - [paper](https://arxiv.org/pdf/1909.02950v1.pdf) & [code](https://github.com/huggingface/transformers).
- **Abstractive Summarization**: PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization - [paper](https://arxiv.org/pdf/1912.08777v2.pdf) & [code](https://github.com/google-research/pegasus).

### :cinema: Image (Computer Vision)

- **Semantic Segmentation**: Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation - [paper](https://arxiv.org/pdf/2004.04581v1.pdf) & [code](https://github.com/YudeWang/SEAM).
- **Image Classification**: Self-Supervised Learning For Few-Shot Image Classification - [paper](https://arxiv.org/pdf/1911.06045v2.pdf) & [code](https://github.com/phecy/SSL-FEW-SHOT).
- **Action Recognition**: A Multigrid Method for Efficiently Training Video Models - [paper](https://arxiv.org/pdf/1912.00998v2.pdf) & [code](https://github.com/facebookresearch/SlowFast).
- **Person Re-identification**: Enhancing Person Re-identification in a Self-trained
Subspace - [paper](https://arxiv.org/pdf/1704.06020v2.pdf) & [code](https://github.com/Xun-Yang/ReID_slef-training_TOMM2017).

### :sound: Audio (Automatic Speech Recognition - ASR)

- **Speech recognition**: Multi-task self-supervised learning for Robust Speech Recognition - [paper](https://arxiv.org/pdf/2001.09239v2.pdf) & [code](https://github.com/santi-pdp/pase).
- **Speeche Enhancement**: More Grounded Image Captioning by Distilling Image-Text Matching Model - [paper](https://arxiv.org/pdf/2004.00390v1.pdf) & [code](https://github.com/YuanEZhou/Grounded-Image-Captioning).
- **Speaker Verification**: AutoSpeech: Neural Architecture Search for Speaker Recognition - [paper](https://arxiv.org/pdf/2005.03215v1.pdf) & [code](https://github.com/TAMU-VITA/AutoSpeech).

## 5.	Reinforcement Learning (RL) 

Reinforcement learning has no kind of human supervision. It is a completely different approach, where the machine – called the agent – learns by observing the environment: it selects actions from a list of possibilities and acts accordingly, getting a reward if the result is good or a penalty if the result is bad. 
 
**When do we use Reinforcement Learning?**

Reinforcement learning is used in Games ([DeepMind’s AlphaGo](https://deepmind.com/research/case-studies/alphago-the-story-so-far) – from Google: [paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ), the official code hasn’t been released, but here’s an [alternative](https://github.com/tensorflow/minigo)), Real-Time Decisions (Traffic Light Control – [paper](http://web.eecs.utk.edu/~ielhanan/Papers/IET_ITS_2010.pdf), [paper](https://arxiv.org/pdf/1903.04527.pdf) & [code](https://github.com/cts198859/deeprl_network/blob/master/README.md)), Robot Navigation ([MuJoCo](http://www.mujoco.org/book/index.html) – physics simulator), Skill Acquisition (Self-Driving Car – [paper](https://arxiv.org/pdf/1801.02805.pdf) & [code](https://github.com/lexfridman/deeptraffic)). 

--------------------------

<img align="centre" src="https://media.giphy.com/media/4T1Sf6UvSXYyLJ5tUS/giphy.gif" width="400" height="400">

<div align="right">
<b> NEXT:  </b> 
<a href="https://github.com/laviniaflorentina/Tutorials/blob/master/ArtificialNeuralNets/algorithms.md#algorithms-by-tasks" ><i> Algorithms by tasks</i></a> 
</div>  

