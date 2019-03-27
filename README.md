[TOC]

# Algorithm of Artificial Intelligence

### Target

+ View state of art artificial intelligence work，reach the boundaries of current research
- Organize the basic ideas and methods in the filed of AI, including Math, Machine Learning, Deep Learinng, Natural Language Processing, Graph Model, Bayesian Theory
- Plan to carding some knowledge of Genetic Algorithm, Transfer Learning, Reinforce Learning, Manifold Learning, Semi-supervised Learning
- Looking for a better information representation
- How to build Logic ?
  - 导入先验知识
  - 大规模检索发现（RL）
- How to automate the search for information training network
- How to represent Information? Matrix or Graph, even more complex Data Structure?
- Probability Graph Model and Knowledge
- 从生物进化的角度思考ANN的进化
- 整体端到端无监督，多任务学习有监督和半监督
    - 无监督
        - BERT OPEN-GPT
    - 多任务
        - Mulit-Task DNN
### Main work
- 公式推导
- 代码复现
- 思路整理
- 各个子文件夹README.md 和 jupyter 编写
- 第一层文件夹下的Readme.md 是 Outline
- 其余文件夹下的Readme.md 是View

### Doing
+ Attention
+ Memmory
+ Pointer Network
+ Capsule
+ GAN
  + 文本生成
+ BERT/Transformer
+ Reinforce Learning
+ Transfor Leaning

### Problems
- DL/cnn ：（1）CNN 缺失较多需要添加 （2）TextCNN and Variants
- DL/NER ： （1）特征评价指标 （2）相关比赛/数据集整理 （3）代码整理
- Tuning ： 当前整理不够深刻, 还需要梳理 一些自动调参的方法需要了解一
- MAP/MLE 细化
- 变分
- 蒙特卡洛
- EM/MLE

### Abbreviations
- To better distinguish file types, the abbreviations in filenames have the following meanings:
    - ML : Machine Learning
    - DL : Deep Learning
    - NLP : Natural language processing

### Corpus Preprocessing
- Data Augmentation
- Data Clean
- Data Smooth
- Data Translate
- Data Visualization
- README.md (Done, Add Link)

### NLP-Tools

### Neural Evoluation

### Traditional Machine Learning
- Basic
    - Perceptron
    - Activation
    - Loss
    - Back Propagation
    - Tuning Parameter
    - Tuning HyperParameter
- Feature Engineering
- Linear Model
  - Linear Regression
  - Logistic Regression
  - Linear Discriminant
  - Linear SVM
- SVM
  - Hinge Loss
  - Kernal 
  - SVR
- Diemnsion Reduction
  - Sparse Representation
  - PCA
  - Manifold Learning
  	- LLE
  	- t-SNE
  - Laplacian Eigenmaps(Unfinished)
  - Auto Encoder 
- Topic Model
  - LSA
  - pLSA
  - LDA(Latent Dirichlet Allocation) 
- Regression
- EM
- MLE

### Ensemble Learning
- Bagging
- Boosting
- Blending
- Stacking
- Decision Tree
- Random Forest
- Adaboost
- Gradient Boosting
- GBDT
- XGboost
- LightGBM
- ThunderGBM

### Deep Learning
- Perceptron and Back propagation
- Tuning method
  - Regularization method
  - Optimizer method
  - Initialization strategy
  - Data standardization
- RNNS
- CNNs
- End2End
- HighWay
- ResNet
- DenseNet
- Capsule
- GANs
- VAEs

### Semi-supervised

### Un-superivised
- Clustering
- Performance metric fo Clustering
- Distance calculation
- Prototype clustering
    - Kmeans
- Learning vector quantization(学习向量量化)
- Gaussian hybrid clustering
- Density clustering
- Hierarchical clustering 

### Sequence Graph Network Model
#### Bayesian Theory
+ Bayesian Decision Theory
+ Naive Bayes
+ Semi Naive Bayes
+ Bayesian Net
#### Sequnece & Graph Model
+ Directed graphical model (Also known as Bayesian Network)
    + Static Bayesian networks
    + Dynamic Bayesian networks
        + Hidden Markov Model
        + Kalman filter
+ Undirection graphical models
    + Markov networks(Also known as Markov random field)
        + Gibbs Boltzman machine
        + Conditional random field
- Structure SVM((Unfinished))
- Viterbi
#### Network
- DeepWalk
- RandomWalk
- LINE
- SDNE

#### Knowledge Base
- DataSet

### Reinforce Learing

### Transfor Learning

### Nature Language Processing
#### Target
- Record SOTA algorithms
- Record dataset, projects, papers in README.md
#### outline
- Repersent Learning
    - Sparse Representation
    - Similarity
    - Embedding
    - Language Model
    - Sementic Analysic
- Text Classification
- Sentiment Analysis
- Text Clustering
- Lexical Analysis
- Syntax Analysis
- Chapter Analysis
- Text Summary
- Text Generation
- Sequence Labeling
- Information Retrieval
- Machine-Reading-Comprehension
- Dialog-System
    - Pipeline
        - NLU
        - DM
        - NLG
   - End2End
#### Projects Review
+ BERT
+ Open-GPT 2
+ MT-DNN
