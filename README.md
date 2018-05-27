## Knowledge

- There are two main functions of this repository:
	- organize the required knowledge(math, ml, dl, rl, nlp, cv theory) in projects
	- build a knowledge base of state of art artifical intelligence work，reach the boundaries of current research

- Target
	- make cool things

- To better distinguish file types, the abbreviations in filenames have the following meanings:
    - CU : Self Clear up（default）
    - Math : Math
    - ML : Machine Learning
    - NN : Deep Learning
    - NLP : Natural language processing
    - OR : Operation Record

- This repository contains following files:
	- AlgorithmTree
		- The curent summary of the algorithm encountered
		- update new algorithm to this file
		- maintain levle : A
		- strength : medium
	- Genetic Algorithm
		- Chinese Name : 遗传算法
		- strength : weak
		- problem : 算法原理和应用场景不清楚，感觉有一定的可解释性
	- (U)Math
		- Deriavtive
		- Least Suqares
		- Linear Algebra
		- Matrix factorization
			- SVD
		- Parameter Estimation
		- Probability

	- Machine Learning
		- tips : Implement following method in Holy Miner
		- Basic
			- MLE,MAE
			- Evaluate
				- accuracy
				- P,R,F1
				- ROC,AUC
			- Generating Model
				- HMM、高斯混合模型GMM、LDA、PLSA、Naive Bayes
			- Disccriminating Model
				- SVM,NN,LR,CRF
		- Bayes（Plan）
            - ZZH C7
            - 
        - Cluster
			- (U)性能度量
			- 距离计算
			- 原型聚类
				- Kmeans
				- (U)学习向量量化
				- (U)高斯混合聚类
			- 密度聚类
			- 层次聚类
				- (U)HAC
		- Dimension Reduction(有点乱，需要整理)
			- PCA
				- code
			- Manifold Learning
				- LLE
					- (U) code
				- t-SNE
					- code
				- (U) Laplacian Eigenmaps
			- Auto Encoder
		- Ensemble
			- Bagging
			- Boosting
			- RF
			- Adaboost
			- GBDT
			- XGboost
			- LightGBM
		- KNN
			- weak
		- LDA(线性判别分析)
		- LR
			- Undo
		- Regression
			- Undo
		- SVM
			- Hinge loss
			- (U)Kernel
		- Topic Model
			- LSA
			- pLSA
			- LDA(Latent Dirichlet Allocation)
	- (U) NN
		- (U) Deep Generation Model
			- 参照 book Deep Learning
			- AE
			- VAE
		- Capsule
		- CNN
		- Optimizer
		- RNN
		- S2S-AT
		- ResNet
		- Tune
	- NLP
		- (U) Basic
			- Co-occurrence matrix
			- TF-IDF
			- stem(词干提取)
			- lemma(词型还原)
		- CLF（ a cool name）
			- implemented by (a cool name)
		- Embedding
			- (U)fine tune
			- word2vec
				- gensim train
			- glove
				- download en
			- fasttext
				- download zh
				- (U) Principle of algorithm
		- Knowledge Extract
		- Language Model
		- Recommended System
		- RC(QA)
		- Semantic Analysis
		- Sentiment Analysis
		- Sequence Label
			- Seg
			- Pos
			- NER
		- Syntax Analysis

	- PGM
		- HMM
			- forward and backward probability compute
			- EM
			- viterbi
		- CRF
			- Feature Vector $\phi$
			- CRF++
		- (U)Structure SVM
		- example of HMM v.s. CRF
		- example of CRF v.s. DL+CRF
		- (???)why CRF is better than CRF
		- LHY 书 CRF 部分学习
	- (U)RL

---

## format
```
[TOC]
# Name
## Reference
## A
##### B
...
## A

---
## problem
博客图片  先存放与 Pictures 中，部分图片不清晰
图片大小放置不合适，不美观