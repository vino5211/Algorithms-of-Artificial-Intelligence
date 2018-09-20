# Part I : Network Embedding: Recent Progress and Applications
+ Traditional Network Representation
+ Concepts
	+ Representation learning
	+ Distributed representation
	+ Embedding
+ Network Embedding
	+ Map the nodes in a network **into** a low dimensional space
		+ Distributed representation for nodes
		+ Similarity between nodes indicate the link strength(节点间的相似性表示链路强度)
		+ Encode network information and generate node representation
+ Problems with previous Models
	+ Classical graph embedding algorithms
		+ MDS,IsoMap,LLE,Laplacian Eigenmap(???)
		+ Most of them follow a matrix factorization(矩阵分解) or computation approach(???)
		+ Hard to scale up(难以扩展)
		+ Diffcult to extend to new settings
+ Outline
	+ Preliminaries(初步措施)
		+ word2vec
	+ Basic Network Embedding Models
		+ DeepWalk,Node2Vec,LINE,GrapRep,SDNE
	+ Advanced Network Embedding Models
		+ Beyond embedding,vertex information,edge information(超越嵌入, 顶点信息, 边缘信息)
	+ Applications of Network Embedding
		+ Basic applications
		+ visualization
		+ text classification
		+ recommendation
+ Preliminaries
	+ Softmax functions
		+ sigmoid function
			$$ \phi(x) = \frac{1}{1 + e^{-x}}$$
		+ 
	+ Distributional semantics
	+ Word2Vec
		+ CROW
		+ Skip-gram