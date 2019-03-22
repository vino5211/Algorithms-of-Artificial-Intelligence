# BERT:Pre-training of Deep Bidirectional Transformers for Language Understanding

## Learn step
+ Tansformer
+ 

## Reference


## Abstract
+ 与ELMo 比较
+ Unlike recent language representation models (Peters et al., 2018; Radford
et al., 2018), BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.
+ 通过所有层的上下文来预训练深度双向的表示
+ 预训练的BERT能够仅仅用一层output layer进行fine-turn, 就可以在许多任务上取得SOTA(start of the art) 的结果, 并不需要针对特殊任务进行特殊的调整


## Introduction
+ 使用语言模型进行预训练可以提高许多NLP任务的性能
	+ Dai and Le, 2015
	+ Peters et al.2017, 2018
	+ Radford et al., 2018
	+ Howard and Ruder, 2018

+ 提升的任务有
	+ sentence-level tasks(predict the relationships between sentences)
		+ natural language inference
			+ Bowman et al., 2015
			+ Williams et al., 2018
		+ paraphrasing(释义)
			+ Dolan and Brockett, 2005
	+ token-level tasks(models are required to produce fine-grained output at token-level)
		+ NER
			+ Tjong Kim Sang and De Meulder, 2003
		+ SQuAD question answering

### 预训练language representation 的两种策略
+ feature based
  + ELMo(Peters et al., 2018) [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
    + use **task-specific** architecture that include pre-trained representations as additional features representation
    + use shallow concatenation of independently trained left-to-right and right-to-left LMs
+ fine tuning
  + Generative Pre-trained Transformer(OpenAI GPT) [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)
  	+ introduces minimal task-specific parameters, and is trained on the
  downstream tasks by simply fine-tuning the pre-trained parameters
  	+ left-to-right

### 语言模型的缺陷(无向, 限制了在预训练中的使用)
+ The major limitation is that standard language models are **unidirectional**, and this limits the choice of architectures that can be used during pre-training
+ 例子, 在 OpenAI GPT 中 使用了 left-to-right architecture, 每个token只能注意到Transformer中self-attention layer 的 previous token
+ Such restrictions are sub-optimal for sentence-level tasks, and could be devastating when applying fine-tuning based approaches to token-level tasks such as SQuAD question answering (Rajpurkar et al., 2016), where it is crucial(关键) to incorporate(合并) context from both directions

### Outline of BERT
+ improve the fine tuning based approaches by proposing **BERT**:Bidirectional Encoder Representations from Trnaofrmers
+ address unidirectional constraints by proposing a new pre-training objective
	+ the 'masked language model' : MLM
	+ inspried by the Cloze task
+ MLM **randomly** mask some of tokens from the input(随机遮盖一些tokens)
+ and the objective is to predict the original vocabulary id of the masked word based only on its context(根据上下文预测被遮盖的tokens)
+ 与使用left-to-right LM 预训练不同的是，　MLM 使用全部上下文, 这可以允许我们预训练一个深度双向的Transformer
+ 额外的, 提出Mask Sentence Model进行Next Sentence prediction
	+ 通过使用预训练的 text-pair representations

### Contributions of this paper
+ 解释了双向预训练对Language Representation的重要性
	+ 使用 MLM 预训练 深度双向表示
	+ 与ELMo区别
+ 消除(eliminate)了 繁重的task-specific architecture 的工程量
	+ BERT is the first fine-tuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many systems with task-specific architectures
	+ extensive ablations
		+ goo.gl/language/bert

## Related Work
+ review the most popular approaches of pre-training general language represenattions
+ Feature-based Appraoches
	+ non-neural
	+ neural
	+ coarser granularities
		+ sentence embedding
		+ paragrqph embedding
		+ As with traditional word embeddings,these learned representations are also typically used as features in a downstream model.
	+ ELMo
		+ 从LM中提取上下文敏感的特征
		+ ELMo advances the state-of-the-art for several major NLP bench-
marks (Peters et al., 2018) including question 
			+ answering (Rajpurkar et al., 2016) on SQuAD
			+ sentiment analysis (Socher et al., 2013)
			+ and named entity recognition (Tjong Kim Sang and De Meul-
der, 2003).
+ Fine tuning Approaches
	+ 在LM进行迁移学习有个趋势是预训练一些关于LM objective 的 model architecture, 在进行有监督的fine-tuning 之前
	+ The advantage of these approaches is that few parameters need to be learned
from scratch
	+ OpenAI GPT (Radford et al., 2018) achieved previously state-of-the-art results on many sentencelevel tasks from the GLUE benchmark (Wang et al., 2018).

+ Transfer Learning from Supervised Data 
	+ 无监督训练的好处是可以使用无限制的数据
	+ 有一些工作显示了transfer 对监督学习的改进
		+ natural language inference (Conneau et al., 2017)
		+ machine translation (McCann et al., 2017)
	+ 在CV领域, transfer learning 对 预训练同样发挥了巨大作用
		+ Deng et al.,2009; Yosinski et al., 2014

## BERT
+ BERT v.s. OpenAI GPT v.s. ELMo
+ ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/DR1.png)

##### Model Architecture
+ BERT’s model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al.(2017) and released in the tensor2tensor library
+ Transformer
