# Summary of Attention

# Attention Mechanism in CV

- Attention可以分成hard与soft两种模型: 
  - hard: Attention每次移动到一个固定大小的区域
  - soft: Attention每次是所有区域的一个加权和
- Reference:  
  - Survey on Advanced Attention-based Models
  - Recurrent Models of Visual Attention (2014.06.24)  
  - Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (2015.02.10) 
  - DRAW: A Recurrent Neural Network For Image Generation (2015.05.20)
  - Teaching Machines to Read and Comprehend (2015.06.04) 
  - Learning Wake-Sleep Recurrent Attention Models (2015.09.22)
  - Action Recognition using Visual Attention (2015.10.12)
  - Recursive Recurrent Nets with Attention Modeling for OCR in the Wild (2016.03.09)

# Condition Generation

+ Encoder-Decoder
+ Building End-to-End Dialogue System using Generative Hierachical Neural Network

# Encoder-Decoder

## Tips

+ What is match?
	+ Cosine similarity of z and h
	+ Small NN whose input is z and h, output a scalar
	+ $$\alpha = h^TWz$$

+ Attention base model
	+ Machine Translate
	+ Speech Recognition
		+ Listen, Attend, and Spell
	+ Image Caption Generation
		+ vector for each region 
		+ Show, Attend and Tell:Neural Image Caption Generation with Visual Attention, ICML 2015
	+ Video  Caption Generation
		+ Describing Videos by Exploiting Temporal Structure

## Framework

+ picture

# Memory Network

## Reference

- End-to-End Memory Networks

## Simple Framework

- query 和某个document $x_i$计算 Match, 得到$a_i$
- 由 $x_i$ 和 $a_i$ 的加权求和 得到 抽取出的信息, 并介入DNN网络, 得到最终Answer

![](/home/apollo/Pictures/Mem2.png)

## Complex Framework

- 计算Match 的时候使用 $x_i$
- 加权求和的时候使用$h_i$
- 使用两种不同的文本表示, 最终得到的结果较好
	![](/home/apollo/Pictures/Mem1.png)

+ Hopping
	+ 将Extract Information 作为 Query 的 vector 再次输入
	+ 可反复输入, 根据经验可以提高精度(具体原理不清楚)(反复思考,可以得到更精确的结果)
		+ picture
	+ Tree LSTM + Attention
		+ picture

# Neural Turing Machine

## Tips

+ NTM not only read form momory
+ Also modify the memory through attention

## Framework

+ 



# Tips for Generation

+ Good Attention : each input component has approximately the same attention weight
	+ attention weight normalization
+ Mismatch between Train and Test
	+ Exposure Bias
		+ Training
			+ the inputs are reference
		+ Testing 
			+ do not know the reference
			+ output of model is the input of the next step
+ Beam Search

# Pointer Network

## Tips

+ 添加End, 当做输出序列结束的标志

## Reference
+ [Pointer Networks]()
+ Reading Comprehension
  + R-NET
  + Match-LSTM
  + AOA
  + AS-READER
+ Summarization
  + Get the point: Summarization with Pointer-Generator Networks
  + Incorporating Copying Mechanisam in Sequence-to-Sequence Learning
  + Mulit-Source Pointer Network for Product Title Summarization
+ Chat-bot

## Overview
+ Pointer Network 可以看做是　Seq2Seq　+ Attention 的简化版 
+ Seq2Seq 无法解决输出序列的词汇表随着输入序列长度的改变而改变的问题，如寻找突包，对于这类问题，输出往往是输入集合的子集
+ 基于以上问题，　Pointer Network 设计了类似于编程语言中的指针，每个指针对应一个输入序列中的一个元素，从而可以直接操作序列而不需要特意设定输出词汇表（传统的seq2seq 需要设计词汇表）
  ![](https://pic3.zhimg.com/80/v2-8d34877424a3d47dc760c3947119813e_hd.jpg)
+ 上图是是给定$$p1$$ 到$$p4$$四个二维点的坐标, 要找到一个凸包, 显然答案是$$p_1 -> p_4 -> p_2 -> p_1 $$
+ 左侧图a是传统的seq2seq的做法(不太理解)
  + 把四个点的坐标作为输入序列输入进去
  + 提供一个词汇表[start, 1, 2, 3, 4, end]
  + 依据词汇表预测输出序列[start, 1, 4, 2, 1, end]
  + 缺点
    + 输入序列长度变化(比如为10)时, 无法预测大于4的数字 
+ 右图是Pointer Network
  + 预测的时候每一步都找当前输入序列中权重最大的那个元素,而且输出完全来自于输入序列,可以适应输入序列长度的变化

## Framework


# Self Attention



# Transformer

