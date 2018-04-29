# Probabilistic Graph Model(Structure Learning)

# Part I Define 
## Classification of PGM
+ directed graphical model (Also known as Bayesian Network)
    + Static Bayesian networks
    + Dynamic Bayesian networks
        + Hidden Markov Model
        + Kalman filter
+ Undirection graphical models
    + Markov networks
        + Gibbs Boltzman machine
        + Conditional random field

+ Structure Diagram

![](https://pic2.zhimg.com/v2-48dd591b8bc4775b95dd032983c5e729_r.jpg)

## Define of Structure Learing
+ Input and output are both objects with structures
+ objects : sequence, list, tree, bounding box, **not vectors**
+ find function f 
$$
	f : X \rightarrow Y
$$
+ Example Application
    + Speech recognition
        + X : speech signal(sequence) $\rightarrow$ Y : text(sequence)
    + Translation
        + 
    + Syntactic Parsing
        + X : sentence $\rightarrow$  Y : parsing tree(tree structure)
    + Object Detection
        + X : image $\rightarrow$ Y : bounding box
    + Summarization
        + X : long ducument $\rightarrow$ summary (short paragraph)
    + Retrieval
        + X : keyword $\rightarrow$ Y : search result(a list of webpage)

## Basic question of Structure Learning
+ Define
    + What does $ F(X,Y) $ look like ?
    + Also can be considered as probability $P(X,Y)$
        + MRF
        + Belief
        + Energy base model
+ Training
    + Find a function F, input is X, Y, output is a real number(R)
    $$ F : X * Y \rightarrow R$$
+ Inference
    + Given an object x, find $ \hat y $ as follow
    $$ \hat y = arg\ \underset{y \epsilon Y }{max} \  F(X,Y)$$ 
+ Detail example of Object Detection


# Part II Detail

## Static Bayesian networks
- pass

## Dynamic Bayesian networks - Hidden Markov Model
- Definition
	- Two basic assumption：
        1. Homogeneous markov assumption
            - The current state is only related to the previous state
        2. Observational independence assumption
            - The current observation is only related to the current state

    - parameters of HMM
        - Initial state probability vector : $\pi$
        - State transition probability matrix : A
        - Observation probability matrix ： B
        - HMM can be expressed as $\lambda = (\pi, A, B)$
    - example of Pos tagging
        - first generate pos sequence, then generate word based on current status
        + $\pi = [0.4, 0, 0.2, 0.3]$ means the probability of nd, v, p, ns as the initial state is 0.4, 0, 0.2, 0.3
        + transition probability A 
        
            |       | nd    | v | p | nt |
            | -     | --- | -- |---|-----|
            | nd    |   0.25 | 0.25 | 0.25 | 0.25 | 
            | v     |   0.25 | 0.25 | 0.25 | 0.25 | 
            | p     |   0.25 | 0.25 | 0.25 | 0.25 | 
            | nt    |   0.25 | 0.25 | 0.25 | 0.25 | 

        + observation probability matrix B
        
            |       |  北航  | 坐落 | 在 | 北京 |
            | -     | --- | -- |---|-----|
            | nd    |   0.25 | 0.25 | 0.25 | 0.25 | 
            | v     |   0.25 | 0.25 | 0.25 | 0.25 | 
            | p     |   0.25 | 0.25 | 0.25 | 0.25 | 
            | nt    |   0.25 | 0.25 | 0.25 | 0.25 | 
   
        ![](http://www.plantuml.com/plantuml/png/SoWkIImgAStDuShBBqbLo4bDAx4ABaa4CeDJ2qjJyv9JkJIqD1LqYpBJCqfqxHIKybAKk12yCcHE0J8dhrY9YmkaMa4t9Ryy3oJqj6VwYuvLIaWs-ISLfnQL9PPavkSXx5CgGzOpTyAB2NkLk9GAa0Ndh02Av1MZclrarngWbGwfUIb0xm00)

        - probability algorithm
            - direct calculation
                - Given $\lambda = (\pi, A, B)$ and observation sequence $O =(o_1, o_2, ..., o_T)$, Calculate the probability of $O =(o_1, o_2, ..., o_T)$'s occurrence
                - Probability of state sequence  $I =(i_1, i_2, ..., i_T)$ is as follow :
                $$ P( I | \lambda) = \pi_{i_1} a_{i_1i_2}a_{i_2i_3}...a_{i_{T-1} i_T}$$
                - Given state sequence $I =(i_1, i_2, ..., i_T)$, probability of observation sequence $O =(o_1, o_2, ..., o_T)$ is as follow:
                $$ P( O | I, \lambda) = b_{i_1}(o_1)b_{i_2}(o_2)...b_{i_T}(o_T)$$
                - Joint probability that state sequence and obserbvation sequence generate together :
                $$ P( O, T | \lambda) = P( O | I, \lambda) P( I | \lambda) = 
                    \pi_{i_1} a_{i_1i_2}a_{i_2i_3}...a_{i_{T-1} i_T}
                    b_{i_1}(o_1)b_{i_2}(o_2)...b_{i_T}(o_T)
                $$
                - Drawback of this method
                    - A large amount of calculation : $O(TN^T)$
                    - forward-backward algorithm will be more effective
            - forward-backward algorithm
                - forward algorithm
                    - Definition 
                    $$\alpha_t(i) = P(o_1, o_2,...,o_t,i_t=q_i|\lambda)$$
                    - forward algorithm of observation sequence 
                        - input : $\lambda$ and O
                        - output : $P(O|\lambda)$
                        1. Initval value
                            $$ \alpha_1(i) = \pi_ib_i(o_1)$$
                            $ o_1 $ is fixed by observation sequence
                        2. Recursive
                            - as t = 1, 2,...,T-1
                            $$ \alpha_{t+1}(i) = [ \sum_{j=1}^{N} \alpha_t(j) \alpha_{ji}]b_i(o_{t+1})\  ,\ i=1,2,...,N $$

                        3. Termination
                            $$ P(O|\lambda) = \sum_{j=1}^{N} \alpha_T(i)$$

                    - advantage
                        - A less amount of calculation $O(N^2 T)$, not $O(TN^T)
                
                - example
                    - ball and box model, $\lambda = (\pi, A, B)$, state set = {1,2,3}, obversivation set = {red, white}
                    
                    - Three different boxes and different ratios of red balls and white balls in each box

                    ![](http://www.plantuml.com/plantuml/png/SoWkIImgAStDuShBBqbLo4bDAx4ABaa4CeDJ2qjJyv9JkJIqD1LqYpBJCqfqxHIKCWsLk91uHYWyOoKkBf0A1TsK_F8yaD3pNOkKa8psJofEBIfBBCdCpqDO03G5MSVClKf08MDKGnAKk1nIyrA03WK0)

                    - transition matrix A : $$A = \begin{bmatrix} 0.5 \ 0.2  \ 0.3\\ 0.3 \ 0.5 \ 0.2  & \\ 0.2 \ 0.3 \ 0.5 &  & \end{bmatrix}$$
                    - obversition matrix B : $$ B = \begin{bmatrix} 0.5 \ 0.5 \\ 0.4 \ 0.6 & \\ 
                        0.7 \ 0.3 &  & \end{bmatrix} $$
                    - T = 3, O = {red, white, red}, $\pi = [0.2, 0.4, 0.4]$
                    - calculation process:
                        - initval value
                            - observe red
                            - choose box 1 : $\alpha_1(1)$ = 02 * 0.5 = 0.10
                            - choose box 2 : $\alpha_1(2)$ = 0.4 * 0.4 = 0.16
                            - choose box 3 : $\alpha_1(3)$ = 0.4 * 0.7 = 0.28
                        - Recursive
                            - demo 
                                
                                $(\alpha_1(1)*a_{11}$  means from box 1 to box 1
                                
                                $(\alpha_1(2)*a_{21}$  means from box 2 to box 1
                                
                                $(\alpha_1(3)*a_{31}$  means from box 3 to box 1
                                
                                $b_{11}$ means in box 1 (first 1) select red(second 1)
                                
                                $b_{12}$ means in box 1 (1) select white(2)
                                $b_{32}$ means in box 3 (3) select white(2)


                            - T=2, observed = white
                                + T=2, box(state)=1
                                
                                $\alpha_2(1) = (\alpha_1(1)*a_{11}+\alpha_1(2)*a_{21}+\alpha_1(3)*a_{31})*b_{12}$ = (0.10*0.5 + 0.16*0.3 + 0.28*0.3)*0.5 = 0.077

                                + T=2, box(state)=2
                                 $\alpha_2(2) = (\alpha_1(1)*a_{12}+\alpha_1(2)*a_{22}+\alpha_1(3)*a_{32})*b_{22}$ = (0.10*0.2 + 0.16*0.5 + 0.28*0.3)*0.6 = 0.1104
                                + T=2, box(state)=3 : $\alpha_2(1)$ = 0.0606
                            - T=3, observed = red
                                + T=3, box(state)=1 : $\alpha_3(1)$ = 0.04187
                                + T=3, box(state)=2 : $\alpha_3(1)$ = 0.03551
                                + T=3, box(state)=3 : $\alpha_3(1)$ = 0.05284
                        
                        + Termination
                            
                            $P(O|\lambda) = sum_{i=1}^{3} \alpha_3(i)$ = 0.13022

            - backward algorithm
            - 一些概率与期望值的计算

- Train
    - 监督学习方法
    - Baum-Welch 算法
    - Baum-Welch 模型参数估计公式 
- Inference            
    - 近似算法
    - 维特比算法

## CRF
- MRF
	- 模型定义
	- 概率无向图模型的因式分解
- CRF的定义与形式
	- 定义
	- 参数化形式
	- 简化形式
	- 矩阵形式
- 概率计算问题
	- 前向后向算法
	- 概率计算
	- 期望值计算
- 学习算法
	- 改进的迭代尺度法
	- 拟牛顿法
- 预测算法（根据观测序列预测状态序列）
	- 

## MEMM
+  用一个分布P(yi | yi-1,xi)来替代HMM中的两个条件概率分布，它表示从先前状态yi-1，在观察值xi下得到当前状态的概率，即根据前一状态和当前观察预测当前状态。每个这样的分布函数都是一个服从最大熵的指数模型。
![](http://s6.sinaimg.cn/mw690/001LIdk2zy6Fys1NGdL05&690)

## MRF
+ 马尔可夫性：对Markov随机场中的任何一个随机变量，给定场中其他所有变量下该变量的分布，等同于给定场中该变量的邻居节点下该变量的分布

## CRF
+ 一阶链式CRF示意图（不同于隐马尔科夫链，条件随机场中的xi **除了依赖于当前状态，还可能与其他状态有关**
![](http://s6.sinaimg.cn/mw690/001LIdk2gy6FzSV0xh3b5&690)
+ 全局最优体现在
	+ https://www.zhihu.com/question/53458773 
	+ CRF和HMM都有全局最优。这个全局最优和两者的区别是两件事情。你写下两者的loss function就能看到两个都是convex的，所以存在全局最优解。
+ crf++里的特征模板得怎么理解？
	+ https://www.zhihu.com/question/20279019
	```
	Unigram和Bigram模板分别生成CRF的状态特征函数  和转移特征函数  。
	其中  是标签，  是观测序列，  是当前节点位置。每个函数还有一个权值，具体请参考CRF相关资料。
    crf++模板定义里的%x[row,col]，即是特征函数的参数  。
    举个例子。假设有如下用于分词标注的训练文件：
    北    N    B
    京    N    E
    欢    V    B
    迎    V    M
    你    N    E
    其中第3列是标签，也是测试文件中需要预测的结果，有BME3种状态。
    第二列是词性，不是必须的。
    特征模板格式：%x[row,col]。x可取U或B，对应两种类型。
    方括号里的编号用于标定特征来源，row表示相对当前位置的行，0即是当前行；col对应训练文件中的列。
    这里只使用第1列（编号0），即文字。
    1、Unigram类型
    每一行模板生成一组状态特征函数，数量是L*N 个，L是标签状态数，N是模板展开后的特征数，也就是训练文件中行数， 这里L*N = 3*5=15。
    例如：
    U01:%x[0,0]，生成如下15个函数：
    func1 = if (output = B and feature=U01:"北") return 1 else return 0
    func2 = if (output = M and feature=U01:"北") return 1 else return 0
    func3 = if (output = E and feature=U01:"北") return 1 else return 0
    func4 = if (output = B and feature=U01:"京") return 1 else return 0
    ...
    func13 = if (output = B and feature=U01:"你") return 1 else return 0
    func14 = if (output = M and feature=U01:"你") return 1 else return 0
    func15 = if (output = E and feature=U01:"你") return 1 else return 0
    这些函数经过训练后，其权值表示函数内文字对应该标签的概率（形象说法，概率和可大于1）。
    又如
    U02:%x[-1,0]，训练后，该组函数权值反映了句子中上一个字对当前字的标签的影响。

    2、Bigram类型与Unigram不同的是，Bigram类型模板生成的函数会多一个参数：上个节点的标签  。生成函数类似于：func1 = if (prev_output = B and output = B and feature=B01:"北") return 1 else return 0这样，每行模板则会生成 L*L*N 个特征函数。经过训练后，这些函数的权值反映了上一个节点的标签对当前节点的影响。每行模版可使用多个位置。例如：U18:%x[1,1]/%x[2,1]字母U后面的01，02是唯一ID，并不限于数字编号。如果不关心上下文，甚至可以不要这个ID。
    参考：CRF++: Yet Another CRF toolkit
	```



## Diff
+ 条件随机场和隐马尔科夫链的关系和比较
	+ 条件随机场是隐马尔科夫链的一种扩展。
		+ 不同点：观察值xi不单纯地依赖于当前状态yi，可能还与前后状态有关；
		+ 相同点：条件随机场保留了状态序列的马尔科夫链属性——状态序列中的某一个状态只与之前的状态有关，而与其他状态无关。（比如句法分析中的句子成分）
+ MRF和CRF的关系和比较
	+ 条件随机场和马尔科夫随机场很相似，但又说不同，很容易弄混淆。最通用角度来看，CRF本质上是给定了观察值 (observations)集合的MRF。
    + 在图像处理中，MRF的密度概率 p(x=labels, y=image) 是一些随机变量定义在团上的函数因子分解。而CRF是根据特征产生的一个特殊MRF。因此一个MRF是由图和参数（可以无数个）定义的，如果这些参数是输入图像的一个函数（比如特征函数），则我们就拥有了一个CRF。
    + 图像去噪处理中，P(去噪像素|所有像素)是一个CRF，而P(所有像素)是一个MRF。

+ HMM,MEMM,CRF
	+ HMM模型是对转移概率和表现概率直接建模，统计共现概率。MEMM模型是对转移概率和表现概率建立联合概率，统计时统计的是条件概率，但MEMM容易陷入局部最优，是因为MEMM只在局部做归一化。CRF模型中，统计了全局概率，在做归一化时，考虑了数据在全局的分布，这样就解决了MEMM中的标记偏置的问题
	+ 下图很好诠释了HMM模型中两个假设：一是输出观察值之间严格独立，二是状态的转移过程中当前状态只与前一状态有关(一阶马尔可夫模型)。
    ![](https://flystarhe.github.io/images/2016-07-15-hmm-memm-crf-02.png)
	+ 下图说明MEMM模型克服了观察值之间严格独立产生的问题，但是由于状态之间的假设理论，使得该模型存在标注偏置问题。
	![](https://flystarhe.github.io/images/2016-07-15-hmm-memm-crf-03.png)
	+ 下图显示CRF模型解决了标注偏置问题，去除了HMM中两个不合理的假设。当然，模型相应得也变复杂了。
	![](https://flystarhe.github.io/images/2016-07-15-hmm-memm-crf-04.png)

    + HMM :
    	+ HMM模型将标注任务抽象成马尔可夫链，一阶马尔可夫链式针对相邻标注的关系进行建模，其中每个标记对应一个概率函数。HMM是一种产生式模型，定义了联合概率分布p(x,y)，其中x和y分别表示观察序列和相对应的标注序列的随机变量。为了能够定义这种联合概率分布，产生式模型需要枚举出所有可能的观察序列，这在实际运算过程中很困难，所以我们可以将**观察序列**的元素看做是彼此孤立的个体，即假设每个元素彼此独立，任何时刻的观察结果只依赖于该时刻的状态。
    	+ HMM模型的这个假设前提在比较小的数据集（也不全是吧）上是合适的，但实际上在大量真实语料中观察序列更多的是以一种多重的交互特征形式表现的，观察元素之间广泛存在长程相关性。例如，在命名实体识别任务中，由于实体本身结构所具有的复杂性，利用简单的特征函数往往无法涵盖所有特性，这时HMM的假设前提使得它无法使用复杂特征(它无法使用多于一个标记的特征)，这时HMM的弊端就显现无疑了。突破这一瓶颈的方法就是引入最大熵模型(ME)
    + ME : 
    	+ 最大熵模型可以使用任意的复杂相关特征，在性能上也超过了Bayes分类器。
        + 最大熵模型的优点：
            + 首先，最大熵统计模型获得的是所有满足约束条件的模型中信息熵极大的模型；
            + 其次，最大熵统计模型可以灵活地设置约束条件，通过约束条件的多少可以调节模型对未知数据的适应度和对已知数据的拟合程度；
            + 再次，它还能自然地解决了统计模型中参数平滑的问题。
        + 最大熵模型的不足：
            + 首先，最大熵统计模型中二值化特征只是记录特征的出现是否，而文本分类需要知道特征的强度，因此，它在分类方法中不是最优的；
            + 其次，由于算法收敛的速度较慢，所以导致最大熵统计模型它的计算代价较大，时空开销大；
            + 再次，数据稀疏问题比较严重。最致命的是，作为一种分类器模型，最大熵对每个词都是单独进行分类的，标记之间的关系无法得到充分利用。然而，具有马尔可夫链的HMM模型可以建立标记之间的马尔可夫关联性，这是最大熵模型所没有的。
	+ MEMM :
		+ 简单来说，MEMM把HMM模型和ME模型的优点集合成一个统一的产生式模型，这个模型允许状态转移概率依赖于序列中彼此之间非独立的特征上，从而将上下文信息引入到模型的学习和识别过程中，达到了提高识别的准召率的效果。有实验证明，MEMM在序列标注任务上表现的比HMM和无状态的最大熵模型要好得多。然而，如上面所述，MEMM并不完美，它存在明显的标记偏置问题
	+ CRF :+
		+ CRF模型具有以下特点：
			+ CRF在给定了观察序列的情况下，对整个的序列的联合概率有一个统一的指数模型，它具备一个比较吸引人的特性就是其损失函数的凸面性；
			+ CRF具有很强的推理能力，并且能够使用复杂、有重叠性和非独立的特征进行训练和推理，能够充分地利用上下文信息作为特征，还可以任意地添加其他外部特征，使得模型能够获取的信息非常丰富；
			+ CRF解决了MEMM中的标记偏置问题，这也正是CRF与MEMM的本质区别所在。最大熵模型在每个状态都有一个概率模型，在每个状态转移时都要进行归一化。如果某个状态只有一个后续状态，那么该状态到后续状态的跳转概率即为1。这样，不管输入为任何内容，它都向该后续状态跳转。而CRF是在所有的状态上建立一个统一的概率模型，这样在进行归一化时，即使某个状态只有一个后续状态，它到该后续状态的跳转概率也不会为1。


---

## Reference websites
+ https://www.zhihu.com/question/23255632
+ (很好) https://zhuanlan.zhihu.com/p/33397147
+ http://blog.sina.com.cn/s/blog_4b1645570102vk3d.html
+ 较好：
    + https://flystarhe.github.io/2016/07/13/hmm-memm-crf/
+ LSTM+CRF:
    + https://createmomo.github.io/2018/01/27/Table-of-Contents/