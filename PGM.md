# Probabilistic Graph Model(Structure Learning)

# Part I Define 

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
        + X : text(sequence) $\rightarrow$ Y : text(sequence)
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

## Algorithm of Structure Learning(Probability of Graph Model)
+ directed graphical model (Also known as Bayesian Network)
    + Static Bayesian networks
    + Dynamic Bayesian networks
        + Hidden Markov Model
        + Kalman filter
+ Undirection graphical models
    + Markov networks(Also known as Markov random field)
        + Gibbs Boltzman machine
        + Conditional random field

+ Structure Diagram

![](https://pic2.zhimg.com/v2-48dd591b8bc4775b95dd032983c5e729_r.jpg)

# Part II Algorithm

## Static Bayesian Networks
- pass

## Dynamic Bayesian Networks - Hidden Markov Model
- problems that can be solved by the HMM
    - Speech Recognition
    - Text Seg
    - Text Pos

#### Definition
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

#### probability algorithm
- direct calculation
    - Given $\lambda = (\pi, A, B)$ and observation sequence $O =(o_1, o_2, ..., o_T)$, Calculate the probability of $O =(o_1, o_2, ..., o_T)$'s occurrence
    - Probability of state sequence  $I =(i_1, i_2, ..., i_T)$ is as follow :
    $$ P( I | \lambda) = \pi_{i_1} a_{i_1i_2}a_{i_2i_3}...a_{i_{T-1} i_T}$$
    - Given state sequence $I =(i_1, i_2, ..., i_T)$, probability of observation sequence $O =(o_1, o_2, ..., o_T)$ is as follow:
    $$ P( O | I, \lambda) = b_{i_1}(o_1)b_{i_2}(o_2)...b_{i_T}(o_T)$$
    - Joint probability that state sequence and obserbvation sequence generate together :
    $$ P( O, I | \lambda) = P( O | I, \lambda) P( I | \lambda) = 
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
        - Process
            - input : $\lambda$ and O
            - output : $P(O|\lambda)$
            1. Initial value
                $$ \alpha_1(i) = \pi_ib_i(o_1)$$
                $ o_1 $ is fixed by observation sequence
            2. Recursive
                - as t = 1, 2,...,T-1
                $$ \alpha_{t+1}(i) = [ \sum_{j=1}^{N} \alpha_t(j) \alpha_{ji}]b_i(o_{t+1})\  ,\ i=1,2,...,N $$

            3. Termination
                $$ P(O|\lambda) = \sum_{j=1}^{N} \alpha_T(i)$$

        - advantage
            - A less amount of calculation $O(N^2 T)$, not $O(TN^T)$
    
        + example
            + ball and box model, $\lambda = (\pi, A, B)$, state set = {1,2,3}, obversivation set = {red, white}
            
            + Three different boxes and different ratios of red balls and white balls in each box

            ![](http://www.plantuml.com/plantuml/png/SoWkIImgAStDuIh8LD2rKr3AD5JYqYtAJCye0VECK7Z6IbnS81KAkYdvvNaWeUUx5Ya1XOoGXMRk1GaPewfAKE9oICrB0Te40000)
            
            + transition matrix A : 
             $$A = \begin{bmatrix} 0.5 \ 0.2 \ 0.3;\\ 0.3 \ 0.5 \ 0.2;\\ 0.2 \ 0.3 \ 0.5; \end{bmatrix}$$

            + obversition matrix B : 
             $$ B = \begin{bmatrix} 0.5 \ 0.5;\\ 0.4 \ 0.6;\\ 
                0.7 \ 0.3;\end{bmatrix} $$
    
            + T = 3, O = {red, white, red}, $\pi = [0.2, 0.4, 0.4]$
    
            + **calculation process:**
                + initval value
                    + observe red
                    + choose box 1 : $\alpha_1(1)$ = 02 * 0.5 = 0.10
                    + choose box 2 : $\alpha_1(2)$ = 0.4 * 0.4 = 0.16
                    + choose box 3 : $\alpha_1(3)$ = 0.4 * 0.7 = 0.28
                + Recursive
                    + meaning of the symbol
                        o_1 = red, o_2 = white, o_3 = red

                        $ \alpha_1(1)*a_{11}$  means from box 1 to box 1
                        
                        $ \alpha_1(2)*a_{21}$  means from box 2 to box 1
                        
                        $ \alpha_1(3)*a_{31}$  means from box 3 to box 1
                        
                        $b_1(o_1)$ means in box 1 select red(o_1)
                        
                        $b_1(o_3)$ means in box 1 select red(o_3)
                        
                        $b_3(o_2)$ means in box 3 select white(o_2)

                    + T=2, observed = white
                        - T=2, box(state)=1
                        
                        $$\alpha_2(1) = (\alpha_1(1)*a_{11}+\alpha_1(2)*a_{21}+\alpha_1(3)*a_{31})*b_1(o_2) = (0.10*0.5 + 0.16*0.3 + 0.28*0.3)*0.5 = 0.077$$

                        - T=2, box(state)=2
                         
                        $$\alpha_2(2) = (\alpha_1(1)*a_{12}+\alpha_1(2)*a_{22}+\alpha_1(3)*a_{32})*b_2(o_2) = (0.10*0.2 + 0.16*0.5 + 0.28*0.3)*0.6 = 0.1104$$
                        
                        - T=2, box(state)=3 
                        $$\alpha_2(1) = 0.0606$$
                                
                    + T=3, observed = red
                        * T=3, box(state)=1 : $\alpha_3(1)$ = 0.04187
                        * T=3, box(state)=2 : $\alpha_3(1)$ = 0.03551
                        * T=3, box(state)=3 : $\alpha_3(1)$ = 0.05284
                        
                    + Termination
                            
                        $P(O|\lambda) = \sum_{i=1}^{3} \alpha_3(i)$ = 0.13022

    + backward algorithm
        + Definition
            $$\beta_t(i) = P(o_t+1,o_t+2,...,o_T|i_t=q_i,\lambda)$$
        + Process
            + input : $\lambda$ and O
            + output : $P(O|\lambda)$
            1. initial value
                $$\beta_T(i)=1,\ i=1,2,...,N$$
            2. as t = T-1, T-2,...,1
                $$ \beta_{t}(i) = \sum_{j=1}^{N} a_{ij}\ b_j(o_{t+1}) \ \beta_{t+1}(j),\ i=1,2,...,N $$
            3. Termination
                $$ P(O|\lambda) = \sum_{i=1}^{N} \pi_{i} \ b_i(o_1) \ \beta_1(i)$$
        + example
            + pass

    + forward-backward algorithm
        $$ P(O|\lambda) = \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_t(i)\ a_{ij}\ b_j(o_{t+1})\ \beta_{t+1}(j)$$

#### Train
- Supervised learning methods (MLE)
    - Assume that training data has been given $\{(O_1,I_1),(O_2,I_2),...,(O_T,I_T)\}$, T is the sequence length
    - Use maximum likelihood estimation to estimate HMM parameters
    1. estimation of transition matrix
    2. estiamtion of observation matrix
    3. estimation of initial value


- Unsuperised learning methods (Unknown label is a hidden variable， EM)
    - Baum-Welch algorithm(Also known as expectation maximization algorithm (EM))
    - Estimation formula of Baum-Welch model parameters 
    1. Determine the log-likelihood function of complete data
        + observation data $O=(o_1,o_2,...o_T)$
        + hidden data $I=(i_1,i_2,...,i_T)$
        + complete data $(O,I)={o_1,o_2,...o_T,i_1,i_2,...,i_T}$
        + log-likelihood function of complete data $log P(O,I|\lambda)$

    2. E-Step
    	+ reference : cross entropy

            ![](https://images2015.cnblogs.com/blog/517947/201702/517947-20170223095747882-1710070465.png) 

        + Find Q function(Why define like this?) 
            $$Q(\lambda, \bar{\lambda}) = \sum_I logP(O,I|\lambda)P(O,I|\bar\lambda)$$
            + $\bar{\lambda}$ is the current estimate of the hmm parameter
            + $\lambda$ is the hmm parameter to maximize

        + Due to 
            $$ P( O, T | \lambda) = P( O | I, \lambda) P( I | \lambda) = 
                \pi_{i_1} a_{i_1i_2}a_{i_2i_3}...a_{i_{T-1} i_T}
                b_{i_1}(o_1)b_{i_2}(o_2)...b_{i_T}(o_T)
            $$
        
        + So
            $$Q(\lambda, \bar{\lambda}) = \sum_{I}log\pi_{i_1}P(O,I|\lambda)) + \sum_{I}\{\sum_{t=1}^{T-1}log a_{i_ti_{t+1}}\}P(O,I|\lambda)) + \sum_{I}\{\sum_{t=1}^{T-1}log b_{i_t}(o_t)\}P(O,I|\lambda))\ \ \ (1-1)$$
    
    3. M-Step
        + Maximize Q Function to Find Model Parameters $\pi, A, B$
        + Maximize the three terms above
            + First Item 
                + $\sum_{I}log\pi_{i_1}P(O,I|\lambda)) = \sum_{i=1}^{N}log\pi_iP(O,i_1=i|\bar\lambda )$
                + Constraints are $\sum_{i=1}^{N} \pi_i =1$
                + Lagrangian multiplier method
                    $$ L =  \sum_{i=1}^{N}log\pi_iP(O,i_1=i|\bar\lambda ) + \gamma\{\sum_{i=1}^{N} \pi_i =1\}$$
                    + Get Partial derivative
                    $$ \frac{\partial L}{\partial \pi_i} = 0 \ \ \ (1-2)$$
                    + Get 
                    $$ P(O,i_1=i|\bar\lambda ) + \gamma\pi_i $$
                    + Sum i then get
                    $$ \gamma = - P(O|\bar\lambda) $$
                    + Bring into Formula 1-2 then get
                    $$ \pi_i = \frac{P(O,i_1=i| \bar\lambda )}{P(O|\bar\lambda))}\ \ \ (1-3)$$
            
            + Second Item
                + Constraints are $\sum_{j=1}^N a_{ij} =1$
                + Get 
                    $$ a_{ij} = \frac{ \sum_{t=1}^{T-1} P(O,i_t=i,i_t+1 =j | \bar\lambda )}{P(O,i_t = i|\bar\lambda))}\ \ \ (1-4)$$
            
            + Third Item
                + Constraints are $\sum_{j=1}^{N} b_j(k) =1$
                    $$ b_{j}(k) = \frac{ \sum_{t=1}^{T-1} P(O,i_t=j | \bar\lambda ) I(o_t = v_k)}{P(O,i_t = j|\bar\lambda))}\ \ \ (1-5)$$
    
    4. Baum-Welch model parameter estimation formula
            
#### Inference
- Approximation algorithm
    - The idea of this algorithm is Select the most likely state $i^{*}_{t}$ at each time t
        + Given O and $\lambda$, then the probability of being in state q_i at time t is
            $$\gamma_{t}(i) = \frac{\alpha_t(i) \beta_t(i)}{P(O|\lambda)} = \frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^{N}\alpha_t(j) \beta_t(j)}$$
        + The most likely state $i^{*}_{t}$ is
            $$ i^{*}_{t} = arg\ \underset{1\leq i \leq N}{max} [\gamma_t(i)], t = 1,2,...,T$$

- Viterbi algorithm
    - Overview
        - The viterbi algorithm uses dynamic programming to solve the HMM inference problem
        - The path with the greatest probability, which corresponds to a sequence of states
        - 根据这一原理，从t=1时刻开始，递推的计算在时刻状态为i的各条路径的最大概率，直到时刻t=T状态为i的各条路径的最大概率
        - 时刻t=T的最大概率即最有路径的概率$p^*$ ，最优路径的终止节点为$i^*_T$ 
        - 从终止节点开始由后向前得到各个节点，进而得到最有路径
    
    - key in formula 
        - t is Timing subscript
        - i, j are State subscript
        - $o_1,...,o_T$ is Observation label

    - Detail
        - Define variable $\delta$
            - The time is t, the state is i, the maximum value of the path (path consisit of $i_1,...,i_{t_1}$), variable is $i_1,...,i_{t_1}$
            $$\delta_t(i) = \underset{i_1,i_2,...,i_{t-1}}{max} P(i_t=i,i_{t-1}...,i_1,o_t,...,o_1|\lambda) $$
            $$ \delta_{t+1}(i) = \underset{1\leq j \leq N}{max} [\delta_t(i) a_{ji}]b_i(o_{t+1})$$
        
        - Define variable $\varphi $
            - The time is t, the state is i, maximum probability path is $i_1,...,i_{t_1}$
            - the value fo state t-1 is as follow, the range of values is $1\leq j \leq N$
            $$  \varphi_t(i) = arg \underset{1\leq j \leq N}{max} [\delta_{t-1}(j) a_{ji}] $$
    
    - Summary of viterbi
        - input : $\lambda = (A, B, \pi)$ and O
        - output : Optimal path
        1. initial
            
            $$\delta_1(i) = \pi_i b_i(o_1)$$
            
            $$\varphi_1(i) = 0$$

        2. Recursive
            $$ \delta_{t+1}(i) = \underset{1\leq j \leq N}{max} [\delta_t(i) a_{ji}]b_i(o_{t+1})$$
            $$  \varphi_t(i) = arg \underset{1\leq j \leq N}{max} [\delta_{t-1}(j) a_{ji}] $$
        3. Termination
            $$ P^{\star} = \underset{1\leq j \leq N}{max} \delta_T(i)$$
            $$ i_T^{\star} = arg \underset{1\leq j \leq N}{max} [\delta_T(i)]$$
    
    - Example
        - condition 
            - $\lambda = (A, B, \pi)$
            - A : $$ A = \begin{bmatrix} 0.5 \ 0.2  \ 0.3\\ 0.3 \ 0.5 \ 0.2  \\ 0.2 \ 0.3 \ 0.5\end{bmatrix}$$
            - B : $$ B = \begin{bmatrix} 0.5 \ 0.5 \\ 0.4 \ 0.6\\ 0.7 \ 0.3 \end{bmatrix} $$
            - $\pi = [0.2, 0.4, 0.4]$ 
            - T = 3
            - O = {red, white, red}
        
        - question : Finding the optimal state sequence?
        
        - process
            - initial
                - $\delta_1(1) = 0.1$, $\delta_1(2)=0.16$, $\delta_1(3)=0.28$
                - $\varphi_1(1) = 0$, $\varphi_1(2) = 0 $, $\varphi_1(3) = 0$
            
            - resursive
                - $\delta_2(1) = \underset{1\leq i \leq 3}{max} [\delta_1(j) a_{ji}]b_1(o_2)0.028$, $\varphi_2(1) = 3$
                - $\delta_2(2) = 0.0504$, $\varphi_2(2) = 3$
                - $\delta_2(3) = 0.042$, $\varphi_2(3) = 3$
                - $\delta_3(1) = 0.00756$, $\varphi_3(1)=2$
                - $\delta_3(2) = 0.01008$, $\varphi_3(2)=2$
                - $\delta_3(3) = 0.0147$, $\varphi_3(3)=2$
            
            - termination
                - $P^{\star} = \underset{1\leq i \leq 3}{max}  \delta_3(i) = \delta_3(3) = 0.0147$
                - $i_3^{\star} = 3$
                    - t=2, $i_2^{\star} = 3$
                    - t=1, $i_1^{\star} = 3$
                - optimal state sequence is (3,3,3)

#### Drawbacks of HMM
- correct result is $(x,\hat y)$ 
- cant ensure $P(x,\hat y) > P(x,y)$, y is other label
- example
    - P(V|N)=9/10, P(D|N)=1/10
    - P(a|V)=1/2, P(a|D)= 1
    - N -> V --> c 9
    - P -> V --> a 9
    - N -> D --> a 1
    - request N -> ? --> a
    - use HMM will get N -> V --> a (Highest probability of HMM), not contain in traing data
- Will generate some sequence never happened in training data
    - transition matrix and emission matrix are trained separatly
- The (x,y) never seen in the training data can have large probability P(x,y)
- More complex model can deal with this problem
- However, CRF can deal with this problem based on the same model

#### Benefit of HMM
    - Suitable for training data is small 

## Markov networks - CRF
- Define and Probability calculation
    + $P(x,y)\ \epsilon \ exp(w\ \cdot\ \phi (x,y) )$
        + $\phi$ (x,y) is a feature vector
        + w is the weight vector to be learned from training data
    + $P(y\ |\ x) = \frac{P(x,y)} { \sum_{y'} P(x, y')}$ 

    + Diff from HMM (P(x,y) for CRF)
        + In HMM $P(x,y) = P(y_1|\ start)\ \prod_{l=1}^{L-1} P(y_{l+1}|y_l)\ P(end|y_L)\ \prod^{L}_{l=1}P(x_l|y_l)$
        + $log P(x,y) = logP(y_1 | start) + \sum_{l=1}^{L-1}logP(y_{l+1}|y_l) + log P(end|y_L) + \sum_{l=1}^{L} logP(x_l|y_l)$
        + the last item of last formula
            + $\sum_{l=1}^{L} logP(x_l|y_l)$ = \sum_{s,t} log P(s|t) \times N_{s,t}(x,y)$
            + $log P(s|t)$ : Log probability of word given tag s ()
            + $N_{s,t}(x,y)$ : Number fo tag s and word t appears together in (x,y)
            ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM1.png)
            ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM2.png)
            ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM3.png)
        + Define Feature Vector
            ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM4.png)
            ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM5.png)
            ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM6.png) 

- Training
    - cost function like crosss entropy
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM7.png)
        - Maximize what we boserve in training data
        - Minimize what we dont observe in training data
    - gredient Assent
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM8.png)
    - process
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM9.png)
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM10.png)
    - right - wrong
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM11.png)


- Inference
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM12.png)
- CRF v.s. HMM
    - adjust P(a|V) -> 0.1
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM13.png)
- Synthetic Data
    - First paper purpose CRF
    - comparing HMM and CRF
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM14.png)

- CRF Summary
   ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM15.png) 

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

## Diff
- Reference
    - https://www.zhihu.com/question/46688107/answer/117448674
        - LSTM：像RNN、LSTM、BILSTM这些模型，它们在序列建模上很强大，它们能够capture长远的上下文信息，此外还具备神经网络拟合非线性的能力，这些都是crf无法超越的地方，对于t时刻来说，输出层y_t受到隐层h_t（包含上下文信息）和输入层x_t（当前的输入）的影响，但是y_t和其他时刻的y_t`是相互独立的，感觉像是一种point wise，对当前t时刻来说，我们希望找到一个概率最大的y_t，但其他时刻的y_t`对当前y_t没有影响，如果y_t之间存在较强的依赖关系的话（例如，形容词后面一般接名词，存在一定的约束），LSTM无法对这些约束进行建模，LSTM模型的性能将受到限制。

        - CRF：它不像LSTM等模型，能够考虑长远的上下文信息，它更多考虑的是整个句子的局部特征的线性加权组合（通过特征模版去扫描整个句子）。关键的一点是，CRF的模型为p(y | x, w)，注意这里y和x都是序列，它有点像list wise，优化的是一个序列y = (y1, y2, …, yn)，而不是某个时刻的y_t，即找到一个概率最高的序列y = (y1, y2, …, yn)使得p(y1, y2, …, yn| x, w)最高，它计算的是一种联合概率，优化的是整个序列（最终目标），而不是将每个时刻的最优拼接起来，在这一点上CRF要优于LSTM。

        - HMM：CRF不管是在实践还是理论上都要优于HMM，HMM模型的参数主要是“初始的状态分布”，“状态之间的概率转移矩阵”，“状态到观测的概率转移矩阵”，这些信息在CRF中都可以有，例如：在特征模版中考虑h(y1), f(y_i-1, y_i), g(y_i, x_i)等特征。

        - CRF与LSTM：从数据规模来说，在数据规模较小时，CRF的试验效果要略优于BILSTM，当数据规模较大时，BILSTM的效果应该会超过CRF。从场景来说，如果需要识别的任务不需要太依赖长久的信息，此时RNN等模型只会增加额外的复杂度，此时可以考虑类似科大讯飞FSMN（一种基于窗口考虑上下文信息的“前馈”网络）。

        - CNN＋BILSTM＋CRF：
            - 这是目前学术界比较流行的做法，BILSTM＋CRF是为了结合以上两个模型的优点，CNN主要是处理英文的情况，英文单词是由更细粒度的字母组成，这些字母潜藏着一些特征（例如：前缀后缀特征），通过CNN的卷积操作提取这些特征，在中文中可能并不适用（中文单字无法分解，除非是基于分词后），这里简单举一个例子，例如词性标注场景，单词football与basketball被标为名词的概率较高， 这里后缀ball就是类似这种特征。

        - BILSTM+CRF的Tensorflow版本：https://github.com/chilynn/sequence-labeling，主要参考了GitHub - glample/tagger: Named Entity Recognition Tool的实现，tagger是基于theano实现的，每一轮的参数更新是基于一个样本的sgd，训练速度比较慢。sequence-labeling是基于tensorflow实现的，将sgd改成mini-batch sgd，由于batch中每个样本的长度不一，训练前需要padding，最后的loss是通过mask进行计算（根据每个样本的真实长度进行计算）。
        - 参考论文：
            - https://arxiv.org/pdf/1603.01360v3.pdf
            - https://arxiv.org/pdf/1603.01354v5.pdf
            - http://arxiv.org/pdf/1508.01991v1.pdf




---

## Reference websites
[1] LHY : http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/Sequence.pdf
[2] 李航，统计学习方法


+ https://www.zhihu.com/question/23255632
+ (很好) https://zhuanlan.zhihu.com/p/33397147
+ http://blog.sina.com.cn/s/blog_4b1645570102vk3d.html
+ 较好：
    + https://flystarhe.github.io/2016/07/13/hmm-memm-crf/
+ LSTM+CRF:
    + https://createmomo.github.io/2018/01/27/Table-of-Contents/
