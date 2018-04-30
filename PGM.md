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
                            - meaning of the symbol
                                o_1 = red, o_2 = white, o_3 = red

                                $(\alpha_1(1)*a_{11}$  means from box 1 to box 1
                                
                                $(\alpha_1(2)*a_{21}$  means from box 2 to box 1
                                
                                $(\alpha_1(3)*a_{31}$  means from box 3 to box 1
                                
                                $b_{1o_1}$ means in box 1 select red(o_1)
                                
                                $b_{1o_3}$ means in box 1 select red(o_3)
                                
                                $b_{3o_2}$ means in box 3 select white(o_2)


                            - T=2, observed = white
                                + T=2, box(state)=1
                                
                                $\alpha_2(1) = (\alpha_1(1)*a_{11}+\alpha_1(2)*a_{21}+\alpha_1(3)*a_{31})*b_{1o_2}$ = (0.10*0.5 + 0.16*0.3 + 0.28*0.3)*0.5 = 0.077

                                + T=2, box(state)=2
                                 
                                 $\alpha_2(2) = (\alpha_1(1)*a_{12}+\alpha_1(2)*a_{22}+\alpha_1(3)*a_{32})*b_{2o_2}$ = (0.10*0.2 + 0.16*0.5 + 0.28*0.3)*0.6 = 0.1104
                                + T=2, box(state)=3 : $\alpha_2(1)$ = 0.0606
                                
                            - T=3, observed = red
                                + T=3, box(state)=1 : $\alpha_3(1)$ = 0.04187
                                + T=3, box(state)=2 : $\alpha_3(1)$ = 0.03551
                                + T=3, box(state)=3 : $\alpha_3(1)$ = 0.05284
                        
                        + Termination
                            
                            $P(O|\lambda) = sum_{i=1}^{3} \alpha_3(i)$ = 0.13022

            - backward algorithm
                - Definition
                    $$\beta_t(i) = P(o_t+1,o_t+2,...,o_T|i_t=q_i,\lambda)$$
                - Process
                    - input : $\lambda$ and O
                    - output : $P(O|\lambda)$
                    1. initial value
                        $$\beta_T(i)=1,\ i=1,2,...,N$$
                    2. - as t = T-1, T-2,...,1
                        $$ \beta_{t}(i) = \sum_{j=1}^{N} a_{ij}\ b_j(o_{t+1}) \ \beta_{t+1}(j),\ i=1,2,...,N $$
                    3. Termination
                        $$ P(O|\lambda) = sum_{i=1}^{N} \pi_{i} \ b_i(o_1) \ \beta_1(i)$$
            - forward-backward algorithm
                $$ P(O|\lambda) = \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_t(i)\ a_{ij}\ b_j(o_{t+1})\ \beta_{t+1}(j)$$

- Train
    - Supervised learning methods (MLE)
        - Assume that training data has been given $\{(O_1,I_1),(O_2,I_2),...,(O_T,I_T)\}$, T is the sequence length
        - Use maximum likelihood estimation to estimate HMM parameters
        1. estimation of transition matrix
        2. estiamtion of observation matrix
        3. estimation of initial value
    - Unsuperised learning methods (Unknown label is a hidden variable， EM)
        - Baum-Welch algorithm(Also known as expectation maximization algorithm (EM))
        - Estimation formula of Baum-Welch model parameters 
- Inference            
    - Approximation algorithm
    - Viterbi algorithm




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




---

## Reference websites
[1] LHY
[2] 李航，统计学习方法


+ https://www.zhihu.com/question/23255632
+ (很好) https://zhuanlan.zhihu.com/p/33397147
+ http://blog.sina.com.cn/s/blog_4b1645570102vk3d.html
+ 较好：
    + https://flystarhe.github.io/2016/07/13/hmm-memm-crf/
+ LSTM+CRF:
    + https://createmomo.github.io/2018/01/27/Table-of-Contents/