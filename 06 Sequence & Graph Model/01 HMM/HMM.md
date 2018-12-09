[TOC]

# Hidden Markov Model

### Reference

+ 李开复1988年的博士论文发表了第一个基于隐马尔科夫模型（HMM）的语音识别系统Sphinx，被《商业周刊》评为1988年美国最重要的科技发明
  + 出处请见[KaifuLeeHMM](http://tech.sina.com.cn/zl/post/detail/it/2014-04-09/pid_8446205.htm#483253-tsina-1-23485-1cf60a7c37a7bc296a2ba7aba0120190)
+ https://yanyiwu.com/work/2014/04/07/hmm-segment-xiangjie.html


### Application Scenario

- Speech Recognition
  - EM
- Text Sequence Labeling
  - SEG/POS/NER

### Definition

+ Parameters of HMM

  + length of text : T

    + **t is text subscript**

  + word set : WS

  + number of word set : M

  + hidden state set : HS

  + number of hidden state set : N

    + **i, j are hidden state subscript**

  + Initial state probability vector : $\pi$

  + State transition probability matrix : A ( dimension is : N * N )
    $$
    \left\{
    \begin{matrix}
    a_{I_1 I_1} & a_{I_1 I_2} & ... & a_{I_1 I_N} \\
    a_{I_2 I_1} & a_{I_2 I_2} & ... & a_{I_2 I_N} \\
    ... & ... & ... & ... \\
    a_{I_N I_1} & a_{I_N I_2} & ... & a_{I_N I_N}
    \end{matrix}
    \right\} \tag{1}
    $$

  + Observation probability matrix ： B(dimension is : M * N)
    $$
    \left\{
    \begin{matrix}
    b_{I_1 O_1} & b_{I_1 O_2} & ... & b_{I_1 O_M} \\
    b_{I_2 O_1} & b_{I_2 O_2} & ... & b_{I_2 O_M} \\
    ... & ... & ... & ... \\
    b_{I_N O_1} & b_{I_N O_2} & ... & b_{I_N O_M}
    \end{matrix}
    \right\} \tag{2}
    $$

  + HMM can be expressed as $\lambda = (\pi, A, B)$

+ Structure of HMM

  + Hidden State Sequence
    + $${I_1, I_2,...,I_T}$$
  + Observation State Sequence
    + $O_1,  O_2, ... , O_T$

  ![](http://www.codeproject.com/KB/recipes/559535/gerative-discriminative.png)

- Basic assumption：

    - Markov assumption : 有限历史假设

        - The current state is only related to the previous state

        ​	
        $$
        P(I_t|I_{t-1},I_{t-2},... ,I_1) = P(I_t|I_{t-1}) \tag{3}
        $$

    - Homogeneous  assumption : 齐次性假设
        $$
        P(I_t|I_{t-1}) = P(I_r|I_{r-1}) \tag{4}
        $$


    + Observational independence assumption : 观测值独立假设 
      + The current observation is only related to the current state
      	$$
      	P(O_t|I_t, I_{t-1},...,I_1) = P(O_t|I_t) \tag{5}
      	$$


### Problem One ：Probability Algorithm

+ Define of problem
  + Given $\lambda = (\pi, A, B)$ and observation sequence $O =(O_1, O_2, ..., O_T)$
  + Calculate the probability of $O =(O_1, O_2, ..., O_T)$'s occurrence

- Solution A ：direct calculation
    - Probability of state sequence  $I =(I_1, I_2, ..., I_T)$ is as follow :
      $$
      P( I | \lambda) = \pi_{I_1} a_{I_1I_2}a_{I_2I_3}...a_{I_{T-1} I_T} \tag{6}
      $$

    - Given state sequence $I =(I_1, I_2, ..., I_T)$, probability of observation sequence $O =(O_1, O_2, ..., O_T) $is as follow:
      $$
      P( O | I, \lambda) = b_{I_1 O_1}b_{I_2 O_2}...b_{I_T O_T} \tag{7}
      $$

    - Joint probability that state sequence and obserbvation sequence generate together :
      $$
      P( O, I | \lambda) = P( O | I, \lambda) P( I | \lambda) = 
        \pi_{I_1} a_{I_1I_2}a_{I_2I_3}...a_{I_{T-1} I_T} b_{I_1 O_1}b_{I_2 O_2}...b_{I_T O_T} \tag{8}
      $$

    - Drawback of this method
        - A large amount of calculation :$O(TN^T)$
          - **need check**
        - forward-backward algorithm will be more effective

- Solution B : forward-backward algorithm
    - B-1 : forward algorithm
        - Definition 

          + $$\alpha_t(I_t) = P(O_1, O_2,...,O_t, I_t|\lambda)$$

        - Process

            - Initial value
                + $$ \alpha_1(I_1) = \pi_{I_1}b_{I_1 O_1}$$
                + Observation State Sequence is {$$O_1$$}
            - Recursive, as t = 1, 2,...,T-1
                - $$ \alpha_{t+1}(I_{t+1}) = \{ \sum_{state}^{HS} \alpha_t(state)  a_{state\  I_{t+1}  } \} b_{I_{t+1}}(O_{t+1})$$

            + Termination
              $$ P(O|\lambda) = \sum_{state}^{HS} \alpha_T(state)$$

    + B-2 : backward algorithm
      + Definition
          + $\beta_t(I_t) = P(O_{t+1},O_{t+2},...,O_T|I_t,\lambda)$
      + Process
          + initial value
              + $\beta_T(state)=1,\ state \in HS$
          + Recursive, as t = T-1, T-2,...,1
              + $$ \beta_{t}(I_t) =\{ \sum_{state}^{HS} a_{I_t\ state}\ b_{state\ O_{t+1}} \ \beta_{t+1}(state) \}$$
          + Termination
              + $$ P(O|\lambda) = \sum_{state}^{HS} \pi_{state} \ b_{state}(o_1) \ \beta_1(state)$$

    + B-3 : forward-backward algorithm
        $$
        P(O|\lambda) = \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_t(i)\ a_{ij}\ b_j(o_{t+1})\ \beta_{t+1}(j),  t=1, 2, ..., T-1 \tag{9}
        $$

    + advantage

        - A less amount of calculation $O(N^2 T)$, not $O(TN^T)$



### Problem Two ：Train

- Solution A：Supervised learning methods (MLE)
    - Assume that training data has been given $\{(O_1,I_1),(O_2,I_2),...,(O_T,I_T)\}$
    - Use maximum likelihood estimation to estimate HMM parameters
      - estimation of transition matrix
      - estiamtion of observation matrix
      - estimation of initial value
    - For Text Sequence Labeling


- Solution B：Unsuperised learning methods (Unknown label is a hidden variable， EM)
    - Baum-Welch algorithm

      - Also known as expectation maximization algorithm (EM)

    - For Speech Recognition


### Problem Three：Inference

+ Define
  + 给定O和$$\lambda$$， 求I

- Solution A ：Approximation algorithm
    - The idea of this algorithm is Select the most likely state $i^{*}_{t}$ at each time t
        + Given O and $\lambda$, then the probability of being in state $q_i$ at time t is
          + $$\gamma_{t}(i) = \frac{\alpha_t(i) \beta_t(i)}{P(O|\lambda)} = \frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^{N}\alpha_t(j) \beta_t(j)}$$
        + The most likely state $i^{*}_{t}$ is
          + $$ i^{*}_{t} = arg\ \underset{1\leq i \leq N}{max} [\gamma_t(i)], t = 1,2,...,T$$

- Solution B：Viterbi algorithm
    - Overview
        - The viterbi algorithm uses dynamic programming to solve the HMM inference problem
        - The path with the greatest probability, which corresponds to a sequence of states
        - 根据这一原理，从t=1时刻开始，递推的计算在时刻状态为i的各条路径的最大概率，直到时刻t=T状态为i的各条路径的最大概率
        - 时刻t=T的最大概率即最有路径的概率$p^*$，最优路径的终止节点为${i^*}_T$ 
        - 从终止节点开始由后向前得到各个节点，进而得到最优路径

    - Detail
        - Define variable $$\delta$$
            - The time is t, the state is i, the maximum value of the path (path consisit of $i_1,...,i_{t_1}$), variable is $i_1,...,i_{t_1}$
              $$\delta_t(i) = \underset{i_1,i_2,...,i_{t-1}}{max} P(i_t=i,i_{t-1}...,i_1,o_t,...,o_1|\lambda) $$
              $$ \delta_{t+1}(i) = \underset{1\leq j \leq N}{max} [\delta_t(i) a_{ji}]b_i(o_{t+1})$$

        - Define variable $$\varphi $$
            - The time is t, the state is i, maximum probability path is i_1,...,i_{t_1}
            - the value fo state t-1 is as follow, the range of values is $1\leq j \leq N$
            - $$  \varphi_t(i) = arg \underset{1\leq j \leq N}{max} [\delta_{t-1}(j) a_{ji}] $$

    - Summary of viterbi
        - input : $\lambda = (A, B, \pi) $ and O
        - output : optimal path

          + initial

            $$\delta_1(i) = \pi_i b_i(o_1)$$

            $$\varphi_1(i) = 0$$

          + Recursive

            $$ \delta_{t+1}(i) = \underset{1\leq j \leq N}{max} [\delta_t(i) a_{ji}]b_i(o_{t+1})$$

            $$  \varphi_t(i) = arg \underset{1\leq j \leq N}{max} [\delta_{t-1}(j) a_{ji}] $$

          + Termination

            $$P^{\star} = \underset{1\leq j \leq N}{max} \delta_T(i)$$
            $$ i_T^{\star} = arg \underset{1\leq j \leq N}{max} [\delta_T(i)]$$



### Demo of HMM

- How to do NER?

  - given x, find y
  - $ y = arg\ max_{y \in Y}^{} p(y|x) = arg\ max_{y \in Y}^{} \frac{p(x, y)}{p(x)} = arg\ max_{y \in Y}^{} p(x, y)$

- 文本长度为7{张，三，在， 北， 京， 朝， 阳}

  - Observation State Set:
    -  {张，三，在， 北， 京， 朝， 阳}

- 5种标签{O, B-PER, I-PER, B-LOC, I-LOC}

  - Hidden State Set:

    - {O, B-PER，I-PER，B-LOC, I-LOC}

  - 第一个隐状态取得标签的可能是由$\pi$ 决定：

    | First Hidden State | 概率值 |
    | ------------------ | ------ |
    | O                  | 0.4    |
    | B-PER              | 0.3    |
    | I-PER              | 0      |
    | B-LOC              | 0.3    |
    | I-LOC              | 0      |

  - Current Hidden State 转移到 Next Hidden State， 由转移矩阵A决定：

    | Current Hidden State/Next Hidden State | O    | B-PER | I-PER | B-LOC | I-LOC | <END> |
    | -------------------------------------- | ---- | ----- | ----- | ----- | ----- | ----- |
    | <BEG>                                  | 0.8  |       |       |       |       |       |
    | O                                      |      | 0.5   |       |       |       |       |
    | B-PER                                  |      |       | 0.5   |       |       |       |
    | I-PER                                  |      |       |       | 0.8   |       |       |
    | B-LOC                                  |      |       |       |       | 0.5   |       |
    | I-LOC                                  |      |       |       |       |       | 0.8   |

  - Current Hidden State 输出到 Current Observation State,  由状态矩阵B决定：

    | Current Hidden State/Current Observation State | 我   | 在   | 北   | 京   | 朝   | 阳   |
    | ---------------------------------------------- | ---- | ---- | ---- | ---- | ---- | ---- |
    | O                                              |      |      |      |      |      |      |
    | B-PER                                          |      |      |      |      |      |      |
    | I-PER                                          |      |      |      |      |      |      |
    | B-LOC                                          |      |      |      |      |      |      |
    | I-LOC                                          |      |      |      |      |      |      |



### Drawbacks of HMM

- 假设正确结果为  (x, $\hat y$) , 不能保证 P(x,$\hat y$) > P(x,y)
- 例子
    - P(我|O)=9/10, P(北|O)=1/10
    - P(北|B)=1/2, P(a|B)= 1
    - O -> B --> c 9
    - P -> V --> a 9
    - N -> D --> a 1
    - request N -> ? --> a
    - use HMM will get N -> V --> a (Highest probability of HMM), not contain in traing data
- Will generate some sequence never happened in training data
    - transition matrix and emission matrix are trained separatly
- The (x,y) never seen in the training data can have large probability P(x,y)
- More complex model can deal with this problem
- However, CRF can deal with this problem based on the same model

### Benefit of HMM
- Suitable for training data is small 