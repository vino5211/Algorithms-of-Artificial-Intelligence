# Hidden Markov Model
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