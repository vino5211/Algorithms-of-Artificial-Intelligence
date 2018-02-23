# Strcture
## Model-free Approach
+ Policy-based
	+ Learning an Actor（学习动作来执行）
	+ on-policy : the agent learned and the agent interacting with the environment is the same(自己打篮球)
 	+ off-policy : the agent learned and the agent interacting with the environment is the different（看别人打篮球）
+ Value-based
	+ Learning a critic(对现在的状况给予评价)
+ Policy-based && Value-based
	+ Actor + Critic
	+ 
+ DQN + A3C
+ A3C
	+ Asynchronous(异步) Advantage Actor-Critic
	+ Actor is a Neural Network
		+ Input of neural network : the observation of machine represented as a vector or a matrix
		+ Output of neural network : each action corresponds to a neuron in output layer
		+ Actor can also have continuous action
	+ Actor - Goodness of an Actor
		+ Given an **actor** $\pi(s)$ with network parameter $\theta^\pi$
		+ Use the actor $\pi(s)$ to play the video game
			+ Start with observation $s_1$
			+ Machine decides to take $a_1$
			+ Machine obtains reward $r_1$
			+ Start with observation $s_2$
			+ Machine decides to take $a_2$
			+ Machine obtains reward $r_2$
			+ ...
			+ Start with observation $s_T$
			+ Machine decides to take $a_T$
			+ Machine obtains reward $r_T$
			+ Total reward : R = $\sum^{T}_{t=1} r_t$
		+ Even with the same actor, R is different each time
			+ Radnomness in the actor and the game
		+ Difine $\bar{R}_{\theta^\pi}$ expected total reward(将Actor和环境互动无数次之后的结果)
		+ $\bar{R}_{\theta^\pi}$ evalutes the goodness of an actor $\pi(s)$
	+ Actor - Policy Gradient
		+ $ \theta^{\pi'}$ = $ \theta^{\pi}$ + $\eta \triangledown \bar{R}_{\theta^\pi}$
		+ using $\theta^\pi$ to obtian { ${\tau^1,\tau^2,...,\tau^N}$ } , $\tau$ is a other format fo $\theta$
		+ if in $\tau^n$ machine takes $a^n_t$ when seeing $s^n_t$
			+ if $R(\tau^n)$ is positive, tuning $\theta$ to increase $p(a^n_t|s^n_t)$
			+ if $R(\tau^n)$ is nagative, tuning $\theta$ to decrease $p(a^n_t|s^n_t)$
		+ It is very important of consider the cumulative(累积的) reward $R(\tau^n)$ of the whole trajectory(轨迹) $\tau^n$ instead of immediate reward $r^n_t$
	+ Critic
		+ A critic does not determine the action
		+ **Given an actor $\pi$, it evaluates the how good the actor is**
		+ Statte value function $V^{\pi}(s)$
			+ when using actor $\pi$, the cumulated reward expects to be obtained after seeding observation (state) s
		+ How to estimate $V^{\pi}(s)$
			+ Mante-Carlo based approach
				+ The critic watches $\pi$ playing the game
			+ Temporal-dfference approach
				+ $V^{\pi}(s_a)$ + $r_t$ = $V^{\pi}(s_b)$
				+ Some applications have very long episodes, so htat delaying all learning until an episode's end is too low
			+ MC v.s. TD



## Model-based Approach
+ Less used



---

---
## model-free RL
### Policy Gradient
	- - PG属于policy optimization，其目的是优化expected reward，并不关心Q*(s, a).
### Q-Learning
	- QL属于off-policy model-free control，其目的是求出Q*(s, a)。
	- DQN
		- https://www.qcloud.com/community/article/549802?fromSource=gwzcw.114127.114127.114127

### - Actor-critic
	- Actor-critic 是两者的结合，目标是policy optimization (类比为actor)，但是引入了对Q(s, a)的估计 (类比为critic)，使得求PG时的梯度估计方差降低。但是很显然，由于critic的估计并不一定准确，可能会带来bias - 这是典型的bias-variance trade-off

## model-based RL

## Diff


# tmp
https://www.zhihu.com/question/49787932
## ViewPoint One
	- 1) 减小unbiased estimate的variance有很多解决方法，例如baseline, 例如reparameterization trick，在VAE和RL中都很常用到。
	- 2) 感觉David Silver的RL课的slides将这些方法的区别讲得非常清楚。传送门：http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
	- 3) 感觉QL和PG之间的关系，就好像Bayesian ML和Non-Bayesian ML之间的关系，前者描述更多uncertainty，后者优化得更直接。QL的一个弱点在于如果s, a都是非常高维度的，那么即使是从Q*(s, a)得到一个sample都变得非常难，PG就没有这个问题，如果episodes很多（数据很多），数据维度很大，PG就更有优势。
	- 4) 谁来拯救一下model-based RL啊
## ViewPoint Two
	- 1.ql和pg都是为了求解最好的RL决策链
	- 2.ql一般针对离散空间，采用值迭代方法。以value推policy
	- 3.pg针对连续场景，直接在策略空间求解，泛化更好，直推policy
	- 4.actor-critic可以看作是一个共轭，互相作用，策略也更稳定
## ViewPoint Three
	- 我就先针对连续空间说两句。
	- 1. 基于PG的SVG、DPG、GAE用的都是actor-critic的方法并且可以用于解决连续action空间的问题 
	- 2. Q-learning是off-policy，本身有个Max值，在无限action下是不合理的 
	- 3. RDPG里面有谈到Normalized Advantage Function，把一个policy转换成advantage function，本身也好像是actor-critic，但是actor和critic共用一个网络。




---

- http://www.algorithmdog.com/series/rl-series
- 一、马尔可夫决策过程(Markov Decision Processes, MDP)
- 组成
  - S 表示状态集 (states)；
  - A 表示动作集 (Action)；
  - P_{s,a}^{s'} 表示状态 s 下采取动作 a 之后转移到 s’ 状态的概率；
  - R_{s,a} 表示状态 s 下采取动作 a 获得的奖励；
  - $\gamma$ 是衰减因子。
- 和一般的马尔科夫过程不同，马尔科夫决策过程考虑了动作，即系统下个状态不仅和当前的状态有关，也和当前采取的动作有关
- 还是举下棋的例子，当我们在某个局面（状态s）走了一步 (动作 a )。这时对手的选择（导致下个状态 s’ ）我们是不能确定的，但是他的选择只和 s 和 a 有关，而不用考虑更早之前的状态和动作，即 s’ 是根据 s 和 a 随机生成的
- 分类依据
  - 知道马尔科夫决策过程所有信息(状态集合，动作集合，转移概率和奖励)
  - 知道部分信息 (状态集合和动作集合)
  - 还有些时候马尔科夫决策过程的信息太大无法全部存储 (比如围棋的状态集合总数为3^{19\times 19})。
- 强化学习算法按照上述不同情况可以分为两种
  - 基于模型 (Model-based)  ： 知道并可以存储所有马尔科夫决策过程信息
  - 非基于模型 (Model-free) ： 需要自己探索未知的马尔科夫过程
- 二、Demo
  -   下图是一个机器人从任意一个状态出发寻找金币的例子。找到金币则获得奖励 1，碰到海盗则损失 1。找到金币或者碰到海盗则机器人停止。
    
  - 问题建模成马尔科夫决策过程：
    - 图中不同位置为状态，因此 S = {1,…,8}
    - 机器人采取动作是向东南西北四个方向走，因此A={‘n’,’e’,’s’,’w’}
    - 转移概率方面：
      - 当机器人碰到墙壁，则会停在原来的位置
      - 当机器人找到金币时获得奖励 1，当碰到海盗则损失 1, 其他情况不奖励也不惩罚
    - 除了 R{}_{1,s} = -1 , R_{2,s}=1,R_{5,s}=-1 外，其他情况R_{*,*}=0
- 三、策略和价值
  -  强化学习技术是要学习一个策略 (Policy)，即一个函数：输入为当前状态s，输出为采用动作a的概率\pi(s,a)
  - 最佳策略
    	E_{\pi}[{\sum^{\infty}_{k=0} } \gamma^{k}R_{k}] = E_{\pi}[R_{0}+{\gamma}R_{1}+...]
    - 其中的$$










