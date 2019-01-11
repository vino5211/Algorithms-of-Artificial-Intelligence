# View of Reinforce Learning

+ 强化学习大讲堂
	+ https://zhuanlan.zhihu.com/sharerl

- https://web.stanford.edu/~pkurlat/teaching/5%20-%20The%20Bellman%20Equation.pdf

- 业界 | 在个人电脑上快速训练Atari深度学习模型：Uber开源「深度神经进化」加速版
  - Uber 在去年底发表的研究中发现，通过使用遗传算法高效演化 DNN，可以训练含有超过 400 万参数的深度卷积网络在像素级别上玩 Atari 游戏；这种方式在许多游戏中比现代深度强化学习算法或进化策略表现得更好，同时由于更好的并行化能达到更快的速度。
  - 不过这种方法虽好但当时对于硬件的要求很高，近日 Uber 新的开源项目解决了这一问题，其代码可以让一台普通计算机在 4 个小时内训练好用于 Atari 游戏的深度学习模型。现在，技术爱好者们也可以接触这一前沿研究领域了
  - 项目 GitHub 地址：https://github.com/uber-common/deep-neuroevolution/tree/master/gpu_implementation
  - 参考Uber五篇论文：前沿 | 利用遗传算法优化神经网络：Uber提出深度学习训练新方式
- Chess-alpha-zero（1014 stars on Github，来自Samuel）
    - 通过AlphaGo Zero方法进行国际象棋的强化学习
    - 项目地址：
    - https://github.com/Zeta36/chess-alpha-zero
+ Reference Websites:
	+ https://www.zhihu.com/question/41477987
	+ http://www.cnblogs.com/jerrylead/archive/2011/05/13/2045309.html

+ SARSA
	+ a
+ DQN
	+ https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/4-06-DQN/
		+ 简单来说, DQN 有一个记忆库用于学习之前的经历. 在之前的简介影片中提到过, Q learning 是一种 off-policy 离线学习法, 它能学习当前经历着的, 也能学习过去经历过的, 甚至是学习别人的经历. 所以每次 DQN 更新的时候, 我们都可以随机抽取一些之前的经历进行学习. 随机抽取这种做法打乱了经历之间的相关性, 也使得神经网络更新更有效率. 
		+ Fixed Q-targets 也是一种打乱相关性的机理, 如果使用 fixed Q-targets, 我们就会在 DQN 中使用到两个结构相同但参数不同的神经网络, 预测 Q 估计 的神经网络具备最新的参数, 而预测 Q 现实 的神经网络使用的参数则是很久以前的. 有了这两种提升手段, DQN 才能在一些游戏中超越人类	
	+ Playing Atari with Deep Reinforcement Learning
+ Actor-Critic算法小结
	+ https://zhuanlan.zhihu.com/p/29486661
	+ 谷歌 dppo:
		+ Heess N, Dhruva T B, Sriram S, et al. Emergence of Locomotion Behaviours in Rich Environments[J]. 2017.
	+ openai ppo:
		+ Schulman J, Wolski F, Dhariwal P, et al. Proximal Policy Optimization Algorithms[J]. 2017.
	+ 最新ACKTR:
		+ Wu Y, Mansimov E, Liao S, et al. Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation[J]. 2017.

---

## Un Clf
- http://geek.ai/

## papers
- Gorila DeepMind 2015

## Demo
- 《A Tour of Gotchas When Implementing Deep Q Networks with Keras and OpenAi Gym》by Scott Rome O网页链接 
- 【层次增强学习算法】《Learning a Hierarchy | OpenAI》 O网页链接 ref:《Meta Learning Shared Hierarchies》(2017) GitHub: https ://github .com/openai/mlsh
- Q-Learning月球着陆控制“QLearning in OpenAI Lunar Lander” 
	- GitHub: https:\//github.com\/FitMachineLearning/FitML/ 
- 强化学习入门及其实现代码
	- http://www.jianshu.com/p/165607eaa4f9
- Awesome Reinforcement Learning
	- https://github.com/aikorea/awesome-rl
	- lots of resourses
- http://www.jianshu.com/p/165607eaa4f9
	- http://www.jianshu.com/p/165607eaa4f9
- keras-rl
	- 111

## Math
- DP
	- 【转载】近似动态规划与强化学习入门步骤
		- http://www.cnblogs.com/stevenbush/articles/3353227.html
## OpenAI
- Keras+OpenAI增强学习实践：Actor-Critic模型
	《Reinforcement Learning w/ Keras + OpenAI: Actor-Critic Models》by Yash Patel O网页链接 pdf:O网页链接 
- 《Keras+OpenAI强化学习实践：深度Q网络》via:机器之心 O教程 | Keras+OpenAI强化学习实践：深度Q网络
- OpenAI Gym 扩展集
	- 这是一个 OpenAI Gym 库的扩展包，实现了包括：多任务学习、迁移学习、逆增强学习等功能。
	- 项目链接：https://github.com/Breakend/gym-extensions

## Game Demo
- PyTorch Implementation of Deep Q-Learning with Experience Replay in Atari Game Environments, as made public by Google DeepMind
	- https://github.com/diegoalejogm/deep-q-learning
- 重现DeepMind星际争霸强化学习算法
	- https://zhuanlan.zhihu.com/p/29246185
- 迈向通用人工智能：星际争霸2人工智能研究环境SC2LE完全入门指南
	- https://zhuanlan.zhihu.com/p/28434323
- 用 AlphaZero 下五子棋
	- 将 AlphaZero 算法用于五子棋，相比围棋或国际象棋简单得多，所以只需几个小时就可以训练出一个不错的 AI 模型。
	- 论文 | AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
	- 链接 | https://www.paperweekly.site/papers/1297
	- 论文 | AlphaGo Zero: Mastering the game of Go without human knowledge
	- 链接 | https://www.paperweekly.site/papers/942
	- 项目链接：https://github.com/junxiaosong/AlphaZeroGomoku
- Arnold
	- 最擅长玩《毁灭战士》的游戏AI
	- 本项目来自卡耐基梅隆大学，是 2017 年 VizDoom《毁灭战士》AI 死亡竞赛冠军 Arnold 的 PyTorch 开源代码。
	- 论文链接：https://www.paperweekly.site/papers/1440
	- 项目链接：https://github.com/glample/Arnold
## Course
- 强化学习入门 第一讲 MDP
	- https://zhuanlan.zhihu.com/p/25498081
- 如2015年David Silver的经典课程Teaching
- 2017年加州大学伯克利分校Levine, Finn, Schulman的课程 CS 294 Deep Reinforcement Learning, Spring 2017 
- 卡内基梅隆大学的2017 春季课程Deep RL and Control
- 【Xi(Peter) Chen主讲的深度增强学习课程(Videos/Slides)】“Deep Reinforcement Learning” by Xi(Peter) Chen OpenAI/Berkeley AI Research Lab(http://t.cn/RHWQreD)

## Mate Learning
- https://zhuanlan.zhihu.com/p/32270990

# View-RL

+ Reference Websites:
	+ https://www.zhihu.com/question/41477987
	+ http://www.cnblogs.com/jerrylead/archive/2011/05/13/2045309.html

+ SARSA
	+ a
+ DQN
	+ https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/4-06-DQN/
		+ 简单来说, DQN 有一个记忆库用于学习之前的经历. 在之前的简介影片中提到过, Q learning 是一种 off-policy 离线学习法, 它能学习当前经历着的, 也能学习过去经历过的, 甚至是学习别人的经历. 所以每次 DQN 更新的时候, 我们都可以随机抽取一些之前的经历进行学习. 随机抽取这种做法打乱了经历之间的相关性, 也使得神经网络更新更有效率. 
		+ Fixed Q-targets 也是一种打乱相关性的机理, 如果使用 fixed Q-targets, 我们就会在 DQN 中使用到两个结构相同但参数不同的神经网络, 预测 Q 估计 的神经网络具备最新的参数, 而预测 Q 现实 的神经网络使用的参数则是很久以前的. 有了这两种提升手段, DQN 才能在一些游戏中超越人类	
	+ Playing Atari with Deep Reinforcement Learning
+ Actor-Critic算法小结
	+ https://zhuanlan.zhihu.com/p/29486661
	+ 谷歌 dppo:
		+ Heess N, Dhruva T B, Sriram S, et al. Emergence of Locomotion Behaviours in Rich Environments[J]. 2017.
	+ openai ppo:
		+ Schulman J, Wolski F, Dhariwal P, et al. Proximal Policy Optimization Algorithms[J]. 2017.
	+ 最新ACKTR:
		+ Wu Y, Mansimov E, Liao S, et al. Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation[J]. 2017.

---

## Un Clf
- http://geek.ai/

## papers
- Gorila DeepMind 2015

## Demo
- 《A Tour of Gotchas When Implementing Deep Q Networks with Keras and OpenAi Gym》by Scott Rome O网页链接 
- 【层次增强学习算法】《Learning a Hierarchy | OpenAI》 O网页链接 ref:《Meta Learning Shared Hierarchies》(2017) GitHub: https ://github .com/openai/mlsh
- Q-Learning月球着陆控制“QLearning in OpenAI Lunar Lander” 
	- GitHub: https:\//github.com\/FitMachineLearning/FitML/ 
- 强化学习入门及其实现代码
	- http://www.jianshu.com/p/165607eaa4f9
- Awesome Reinforcement Learning
	- https://github.com/aikorea/awesome-rl
	- lots of resourses
- http://www.jianshu.com/p/165607eaa4f9
	- http://www.jianshu.com/p/165607eaa4f9
- keras-rl
	- 111

## Math
- DP
	- 【转载】近似动态规划与强化学习入门步骤
		- http://www.cnblogs.com/stevenbush/articles/3353227.html
## OpenAI
- Keras+OpenAI增强学习实践：Actor-Critic模型
	《Reinforcement Learning w/ Keras + OpenAI: Actor-Critic Models》by Yash Patel O网页链接 pdf:O网页链接 
- 《Keras+OpenAI强化学习实践：深度Q网络》via:机器之心 O教程 | Keras+OpenAI强化学习实践：深度Q网络
- OpenAI Gym 扩展集
	- 这是一个 OpenAI Gym 库的扩展包，实现了包括：多任务学习、迁移学习、逆增强学习等功能。
	- 项目链接：https://github.com/Breakend/gym-extensions

## Game Demo
- PyTorch Implementation of Deep Q-Learning with Experience Replay in Atari Game Environments, as made public by Google DeepMind
	- https://github.com/diegoalejogm/deep-q-learning
- 重现DeepMind星际争霸强化学习算法
	- https://zhuanlan.zhihu.com/p/29246185
- 迈向通用人工智能：星际争霸2人工智能研究环境SC2LE完全入门指南
	- https://zhuanlan.zhihu.com/p/28434323
- 用 AlphaZero 下五子棋
	- 将 AlphaZero 算法用于五子棋，相比围棋或国际象棋简单得多，所以只需几个小时就可以训练出一个不错的 AI 模型。
	- 论文 | AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
	- 链接 | https://www.paperweekly.site/papers/1297
	- 论文 | AlphaGo Zero: Mastering the game of Go without human knowledge
	- 链接 | https://www.paperweekly.site/papers/942
	- 项目链接：https://github.com/junxiaosong/AlphaZeroGomoku
- Arnold
	- 最擅长玩《毁灭战士》的游戏AI
	- 本项目来自卡耐基梅隆大学，是 2017 年 VizDoom《毁灭战士》AI 死亡竞赛冠军 Arnold 的 PyTorch 开源代码。
	- 论文链接：https://www.paperweekly.site/papers/1440
	- 项目链接：https://github.com/glample/Arnold
## Course
- 强化学习入门 第一讲 MDP
	- https://zhuanlan.zhihu.com/p/25498081
- 如2015年David Silver的经典课程Teaching
- 2017年加州大学伯克利分校Levine, Finn, Schulman的课程 CS 294 Deep Reinforcement Learning, Spring 2017 
- 卡内基梅隆大学的2017 春季课程Deep RL and Control
- 【Xi(Peter) Chen主讲的深度增强学习课程(Videos/Slides)】“Deep Reinforcement Learning” by Xi(Peter) Chen OpenAI/Berkeley AI Research Lab(http://t.cn/RHWQreD)

## Mate Learning
- https://zhuanlan.zhihu.com/p/32270990


# 1.Structue
## Model free
### Policy Gradient
- PG属于policy optimization，其目的是优化expected reward，并不关心Q*(s, a).

### Q-Learning
- QL属于off-policy model-free control，其目的是求出Q*(s, a)。
- DQN
	- https://www.qcloud.com/community/article/549802?fromSource=gwzcw.114127.114127.114127

### Actor-critic
Actor-critic 是两者的结合，目标是policy optimization (类比为actor)，但是引入了对Q(s, a)的估计 (类比为critic)，使得求PG时的梯度估计方差降低。但是很显然，由于critic的估计并不一定准确，可能会带来bias - 这是典型的bias-variance trade-off

## Diff
### Reference websites:
+ https://www.zhihu.com/question/49787932

### ViewPoint One
- 减小unbiased estimate的variance有很多解决方法，例如baseline, 例如reparameterization trick，在VAE和RL中都很常用到。
- 感觉David Silver的RL课的slides将这些方法的区别讲得非常清楚。传送门：http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
- 感觉QL和PG之间的关系，就好像Bayesian ML和Non-Bayesian ML之间的关系，前者描述更多uncertainty，后者优化得更直接。QL的一个弱点在于如果s, a都是非常高维度的，那么即使是从Q*(s, a)得到一个sample都变得非常难，PG就没有这个问题，如果episodes很多（数据很多），数据维度很大，PG就更有优势。

### ViewPoint Two
- ql和pg都是为了求解最好的RL决策链
- ql一般针对离散空间，采用值迭代方法。以value推policy
- pg针对连续场景，直接在策略空间求解，泛化更好，直推policy
- actor-critic可以看作是一个共轭，互相作用，策略也更稳定

### ViewPoint Three
- 基于PG的SVG、DPG、GAE用的都是actor-critic的方法并且可以用于解决连续action空间的问题 
- Q-learning是off-policy，本身有个Max值，在无限action下是不合理的 
- RDPG里面有谈到Normalized Advantage Function，把一个policy转换成advantage function，本身也好像是actor-critic，但是actor和critic共用一个网络。


# 2.LHY Video
## Model-free Approach
+ Policy-based
	+ Learning an Actor（学习动作来执行）
	+ on-policy : the agent learned and the agent interacting with the environment is the same(自己打篮球)
 	+ off-policy : the agent learned and the agent interacting with the environment is the different（看别人打篮球）
+ Value-based
	+ Learning a critic(对现在的状况给予评价)
+ A3C : Policy-based && Value-based
	+ Actor + Critic
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
	+ Pathwise derivative policy gradient
		+ 不仅仅评价当前状态的好坏，而且会指出下一步该如何做
		+ Another Critic(Q-Learning )
			+ state-action value function $Q^\pi(s,a)$
				+ when using actor $\pi$, the cumulated reward expects to be obtained after seeing observation s and taking a
				+ $Q^\pi(s,a)$ 与 $V^\pi(s)$ 不同的的是，V 仅仅考虑s(state), Q 不经考虑s(state)而且考虑a(action)
			+ 不适用于连续的情况
			+ Estimate $Q^\pi(s,a)$ by TD
	+ Dueling DQN
	+ DDPG

---

---
+ Reference Websites:
	+ https://www.zhihu.com/question/41477987
	+ http://www.cnblogs.com/jerrylead/archive/2011/05/13/2045309.html

+ SARSA
	+ a
+ DQN
	+ https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/4-06-DQN/
		+ 简单来说, DQN 有一个记忆库用于学习之前的经历. 在之前的简介影片中提到过, Q learning 是一种 off-policy 离线学习法, 它能学习当前经历着的, 也能学习过去经历过的, 甚至是学习别人的经历. 所以每次 DQN 更新的时候, 我们都可以随机抽取一些之前的经历进行学习. 随机抽取这种做法打乱了经历之间的相关性, 也使得神经网络更新更有效率. 
		+ Fixed Q-targets 也是一种打乱相关性的机理, 如果使用 fixed Q-targets, 我们就会在 DQN 中使用到两个结构相同但参数不同的神经网络, 预测 Q 估计 的神经网络具备最新的参数, 而预测 Q 现实 的神经网络使用的参数则是很久以前的. 有了这两种提升手段, DQN 才能在一些游戏中超越人类	
	+ Playing Atari with Deep Reinforcement Learning
+ Actor-Critic算法小结
	+ https://zhuanlan.zhihu.com/p/29486661
	+ 谷歌 dppo:
		+ Heess N, Dhruva T B, Sriram S, et al. Emergence of Locomotion Behaviours in Rich Environments[J]. 2017.
	+ openai ppo:
		+ Schulman J, Wolski F, Dhariwal P, et al. Proximal Policy Optimization Algorithms[J]. 2017.
	+ 最新ACKTR:
		+ Wu Y, Mansimov E, Liao S, et al. Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation[J]. 2017.









