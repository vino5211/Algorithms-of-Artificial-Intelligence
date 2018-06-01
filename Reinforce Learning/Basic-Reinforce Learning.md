
# Basic Reinforce Learning

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








