# RL
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

