# Outline of Reinforce Learning

### Model-free Approach
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
