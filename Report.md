# Report

## Learning Algorithm

To solve Continuous Control project, I used as a start point the code from the Udacity deep reinforcement learning repository that implements Deep Deterministic Policy Gradients (DDPG) algorithm.
[https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) 
DDPG is using the actor-critic architecture, Actor is used to decide the best action for a specific state and the Critic is used for evaluating action. DDPG is also using target network with soft update for both Actor and Critic and experience replay buffer.
Ornstein-Uhlenbeck random process is used for adding noise to action so the agent has more exploration of action space. Parameters of Ornstein-Uhlenbeck process are

```Python
seed = 2
mu = 0.0
theta = 0.25
sigma = 0.3
```
Original code is in first step customized to use *Unity ML-Agents* environment. In this project I am using Reacher environment with 20 agents. The main part of the program is in following files

```Python
ddpg_agent.py
model.py
Continuous_Control.ipynb
```

*ddpg_agent.py* code implements an environment-aware agent, while in *model.py* is a neural network models of Actor and Critic.
*Continuous_Control.ipynb* contains a code that trains a agent and displays results.

#### DDPG  algorithm with Actor - Critic networks
The model of the neural network used is shown in the picture:

![Network model](./images/model.png  "Network model")

I used a three layers network for the Actor

The size of the hidden layers are:
FC1 size = 256
FC2 size = 256
FC3 size = 4

The input parameter is the state of the environment (size 33) and the output is action (size 4).

For the Critic, I also use a three layers network
FCS1 size = 256
FC2 size + action_size= 256 + 4
FC3 size = 1

Input parameters are state in first layer and action in second layer and output is Q  value.

I used batch normalization on the state input and all layers of the Actor network and all layers of the Critic network prior to the action input as in paper https://arxiv.org/pdf/1509.02971.pdf.

The activation function is leaky_relu on output of all hidden layers except on the last layer of Actor where I used tanh (because the action values are in range [-1.1]), and the last layer of Critic where I did not use any activation function because the output was Q value.

The hyper parameters are: 

```Python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE  = 128       # minibatch size
GAMMA = 0.95            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
WARMUP_TIME = 10000     # time for collecting initial experiences
PLAY_TIME = 40          # timesteps for collecting experiences from agents
WORK_TIME = 10          # timesteps for learning
PLAY_TIME_DECAY = 0.999 # play time discount factor
PLAY_TIME_MIN = 15      # play time minimum value
```

The hyperparameters I introduced are WARMUP_TIME which represents the initial experience buffer fill time.
PLAY_TIME determines how long an agent will collect experience samples before it starts to learn, WORK_TIME represents the number of learning cycles when the agent starts to learn. In other words, the agent collects samples PLAY_TIME time steps and then learn WORK_TIME time steps, and then repeat cycle. Changing PLAY_TIME/WORK_TIME hyperaparameters affects the relationship exploration/exploitation.
PLAY_TIME_DECAY represents a factor that discounts PLAY_TIME similar to GAMMA. 
PLAY_TIME_MIN is minimum value PLAY_TIME  can take.

After lowering value of the hyper parameter GAMMA, I got a significant improvement in the results and by changing LR_ACTOR to 1e-3 I found that agent is learning faster.
Other hyper parameters have the same values as in the original project.

The code in *ddpg_agent.py* has been changed to support the input of 20 agents. Main idea in these code changes to keep model with one Actor and one Critic and to fill experience buffer with experiences of  20 agents to speed up exploration of state and action space.
The experiences in the buffer are added randomly and uniformly so the probability that the experience of each of the agents will be added to the buffer is 50%, in this way we make the buffer more random and brake the correlations even more (something like dropout layer).
I also used clipping gradients of Actor and Critic.

Results after 300 episodes are:

	Episode 100	Average Score: 7.37	 Score: 20.35
	Episode 200	Average Score: 26.30	 Score: 29.45
	Episode 262	Average Score: 30.01	 Score: 36.86
	Environment solved in 262 episodes!	Average Score: 30.01
	Episode 300	Average Score: 30.96	 Score: 27.33

![Plot of rewards](./images/score_graph.png  "Plot of rewards")
In this case agent solves the problem after 262 episodes. 

## Ideas for Future Work 	
In the next step I planed to try multi Critic model, in which every Critic learn from different experience samples and then  taking average of their Q values as output. Implementation of the Prioritized Experience Replay algorithm [https://arxiv.org/pdf/1511.05952](https://arxiv.org/pdf/1511.05952), I think will improve results also.

## References

1. [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971)
2. [https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287) 