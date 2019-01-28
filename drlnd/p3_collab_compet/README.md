# Project 3: Collaboration and competition

### Environment

In this environment, there are two agents whose task is to learn playing tennis. However, their 
goal is to learn collaborate with each other and achieve as many points as it is possible. Agent 
received +0.1 score for successful hitting the ball, -0.01 for letting ball hit the 
ground or hitting ball out of bounds. Environment is consider as solved when agents received average 
score of 0.5 over 100 consecutive episodes (episode score is equal to maximum of agents' scores).

The observation space of each agents is stack of three vectors corresponding to position and 
velocity of ball and racket in current step and two previous. Each agent can perform two 
continuous actions, move toward the net, and jumping.

### Getting started

#### Installation

1. Install Unity ML-agents (version 0.4) by following 
[instruction](https://github.com/Unity-Technologies/ml-agents/tree/0.4.0b).
2. Copy Tennis environment from Unity ML-agent directory to ```./drlnd/p3_collab_compet/env```. 
Code is compatible with single Reacher environment and multi Reacher environment.
3. Run ```pip install -r requirements.txt``` to make sure that all required python packages are 
installed.
4. (Optional) Add repository to PYTHONPATH: ```export PYTHONPATH="${PYTHONPATH}:/path/to/drlnd"```

#### Run agent

For perform learning procedure of agent just run following command in terminal:

```python main.py --train ```

More options could be found after running:

```python main.py --help```

To evaluate learned policy run for example following command:

```python main.py"```

### Report

Description of used architecture and learning process can be found in [report](Report.md)

