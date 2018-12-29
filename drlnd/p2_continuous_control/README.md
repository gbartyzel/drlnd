# Project 2: Continuous control

### Environment

In this environment, the agent's task i to move to target loction. A double-jointed arm is the 
learning agent. A reward of +0.1 is provided for each step when robot reached the goal location. 
So, the goal of agent is simple, stay as long as it possible in goal location.

The observation space has 33 dimensions and contains the agent's position, rotation, velocity and
 angular velocities of the arm.
 
Agent can perform four continuous actions which corresponds to applied torques to joints.

This environment comes into two variants:
* with single agent
* with 20 identical agents.

The task is episodic and in order to solve it, agent must get an average score of +30 over 100 
consecutive episodes. For distributed variants average score of +30 must be achieved over all 
learning agents.

### Getting started

#### Installation

1. Install Unity ML-agents (version 0.4) by following 
[instruction](https://github.com/Unity-Technologies/ml-agents/tree/0.4.0b).
2. Copy Reacher environment from Unity ML-agent directory to ```./drlnd/p1_navigation/env```. 
Code is compatible with single Reacher environment and multi Reacher environment.
3. Run ```pip install -r requirements.txt``` to make sure that all required python packages are 
installed.
4. (Optional) Add repository to PYTHONPATH: ```export PYTHONPATH="${PYTHONPATH}:/path/to/drlnd"```

#### Run agent

For perform learning procedure of single agent just run following command in terminal:

```python main.py --train --logdir="output/single"```

Distributed learning could be run by:

```python main.py --train --use_distributed --logdir="output/distributed"```

More options could be found after running:

```python main.py --help```

To evaluate learned policy run for example following command:

```python main.py --train --use_distributed --logdir="output/distributed"```

### Report