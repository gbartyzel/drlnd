# Project 1: Navigation

### Environment

In this environment, the agent's task is to learn navigate and also collect bananas.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for 
collecting a blue banana. So, the goal of the agent is to maximize reward by collecting as 
many yellow bananas as possible and also avoiding blue ones.

The state space has 37 dimensions and contains the agent's velocity, ray-based 
perception of objects around agent's forward direction. 

Agent can choose four discrete action: 
* 0 - move forward,
* 1 - move backward,
* 2 - turn left,
* 3 - turn right,

The task is episodic, and in order to solve it, agent must get an average score of +13 over 
100 consecutive episodes.

### Getting started

#### Installation

1. Install Unity ML-agents (version 0.4) by following 
[instruction](https://github.com/Unity-Technologies/ml-agents/tree/0.4.0b).
2. Copy Banana environment from Unity ML-agent directory to ```./drlnd/p1_navigation/env```.
3. Run ```pip install -r requirements.txt``` to make sure that all required python packages are 
installed.
4. (Optional) Add repository to PYTHONPATH: ```export PYTHONPATH="${PYTHONPATH}:/path/to/drlnd"```

#### Run agent

For perform learning procedure of the agent just run following command in terminal:

```python main.py --train```

It will create simple DQN Agent. For more option use:

```python main.py --help```

For simple evaluation of the agent run in termina:

```python main.py --train --(options)```

In ```--(options)```, parameters of the agent have to be placed i.e. ```--douleb_q --dueling 
--noiysnet```.
  
### Report

Description of used architecture and learning process can be found in [report](Report.md)

