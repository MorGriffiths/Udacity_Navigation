# Udacity_Navigation

Submission for completing the Udacity Project

## Implementation

The agent is a combination of 4 add-ons to vanilla DQN.

- Priority Replay
- Double DQN
- Dueling DQN
- Polyak Averaging

Contains the weights of the trained RL bot to solve the problem.
Graphs indicating the progress of the agent and when it solved the problem.

The DQN agent solved the enviroment in 625 steps (Average Reward > 13).

## There are two Environments:

Vector Banana

- State space = 37
- Action space = 4

Visual Banana

- State space = Array of raw pixels (1, 84, 84, 3)
- Action space = 4

---

Goal is to navigate a 3d world and collect yellow bananas whlie avoiding blue bananas.
Each yellow banana is worth +1. Each blue banana -1

In each environment the Agent has 4 actions:

- Move forward
- Move backward
- Rotate left
- Rotate right

The environment is considered to be solved when the agent reaches an average reward of +13

## Project Layout

### Agents

DQN, Priority_DQN

### Buffers

Vanilla ReplayBuffer, PriorityReplayBuffer

### Networks

QNetwork, Dueling_QNetwork

### Main files

train.py
checkpoint.pth

## Installation

Clone the repository.

```
git clone git@github.com:MorGriffiths/Udacity_Navigation.git
cd Udacity_Navigation
```

Create a virtual environment and activate it.

```
python -m venv banana
source banana/bin/activate
```

Install Unity ml-agents.

```
git clone https://github.com/Unity-Technologies/ml-agents.git
git -C ml-agents checkout 0.4.0b
pip install ml-agents/python/.
```

Install the project requirements.

```
pip install -r requirements.txt
```

## Download the Vector Banana Unity Environment which matches your operating system

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- [Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- [Windows (64 bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

## Download the Visual Banana Unity Environment

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- [Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- [Windows (64 bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Place the environment into the Environments folder.
If necessary, inside main.py, change the path to the unity environment appropriately

## Run the project

Each project solution is contained within the notebooks "Navigation.ipynb" and "Navigation_Pixels.ipynb"

Make sure the environment path is correctly set in the desired notebook. Then run the cells as wanted.

## Futher details

The Vector Banana report.md is in the Vector_banana folder. Along with the performance graph and the weights.

Additionally, i tried training visual banana from scratch but likely due to memory constraints it essentially broke in the notebook format. I expect i will be able to train effectively to outside of that. And in addition run some refresh to clear the cache every N epsidoes.

[link](https://medium.com/@C5ipo7i/improving-dqn-cde578df5d73?postPublishedType=initial) A medium article describing the different add-ons i implemented to DQN
