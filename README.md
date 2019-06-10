# Udacity_Navigation

![DQN agent](assets/show_result.gif)

Submission for completing the Udacity Project

Uses DQN to solve the navigation project.
Contains the weights of the trained RL bot to solve the problem.
Graphs indicating the progress of the agent and when it solved the problem.

## There are two Environments:

Vector Banana
State space = 37
Action space = 4

Visual Banana
State space = ?
Action space = 4

---

Goal is to navigate a 3d world and collect yellow bananas whlie avoiding blue bananas.
Each yellow banana is worth +1. Each blue banana -1

Agent has 4 actions:
Move forward
Move backward
Rotate left
Rotate right

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
torch.yml

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

## Download the Unity Environment which matches your operating system

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- [Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- [Windows (64 bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

If necessary, inside the ipynb files, change the path to the unity environment appropriately
