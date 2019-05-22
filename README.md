# Udacity_Navigation

Submission for completing the Udacity Project

Uses DQN to solve the navigation project.
Contains the weights of the trained RL bot to solve the problem.
Graphs indicating the progress of the agent and when it solved the problem.
Requires OpenAI gym

Target Environment
Banana_Linux_NoVis/Banana.x86_64

State space = 37
Action space = 4

Goal is to navigate a 3d world and collect yellow bananas whlie avoiding blue bananas

## Project Layout

# Agents/

DQN, Priority_DQN

# Buffers/

Vanilla ReplayBuffer, PriorityReplayBuffer

# Networks/

QNetwork, Dueling_QNetwork

# Main files

train.py
checkpoint.pth
torch.yml

# Requirements

Are contained in the torch.yml file.
