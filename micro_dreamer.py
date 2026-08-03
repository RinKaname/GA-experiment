import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# A tiny, lightweight "Micro-Dreamer" built for text games.
class WorldModel(nn.Module):
    """ Learns the 'physics' of the game. P(Next_State, Reward | State, Action) """
    def __init__(self, state_dim=9, action_dim=3, hidden_dim=64):
        super(WorldModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.state_out = nn.Linear(hidden_dim, state_dim)
        self.reward_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        next_state = self.state_out(x)
        reward = self.reward_out(x)
        return next_state, reward

class Actor(nn.Module):
    """ Chooses actions based on the state. """
    def __init__(self, state_dim=9, action_dim=3, hidden_dim=32):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        raw_action = self.action_out(x)

        # Map raw outputs to bounded actions
        action_land = torch.tanh(raw_action[:, 0:1])
        action_feed = torch.sigmoid(raw_action[:, 1:2])
        action_plant = torch.sigmoid(raw_action[:, 2:3])

        action = torch.cat([action_land, action_feed, action_plant], dim=-1)
        return action

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

    def push(self, state, action, reward, next_state):
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        s = torch.FloatTensor(np.array(self.states)[indices])
        a = torch.FloatTensor(np.array(self.actions)[indices])
        r = torch.FloatTensor(np.array(self.rewards)[indices]).unsqueeze(1)
        ns = torch.FloatTensor(np.array(self.next_states)[indices])
        return s, a, r, ns

    def __len__(self):
        return len(self.states)

def normalize_state(state):
    """ Normalization used previously to keep NN inputs stable. """
    if isinstance(state, torch.Tensor):
        state_np = state.detach().numpy()
    else:
        state_np = np.array(state)

    if state_np.ndim == 1:
        norm = np.array([
            state_np[0] / 12.0,
            state_np[1] / 1000.0,
            state_np[2] / 10000.0,
            state_np[3] / 5000.0,
            state_np[4] / 30.0,
            state_np[5] / 100.0,
            state_np[6] / 100.0,
            state_np[7] / 100.0,
            state_np[8] / 4.0
        ])
    else:
        norm = np.copy(state_np)
        norm[:, 0] /= 12.0
        norm[:, 1] /= 1000.0
        norm[:, 2] /= 10000.0
        norm[:, 3] /= 5000.0
        norm[:, 4] /= 30.0
        norm[:, 5] /= 100.0
        norm[:, 6] /= 100.0
        norm[:, 7] /= 100.0
        norm[:, 8] /= 4.0

    return torch.FloatTensor(norm)
