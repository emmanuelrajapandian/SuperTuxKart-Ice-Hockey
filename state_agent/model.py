import torch.nn as nn
import torch.nn.functional as F
import torch

class Actor(nn.Module):
    def __init__(self, state_dim=17, dropout_rate=0.):
        super(Actor, self).__init__()
        neurons = 512
        neurons2 = 256
        neurons3 = 128
        neurons4 = 64
        
        self.layer_1 = nn.Linear(state_dim, neurons)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_2 = nn.Linear(neurons, neurons2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_3 = nn.Linear(neurons2, neurons3)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer_4 = nn.Linear(neurons3, neurons4)
        self.dropout4 = nn.Dropout(dropout_rate)

        # Separate outputs for each action type, considering their ranges
        self.acceleration = nn.Linear(neurons4, 1)  # Output for acceleration
        self.steering = nn.Linear(neurons4, 1)      # Output for steering
        self.brake = nn.Linear(neurons4, 1)         # Output for brake 

    def forward(self, state):
        x = F.relu(self.dropout1(self.layer_1(state)))
        x = F.relu(self.dropout2(self.layer_2(x)))
        x = F.relu(self.dropout3(self.layer_3(x)))
        x = F.relu(self.dropout4(self.layer_4(x)))
        # Acceleration needs to be [0, 1], so use sigmoid to squash the output
        acc = torch.sigmoid(self.acceleration(x))
        # Steering is already [-1, 1] with tanh
        steer = torch.tanh(self.steering(x))
        # For brake, assuming a binary action, use sigmoid and treat the output as a probability (or threshold)
        brake = torch.sigmoid(self.brake(x))
        return torch.cat([acc, steer, brake], dim=-1)
    