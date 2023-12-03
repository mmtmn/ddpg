import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # Assuming the action space is between -1 and 1
        return x

# Assuming some dimensions for the neural network
input_size = 4  # Example state dimension
hidden_size = 128  # Example number of neurons in hidden layer
output_size = 2  # Example number of actions in the action space

mu = PolicyNetwork(input_size, hidden_size, output_size)
state_representation = [0.0, 0.0, 0.0, 0.0]  # Example state representation
s = torch.tensor(state_representation, dtype=torch.float)

a_star = mu(s)