import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, hidsize1)
        self.fc2 = nn.Linear(hidsize1, hidsize2)
        self.fc3 = nn.Linear(hidsize2, action_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
