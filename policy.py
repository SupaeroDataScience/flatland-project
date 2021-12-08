import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class PolicyNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidsize1=16, hidsize2=32):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidsize2)
        self.fc2 = nn.Linear(hidsize2, hidsize2)
        self.fc3 = nn.Linear(hidsize2, hidsize1)
        self.fc4 = nn.Linear(hidsize1, action_size)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        optimizer.step()

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        #output = F.softmax(x, dim=1)
        output = F.log_softmax(x, dim=1)
        return output


class NeuroevoPolicy:
    """A static policy network to be optimized by evolution"""

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cpu")
        self.model = PolicyNetwork(state_size, action_size).to(self.device)
        self.model = self.model.to(self.device).double()

    def act(self, state):
        state = torch.from_numpy(state).double().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        return np.argmax(action_values.cpu().data.numpy())

    def set_params(self, params):
        if np.isnan(params).any():
        #if pd.isna(params):
            raise
        a = torch.tensor(params, device=self.device)
        torch.nn.utils.vector_to_parameters(a, self.model.parameters())
        self.model = self.model.to(self.device).double()

    def get_params(self):
        with torch.no_grad():
            params = self.model.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().double().numpy()

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename))
        self.model = self.model.to(self.device).double()

    def test(self):
        self.act(np.array([[0] * self.state_size]))
