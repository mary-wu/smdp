import torch
import torch.nn as nn
import torch.nn.functional as F


floatX = 'float32'


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.)


class DQN_FC(nn.Module):
    def __init__(self, state_dim=108, num_actions=3):
        super(DQN_FC, self).__init__()
        self.q1 = nn.Linear(state_dim, 128)
        self.q2 = nn.Linear(128, 64)
        self.q3 = nn.Linear(64, num_actions)

    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))

        return self.q3(q)


class BCQ_FC(nn.Module):
    def __init__(self, state_dim=108, num_actions=3):
        super(BCQ_FC, self).__init__()
        self.q1 = nn.Linear(state_dim, 128)
        self.q2 = nn.Linear(128, 64)
        self.q3 = nn.Linear(64, num_actions)

        self.i1 = nn.Linear(state_dim, 128)
        self.i2 = nn.Linear(128, 64)
        self.i3 = nn.Linear(64, num_actions)


    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))

        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = self.i3(i)
        return self.q3(q), F.log_softmax(i, dim=1), i


class DQN_Conv(nn.Module):
    def __init__(self):
        super(DQN_Conv, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=1),
            nn.ReLU(),
        )
        self.features.apply(init_weights)

        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.fc.apply(init_weights)

    def forward(self, x):
        # print(x.size())
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _feature_size(self):
        # Input tensor: (mini batch size, num channels, (dimension of past INR and demg data))
        return self.features(torch.zeros(1, 3, 7, 7)).view(-1).size(0)


class BCQ_Conv(nn.Module):
    def __init__(self):
        super(BCQ_Conv, self).__init__()
        self.c1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.c2 = nn.Conv2d(16, 16, kernel_size=2, stride=1)

        self.q1 = nn.Linear(64, 64)
        self.q2 = nn.Linear(64, 3)

        self.i1 = nn.Linear(64, 64)
        self.i2 = nn.Linear(64, 3)


    def forward(self, state):
        c = F.relu(self.c1(state))
        c = F.relu(self.c2(c))

        q = F.relu(self.q1(c.reshape(-1, 64)))
        i = F.relu(self.i1(c.reshape(-1, 64)))
        i = self.i2(i)
        return self.q2(q), F.log_softmax(i, dim=1), i

