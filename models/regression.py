import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['regressionNet']


class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.embedding_dim = 32
        self.min_delta = -4297426.0
        self.max_delta = 4668141.0
        self.num_embeddings = int(self.max_delta - self.min_delta + 1)   # #.classes

        self.word_embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)  # 9 tensors of size 6

        # self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        # self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3)
        # self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3)
        #
        # self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3)
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3)

        # self.conv2_drop = nn.Dropout2d()

        self.fc0_0 = nn.Linear(32, 256)
        self.fc0_1 = nn.Linear(256, 256)

        self.fc1 = nn.Linear(256, 1)
        # self.fc2 = nn.Linear(50, 10)
        # self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = x - self.min_delta
        if torch.min(x) < 0:
            raise ValueError('found < 0')
        if torch.max(x) > self.num_embeddings:
            raise ValueError('found max outside')
        x = self.word_embeddings(x)

        x = torch.unsqueeze(x, 1)
        # x = self.conv1(x)
        # x = F.relu(F.max_pool2d(x, 2))
        # x = F.relu(F.max_pool2d((self.conv2(x)), 2))
        # x = F.relu(F.max_pool2d((self.conv3_1(x)), 2))
        # x = F.relu(F.max_pool2d((self.conv3_2(x)), 2))
        # x = F.relu(F.max_pool2d((self.conv4_1(x)), 2))
        # x = F.relu(F.max_pool2d((self.conv4_2(x)), 2))

        # x = x.view(-1, 128*5*5)
        x = F.relu(self.fc0_0(x))
        x = F.relu(self.fc0_1(x))
        x = F.relu(self.fc0_1(x))
        x = F.relu(self.fc0_1(x))

        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # x = self.fc3(x)
        x = torch.squeeze(x)
        x = torch.unsqueeze(x, 1)
        # x = x + self.min_delta
        return x


def regressionNet():
    model = RegressionNet()
    return model
