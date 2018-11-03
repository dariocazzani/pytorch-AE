import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as  np

class FC_Encoder(nn.Module):
    def __init__(self, output_size):
        super(FC_Encoder, self).__init__()
        self.fc1 = nn.Linear(784, output_size)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return h1

class FC_Decoder(nn.Module):
    def __init__(self, embedding_size):
        super(FC_Decoder, self).__init__()
        self.fc3 = nn.Linear(embedding_size, 1024)
        self.fc4 = nn.Linear(1024, 784)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(1, 28, 28)):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size

        #convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                     out_channels=64,
                     kernel_size=4,
                     stride=1,
                     padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.flat_fts = self.get_flat_fts(self.conv)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x.view(-1, *self.input_size))
        x = x.view(-1, self.flat_fts)
        return self.linear(x)

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(1, 28, 28)):
        super(CNN_Decoder, self).__init__()
        self.input_height = 28
        self.input_width = 28
        self.input_dim = embedding_size
        self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x.view(-1, self.input_width*self.input_height)
