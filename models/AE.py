import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

import sys
sys.path.append('../')
from architectures import FC_Encoder, FC_Decoder, CNN_Encoder, CNN_Decoder
from datasets import MNIST, EMNIST, FashionMNIST

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        output_size = args["embedding_size"]
        self.encoder = CNN_Encoder(output_size)

        self.decoder = CNN_Decoder(args["embedding_size"])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        return self.decode(z)

class AE(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args["cuda"] else "cpu")
        self._init_dataset()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader

        self.model = Network(args)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def _init_dataset(self):
        if self.args["dataset"] == 'MNIST':
            self.data = MNIST(self.args)
        elif self.args["dataset"] == 'EMNIST':
            self.data = EMNIST(self.args)
        elif self.args["dataset"] == 'FashionMNIST':
            self.data = FashionMNIST(self.args)
        else:
            print("Dataset not supported")
            sys.exit()

    def loss_function(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        return BCE

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            loss = self.loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args["log_interval"] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.loss_function(recon_batch, data).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
