import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self, hin=32*32, layer_n, layer_info, hout=10):
        super(Net, self).__init__()
        for i in range(len(layer_n)):
            globals()['self.fc{}'.format(i)] = layer_info[i]

        self.fc1 = nn.Linear(hin, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)