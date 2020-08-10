import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self, layer_n, neuron_info, hin=32*32, hout=10):
        """
        :param layer_n: hidden layer number
        :param neuron_info: The number of each neuron placed on the hidden layer
        :param hin: input number
        :param hout: output number
        """
        super(Net, self).__init__()
        self.hin = hin
        self.layer_n = layer_n
        neuron_info.insert(0, self.hin)
        neuron_info.append(hout)
        k = 0
        for i in range(self.layer_n + 1):  # 동적 인스턴스 변수 설정
            setattr(self, "fc{}".format(str(i+1)), nn.Linear(neuron_info[k], neuron_info[k+1]))
            k += 1
        
    # 순전파
    def forward(self, x):
        x = x.view(-1, self.hin)
        for i in range(self.layer_n):
            x = F.relu(getattr(self, "fc{}".format(str(i+1)))(x))
        x = getattr(self, "fc{}".format(str(self.layer_n+1)))(x)
        return x


def run():
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 학습
    epoch = 50
    for i in range(epoch):
        running_loss = 0.0
        for ind in range((len(data['image']))):
            inputs = data['image'][ind]
            labels = data['label'][ind]
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 0 and ind == 9:
                print('epoch : [%d] loss: %.3f' % (i, running_loss))
            running_loss = 0.0
    print('Finish')