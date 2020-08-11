import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


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


def run(layer_n, neuron_info, train_data, test_data, epoch):
    """
    :param layer_n: hidden layer number, type : int
    :param neuron_info: The number of each neuron placed on the hidden layer, type : list
    :param train_data: train_dataset, type : array
    :param test_data: test_dataset, type : array
    :param epoch: Learning iterations, type : int
    :return: layer_n;type(int), neuron_info;type(list), batch_size;type(int),
            accuracy;type(float), train_time;type(float);unit(s), test_time;type(int);unit(ns)
    """
    net = Net(layer_n, neuron_info, hin=32*32, hout=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    batch_size = len(train_data['image'][0])
    # 학습
    train_start = time.time()
    for i in range(epoch):
        running_loss = 0.0
        for j in range((len(train_data))):
            inputs = train_data['image'][j]
            labels = train_data['label'][j]
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if j % 10 == 0:
                print('epoch : [%d] loss: %.3f' % (i+1, running_loss / 10))
                running_loss = 0.0
    train_end = time.time()
    train_time = train_end-train_start

    # 평가
    test_start = time.time_ns()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data:
            inputs = train_data['image'][data]
            labels = train_data['label'][data]
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print('Accuracy of the network: %d' % accuracy)  # max는 1, min은 0
    test_end = time.time_ns()
    test_time = test_end-test_start
    return layer_n, neuron_info, batch_size, accuracy, train_time, test_time
