import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
global hin
hin = 32 * 32

class Net(nn.Module):
    #layer_n, layer_info, hout=10
    def __init__(self):
        super(Net, self).__init__()
        # 수정할 것
        # for i in range(len(layer_n)):
        #     globals()['self.fc{}'.format(i)] = layer_info[i]

        self.fc1 = nn.Linear(hin, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 10)
        
    # 순전파
    def forward(self, x): 
        x = x.view(-1, hin)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.005, momentum = 0.9)

# 학습
epoch = 50
for i in range(epoch):
    running_loss = 0
    for ind in range((len(data['image']))):
        inputs = data['image'][ind]
        labels = data['label'][ind]
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 5 == 0 and ind == 9 : 
            print('epoch : [%d] loss: %.3f' % (i, running_loss))
        running_loss = 0.0
print('Finish')
