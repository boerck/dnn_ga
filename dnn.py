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

        self.fc1 = nn.Linear(hin, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)
        
    # 순전파
    def forward(self, x): 
        # ReLu function 사용
        x = x.view(-1, hin)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x
   # backward는 자동으로 정의됨 
net = Net()
# 학습률은 얼마로? 일단 0.005로 잡고 시작함
# momentum 은 일단 0.9로 잡음
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.005, momentum = 0.9)

# 학습하기
epoch = 100
for i in range(epoch):
    running_loss = 0
    for ind in range((len(data['image']))):
        inputs = data['image'][ind]
        labels = data['label'][ind]
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        print('outputs:',outputs)
        print('labels:',labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' % (i + 1, ind + 1, running_loss / 2000))
        running_loss = 0.0
print('Finish')
