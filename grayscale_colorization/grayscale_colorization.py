import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000,
                                          shuffle=True, num_workers=2)

dataiter = iter(trainloader)
data = {'image':torch.zeros((50000,32,32)), 'label':torch.zeros((50000))}
#data: 이미지의 텐서값, label을 담고 있는 딕셔너리
images, labels = dataiter.next()

img = torch.zeros((50000,32,32))
for a in range(50000) :
    
    for i in range(1024) :
        #Y = 0.299 * R + 0.587 * G + 0.114 * B
        data['image'][a][i//32][i % 32] = 0.299 * images[a][0][i//32][i % 32] + 0.587 * images[a][1][i//32][i % 32] + \
        0.114 * images[a][2][i//32][i % 32]
    data['label'][a] = labels
