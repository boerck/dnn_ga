import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)
data = {'image':[], 'label':[]}
#data: 이미지의 텐서값, label을 담고 있는 딕셔너리

for a in range(10) :
    images, labels = dataiter.next()
    img = torch.zeros((32,32))
    temp2 = -1
    temp3 = 0
    
    for i in range(1024) :
        if i%32==0:
            temp2+=1
            temp3 = 0
        else:
            temp3+=1
        #Y = 0.299 * R + 0.587 * G + 0.114 * B
        img[temp2][temp3] = 0.299 * images[0][0][i//32][i%32] + 0.587 * images[0][1][i//32][i%32] + \
        0.114 * images[0][2][i//32][i%32]
    data['image'].append(img)
    data['label'].append(labels)
    
import matplotlib.pyplot as plt
import numpy as np

# 이미지를 보여주기 위한 함수

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 학습용 이미지를 무작위로 가져오기

# 이미지 보여주기
imshow(torchvision.utils.make_grid(data['image'][3]))
# 정답(label) 출력
print(' '.join('%5s' % classes[data['label'][3]]))
