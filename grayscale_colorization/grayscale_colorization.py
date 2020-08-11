def grayscale(train: bool):
    import torch
    import torchvision
    import torchvision.transforms as transforms

    if train: #train data set
        batch_size = 50000
    else:   #test data set
        batch_size = 10000
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    
    dataiter = iter(trainloader)
    data = {'image':torch.zeros((batch_size,32,32,1)), 'label':torch.zeros((batch_size))}
    #data: 이미지의 텐서값, label을 담고 있는 딕셔너리
    images, labels = dataiter.next()
    
    for a in range(batch_size) :

        for i in range(1024) :
            #Y = 0.299 * R + 0.587 * G + 0.114 * B
            data['image'][a][i//32][i%32][0] = 0.299 * images[a][0][i//32][i%32] + 0.587 * images[a][1][i//32][i%32] + \
            0.114 * images[a][2][i//32][i%32]
        data['label'][a] = labels[a]
        if a%100 == 0:   #실행된 정도를 나타내는 부분. 지워도 문제 X
            print(a)
    return data
