import pickle
import torch

def save(name1: str, data:dict):
    with open(str('img/'+name1+'.bin'), 'wb') as f:   #img라는 폴더를 미리 만들어야함
        pickle.dump(data, f)


def load(name1: str, batch_size=1):
    with open(str('img/'+name1+'.bin'), 'rb') as f:
        loaded_data = pickle.load(f)

    data = {'image': [], 'label': []}
    
    for i in range(len(loaded_data['image']) // batch_size):   #batch_size에 맞게 변환
        data['image'].append(torch.reshape(loaded_data['image'][i * batch_size: (i + 1) * batch_size], (batch_size,1024)))
        #2차원 tensor로 변환
        data['label'].append(loaded_data['label'][i * batch_size: (i + 1) * batch_size])
    return data
