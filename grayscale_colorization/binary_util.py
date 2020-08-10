import pickle

def save(name1:str, data:dict):
    with open (str('img/'+name1+'.bin'), 'wb') as f:
        pickle.dump(data,f)

def load(name1:str, batch_size=1):
    with open (str('img/'+name1+'.bin'), 'rb') as f:
        loaded_data = pickle.load(f)

    data = {'image': [], 'label': []}

    for i in range(50000 // batch_size):
        data['image'].append(loaded_data['image'][i * batch_size: (i + 1) * batch_size])
        data['label'].append(loaded_data['label'][i * batch_size: (i + 1) * batch_size])

    return data


