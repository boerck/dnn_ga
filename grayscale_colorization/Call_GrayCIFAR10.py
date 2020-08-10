def Call_GrayCIFAR10(link, batch_size = 1):
    import pickle
    import torch
    with open(file, 'rb') as fo:
        loaded_data = pickle.load(fo, encoding='bytes')
    
    data = {'image':[], 'label':[]}
    
    for i in range(50000//batch_size):
        data['image'].append( loaded_data['image'][i * batchsize : (i+1) * batchsize] )
        data['label'].append( loaded_data['label'][i * batchsize : (i+1) * batchsize] )
        
    return data
