"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
import glob
import os

class MyDataset(Dataset):
    def __init__(self,path,id):
        temp =  glob.glob(os.path.join(path,'*.npy'))
        self.paths = [f for f in temp if f'section_{str(id).zfill(2)}_' in f]

    def __getitem__(self,item):
        path = self.paths[item]
        mel = np.load(path)
        mel = mel[:,:312]
        mel = mel.reshape(1,mel.shape[0],mel.shape[1])
        return mel, int('anomaly' in path)

    def __len__(self):
        return len(self.paths)



##
def load_data(opt):
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = '../mel_features/{}'.format(opt.dataset)

    
    splits = ['train', 'source_test', 'target_test']
    drop_last_batch = {'train': True, 'source_test': False,'target_test':False}
    shuffle = {'train': True, 'source_test': False,'target_test':False}
    datasets = {x: MyDataset(os.path.join(opt.dataroot, x),opt.id) for x in splits}
    dataloader = {x: DataLoader(datasets[x],opt.batchsize,shuffle[x],drop_last=drop_last_batch[x]) for x in splits}
        
    return dataloader