import xarray as xr
import numpy as np

import torch
from torch.utils.data import Dataset

extreme_threshold = np.log1p(50)

class NetCDFDataset(Dataset):

    def __init__(self, dataset, test_split=0, validation_split=0, is_validation=False, is_test=False, is_2d_model=False):
        super(NetCDFDataset, self).__init__()
        
        self.is_2d_model = is_2d_model
        # orignal data format batch x time x latitude x longitude x channel
        sr = Splitter(test_split, validation_split)
        if (is_test):
            data = sr.split_test(dataset)
        elif (is_validation):
            data = sr.split_validation(dataset)
        else:
            data = sr.split_train(dataset)
        
        # data format batch x channel x time x latitude x longitude
        self.X = torch.from_numpy(data.x.values).float().permute(0, 4, 1, 2, 3)
        self.X = self.X[:,:,:5,:,:]

        # take only channel 0 from data.y.values:
        only_channel_0 = data.y.values[:,:,:,:,:1]

        # self.y = torch.from_numpy(data.y.values).float().permute(0, 4, 1, 2, 3)
        self.y = torch.from_numpy(only_channel_0).float().permute(0, 4, 1, 2, 3)

        if self.is_2d_model:
            self.X = torch.squeeze(self.X)
            self.y = torch.squeeze(self.y)
                  
        del data
        
    def __getitem__(self, index):
        return (self.X[index,:,:,:,:], self.y[index])

    def __len__(self):
        return self.X.shape[0]
  
   
    
class Splitter():
    def __init__(self, test_rate, validation_rate):
        self.test_rate = test_rate
        self.validation_rate = validation_rate / (1. - self.test_rate)
        
    def split_test(self, dataset):
        return self.__split(dataset, self.test_rate, first_part=False)
        
    def split_validation(self, dataset):
        data = self.__split(dataset, self.test_rate, first_part=True)
        return self.__split(data, self.validation_rate, first_part=False)
        
    def split_train(self, dataset):
        data = self.__split(dataset, self.test_rate, first_part=True)
        return self.__split(data, self.validation_rate, first_part=True)
    
    def __split(self, dataset, split_rate, first_part):
        if (split_rate):
            split = int(dataset.sample.size * (1. - split_rate))
            if (first_part):
                return dataset[dict(sample=slice(0,split))]
            else:
                return dataset[dict(sample=slice(split, None))]


def compute_sample_weights(labels, threshold=extreme_threshold):
    labels_np = labels.numpy()
    binary_labels = (labels_np >= threshold).any(axis=(1,2,3,4)).astype(int)
    print("Extreme vs non-extreme counts:", np.bincount(binary_labels))
    # [52959   729] = 52959 + 729 = 53688 = total train samples
    # assert sum(binary_labels) to be equals to total train samples
    class_counts = np.bincount(binary_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[binary_labels]
    return sample_weights