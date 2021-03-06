import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda')


        
class CNN_1D(nn.Module):
    def __init__(self):
        super(CNN_1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 50, stride=5, padding=25)
        self.bn1 = nn.BatchNorm1d(64)
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=10, stride=5)
        
        ##########################################################################
        self.conv2a_1 = nn.Conv1d(64, 256, 1, stride=1, padding=0)
        self.bn2a_1 = nn.BatchNorm1d(256)
        
        self.conv2a_2a = nn.Conv1d(64, 64, 1, stride=1, padding=0)
        self.bn2a_2a = nn.BatchNorm1d(64)
        self.activation2a_2a = nn.ReLU()
        
        self.conv2a_2b = nn.Conv1d(64, 64, 10, stride=1, padding=5)
        self.bn2a_2b = nn.BatchNorm1d(64)
        self.activation2a_2b = nn.ReLU()
        
        self.conv2a_2c = nn.Conv1d(64, 256, 1, stride=1, padding=0)
        self.bn2a_2c = nn.BatchNorm1d(256)
        
        self.activation2a = nn.ReLU()
        
        ##########################################################################
        self.conv2b_2a = nn.Conv1d(256, 64, 1, stride=1, padding=0)
        self.bn2b_2a = nn.BatchNorm1d(64)
        self.activation2b_2a = nn.ReLU()
        
        self.conv2b_2b = nn.Conv1d(64, 64, 10, stride=1, padding=5)
        self.bn2b_2b = nn.BatchNorm1d(64)
        self.activation2b_2b = nn.ReLU()
        
        self.conv2b_2c = nn.Conv1d(64, 256, 1, stride=1, padding=0)
        self.bn2b_2c = nn.BatchNorm1d(256)
        
        self.activation2b = nn.ReLU()
        
        
        ##########################################################################
        self.conv2c_2a = nn.Conv1d(256, 64, 1, stride=1, padding=0)
        self.bn2c_2a = nn.BatchNorm1d(64)
        self.activation2c_2a = nn.ReLU()
        
        self.conv2c_2b = nn.Conv1d(64, 64, 10, stride=1, padding=5)
        self.bn2c_2b = nn.BatchNorm1d(64)
        self.activation2c_2b = nn.ReLU()
        
        self.conv2c_2c = nn.Conv1d(64, 256, 1, stride=1, padding=0)
        self.bn2c_2c = nn.BatchNorm1d(256)
        
        self.activation2c = nn.ReLU()
        
        ##########################################################################
        self.conv3a_1 = nn.Conv1d(256, 512, 1, stride=5, padding=0)
        self.bn3a_1 = nn.BatchNorm1d(512)
        
        self.conv3a_2a = nn.Conv1d(256, 128, 1, stride=5, padding=0)
        self.bn3a_2a = nn.BatchNorm1d(128)
        self.activation3a_2a = nn.ReLU()
        
        self.conv3a_2b = nn.Conv1d(128, 128, 10, stride=1, padding=5)
        self.bn3a_2b = nn.BatchNorm1d(128)
        self.activation3a_2b = nn.ReLU()
        
        self.conv3a_2c = nn.Conv1d(128, 512, 1, stride=1, padding=0)
        self.bn3a_2c = nn.BatchNorm1d(512)
        
        self.activation3a = nn.ReLU()
        
        
        ##########################################################################
        self.conv3b_2a = nn.Conv1d(512, 128, 1, stride=1, padding=0)
        self.bn3b_2a = nn.BatchNorm1d(128)
        self.activation3b_2a = nn.ReLU()
        
        self.conv3b_2b = nn.Conv1d(128, 128, 10, stride=1, padding=5)
        self.bn3b_2b = nn.BatchNorm1d(128)
        self.activation3b_2b = nn.ReLU()
        
        self.conv3b_2c = nn.Conv1d(128, 512, 1, stride=1, padding=0)
        self.bn3b_2c = nn.BatchNorm1d(512)
        
        self.activation3b = nn.ReLU()
        
        ##########################################################################
        self.conv3c_2a = nn.Conv1d(512, 128, 1, stride=1, padding=0)
        self.bn3c_2a = nn.BatchNorm1d(128)
        self.activation3c_2a = nn.ReLU()
        
        self.conv3c_2b = nn.Conv1d(128, 128, 10, stride=1, padding=5)
        self.bn3c_2b = nn.BatchNorm1d(128)
        self.activation3c_2b = nn.ReLU()
        
        self.conv3c_2c = nn.Conv1d(128, 512, 1, stride=1, padding=0)
        self.bn3c_2c = nn.BatchNorm1d(512)
        
        self.activation3c = nn.ReLU()
        
        ##########################################################################
        self.conv3d_2a = nn.Conv1d(512, 128, 1, stride=1, padding=0)
        self.bn3d_2a = nn.BatchNorm1d(128)
        self.activation3d_2a = nn.ReLU()
        
        self.conv3d_2b = nn.Conv1d(128, 128, 10, stride=1, padding=5)
        self.bn3d_2b = nn.BatchNorm1d(128)
        self.activation3d_2b = nn.ReLU()
        
        self.conv3d_2c = nn.Conv1d(128, 512, 1, stride=1, padding=0)
        self.bn3d_2c = nn.BatchNorm1d(512)
        
        self.activation3d = nn.ReLU()
        
        ##########################################################################
        self.conv4a_1 = nn.Conv1d(512, 1024, 1, stride=5, padding=0)
        self.bn4a_1 = nn.BatchNorm1d(1024)
        
        self.conv4a_2a = nn.Conv1d(512, 256, 1, stride=5, padding=0)
        self.bn4a_2a = nn.BatchNorm1d(256)
        self.activation4a_2a = nn.ReLU()
        
        self.conv4a_2b = nn.Conv1d(256, 256, 10, stride=1, padding=5)
        self.bn4a_2b = nn.BatchNorm1d(256)
        self.activation4a_2b = nn.ReLU()
        
        self.conv4a_2c = nn.Conv1d(256, 1024, 1, stride=1, padding=0)
        self.bn4a_2c = nn.BatchNorm1d(1024)
        
        self.activation4a = nn.ReLU()
        
        ##########################################################################
        self.conv4b_2a = nn.Conv1d(1024, 256, 1, stride=1, padding=0)
        self.bn4b_2a = nn.BatchNorm1d(256)
        self.activation4b_2a = nn.ReLU()
        
        self.conv4b_2b = nn.Conv1d(256, 256, 10, stride=1, padding=5)
        self.bn4b_2b = nn.BatchNorm1d(256)
        self.activation4b_2b = nn.ReLU()
        
        self.conv4b_2c = nn.Conv1d(256, 1024, 1, stride=1, padding=0)
        self.bn4b_2c = nn.BatchNorm1d(1024)
        
        self.activation4b = nn.ReLU()
        
        
        ##########################################################################
        self.conv4c_2a = nn.Conv1d(1024, 256, 1, stride=1, padding=0)
        self.bn4c_2a = nn.BatchNorm1d(256)
        self.activation4c_2a = nn.ReLU()
        
        self.conv4c_2b = nn.Conv1d(256, 256, 10, stride=1, padding=5)
        self.bn4c_2b = nn.BatchNorm1d(256)
        self.activation4c_2b = nn.ReLU()
        
        self.conv4c_2c = nn.Conv1d(256, 1024, 1, stride=1, padding=0)
        self.bn4c_2c = nn.BatchNorm1d(1024)
        
        self.activation4c = nn.ReLU()
        
        ##########################################################################
        self.conv4d_2a = nn.Conv1d(1024, 256, 1, stride=1, padding=0)
        self.bn4d_2a = nn.BatchNorm1d(256)
        self.activation4d_2a = nn.ReLU()
        
        self.conv4d_2b = nn.Conv1d(256, 256, 10, stride=1, padding=5)
        self.bn4d_2b = nn.BatchNorm1d(256)
        self.activation4d_2b = nn.ReLU()
        
        self.conv4d_2c = nn.Conv1d(256, 1024, 1, stride=1, padding=0)
        self.bn4d_2c = nn.BatchNorm1d(1024)
        
        self.activation4d = nn.ReLU()
        
        ##########################################################################
        self.conv4e_2a = nn.Conv1d(1024, 256, 1, stride=1, padding=0)
        self.bn4e_2a = nn.BatchNorm1d(256)
        self.activation4e_2a = nn.ReLU()
        
        self.conv4e_2b = nn.Conv1d(256, 256, 10, stride=1, padding=5)
        self.bn4e_2b = nn.BatchNorm1d(256)
        self.activation4e_2b = nn.ReLU()
        
        self.conv4e_2c = nn.Conv1d(256, 1024, 1, stride=1, padding=0)
        self.bn4e_2c = nn.BatchNorm1d(1024)
        
        self.activation4e = nn.ReLU()
        
        ##########################################################################
        self.conv4f_2a = nn.Conv1d(1024, 256, 1, stride=1, padding=0)
        self.bn4f_2a = nn.BatchNorm1d(256)
        self.activation4f_2a = nn.ReLU()
        
        self.conv4f_2b = nn.Conv1d(256, 256, 10, stride=1, padding=5)
        self.bn4f_2b = nn.BatchNorm1d(256)
        self.activation4f_2b = nn.ReLU()
        
        self.conv4f_2c = nn.Conv1d(256, 1024, 1, stride=1, padding=0)
        self.bn4f_2c = nn.BatchNorm1d(1024)
        
        self.activation4f = nn.ReLU()
        
        ##########################################################################
        self.conv5a_1 = nn.Conv1d(1024, 2048, 1, stride=5, padding=0)
        self.bn5a_1 = nn.BatchNorm1d(2048)
        
        self.conv5a_2a = nn.Conv1d(1024, 512, 1, stride=5, padding=0)
        self.bn5a_2a = nn.BatchNorm1d(512)
        self.activation5a_2a = nn.ReLU()
        
        self.conv5a_2b = nn.Conv1d(512, 512, 10, stride=1, padding=5)
        self.bn5a_2b = nn.BatchNorm1d(512)
        self.activation5a_2b = nn.ReLU()
        
        self.conv5a_2c = nn.Conv1d(512, 2048, 1, stride=1, padding=0)
        self.bn5a_2c = nn.BatchNorm1d(2048)
        
        self.activation5a = nn.ReLU()
        
        ##########################################################################
        self.conv5b_2a = nn.Conv1d(2048, 512, 1, stride=1, padding=0)
        self.bn5b_2a = nn.BatchNorm1d(512)
        self.activation5b_2a = nn.ReLU()
        
        self.conv5b_2b = nn.Conv1d(512, 512, 10, stride=1, padding=5)
        self.bn5b_2b = nn.BatchNorm1d(512)
        self.activation5b_2b = nn.ReLU()
        
        self.conv5b_2c = nn.Conv1d(512, 2048, 1, stride=1, padding=0)
        self.bn5b_2c = nn.BatchNorm1d(2048)
        
        self.activation5b = nn.ReLU()
        
        
        ##########################################################################
        self.conv5c_2a = nn.Conv1d(2048, 512, 1, stride=1, padding=0)
        self.bn5c_2a = nn.BatchNorm1d(512)
        self.activation5c_2a = nn.ReLU()
        
        self.conv5c_2b = nn.Conv1d(512, 512, 10, stride=1, padding=5)
        self.bn5c_2b = nn.BatchNorm1d(512)
        self.activation5c_2b = nn.ReLU()
        
        self.conv5c_2c = nn.Conv1d(512, 2048, 1, stride=1, padding=0)
        self.bn5c_2c = nn.BatchNorm1d(2048)
        
        self.activation5c = nn.ReLU()
        
        ##########################################################################
        self.AvgPool = nn.AvgPool1d(21)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.maxpool1(x)
        
        res = x
        res = self.conv2a_1(res)
        res = self.bn2a_1(res)
        x = self.conv2a_2a(x)
        x = self.bn2a_2a(x)
        x = self.activation2a_2a(x)
        x = self.conv2a_2b(x)
        x = self.bn2a_2b(x)
        x = self.activation2a_2b(x)
        x = self.conv2a_2c(x)
        x = self.bn2a_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation2a(x)
        
        res = x
        x = self.conv2b_2a(x)
        x = self.bn2b_2a(x)
        x = self.activation2b_2a(x)
        x = self.conv2b_2b(x)
        x = self.bn2b_2b(x)
        x = self.activation2b_2b(x)
        x = self.conv2b_2c(x)
        x = self.bn2b_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation2b(x)
        
        res = x
        x = self.conv2c_2a(x)
        x = self.bn2c_2a(x)
        x = self.activation2c_2a(x)
        x = self.conv2c_2b(x)
        x = self.bn2c_2b(x)
        x = self.activation2c_2b(x)
        x = self.conv2c_2c(x)
        x = self.bn2c_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation2c(x)
        
        res = x
        res = self.conv3a_1(res)
        res = self.bn3a_1(res)
        x = self.conv3a_2a(x)
        x = self.bn3a_2a(x)
        x = self.activation3a_2a(x)
        x = self.conv3a_2b(x)
        x = self.bn3a_2b(x)
        x = self.activation3a_2b(x)
        x = self.conv3a_2c(x)
        x = self.bn3a_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation3a(x)
        
        res = x
        x = self.conv3b_2a(x)
        x = self.bn3b_2a(x)
        x = self.activation3b_2a(x)
        x = self.conv3b_2b(x)
        x = self.bn3b_2b(x)
        x = self.activation3b_2b(x)
        x = self.conv3b_2c(x)
        x = self.bn3b_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation3b(x)
        
        res = x
        x = self.conv3c_2a(x)
        x = self.bn3c_2a(x)
        x = self.activation3c_2a(x)
        x = self.conv3c_2b(x)
        x = self.bn3c_2b(x)
        x = self.activation3c_2b(x)
        x = self.conv3c_2c(x)
        x = self.bn3c_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation3c(x)
        
        
        res = x
        x = self.conv3d_2a(x)
        x = self.bn3d_2a(x)
        x = self.activation3d_2a(x)
        x = self.conv3d_2b(x)
        x = self.bn3d_2b(x)
        x = self.activation3d_2b(x)
        x = self.conv3d_2c(x)
        x = self.bn3d_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation3d(x)
        
        res = x
        res = self.conv4a_1(res)
        res = self.bn4a_1(res)
        x = self.conv4a_2a(x)
        x = self.bn4a_2a(x)
        x = self.activation4a_2a(x)
        x = self.conv4a_2b(x)
        x = self.bn4a_2b(x)
        x = self.activation4a_2b(x)
        x = self.conv4a_2c(x)
        x = self.bn4a_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation4a(x)
        
        res = x
        x = self.conv4b_2a(x)
        x = self.bn4b_2a(x)
        x = self.activation4b_2a(x)
        x = self.conv4b_2b(x)
        x = self.bn4b_2b(x)
        x = self.activation4b_2b(x)
        x = self.conv4b_2c(x)
        x = self.bn4b_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation4b(x)
        
        res = x
        x = self.conv4c_2a(x)
        x = self.bn4c_2a(x)
        x = self.activation4c_2a(x)
        x = self.conv4c_2b(x)
        x = self.bn4c_2b(x)
        x = self.activation4c_2b(x)
        x = self.conv4c_2c(x)
        x = self.bn4c_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation4c(x)
        
        res = x
        x = self.conv4d_2a(x)
        x = self.bn4d_2a(x)
        x = self.activation4d_2a(x)
        x = self.conv4d_2b(x)
        x = self.bn4d_2b(x)
        x = self.activation4d_2b(x)
        x = self.conv4d_2c(x)
        x = self.bn4d_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation4d(x)
        
        res = x
        x = self.conv4e_2a(x)
        x = self.bn4e_2a(x)
        x = self.activation4e_2a(x)
        x = self.conv4e_2b(x)
        x = self.bn4e_2b(x)
        x = self.activation4e_2b(x)
        x = self.conv4e_2c(x)
        x = self.bn4e_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation4e(x)
        
        res = x
        x = self.conv4f_2a(x)
        x = self.bn4f_2a(x)
        x = self.activation4f_2a(x)
        x = self.conv4f_2b(x)
        x = self.bn4f_2b(x)
        x = self.activation4f_2b(x)
        x = self.conv4f_2c(x)
        x = self.bn4f_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation4f(x)
        
        res = x
        res = self.conv5a_1(res)
        res = self.bn5a_1(res)
        x = self.conv5a_2a(x)
        x = self.bn5a_2a(x)
        x = self.activation5a_2a(x)
        x = self.conv5a_2b(x)
        x = self.bn5a_2b(x)
        x = self.activation5a_2b(x)
        x = self.conv5a_2c(x)
        x = self.bn5a_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation5a(x)
        
        res = x
        x = self.conv5b_2a(x)
        x = self.bn5b_2a(x)
        x = self.activation5b_2a(x)
        x = self.conv5b_2b(x)
        x = self.bn5b_2b(x)
        x = self.activation5b_2b(x)
        x = self.conv5b_2c(x)
        x = self.bn5b_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation5b(x)
        
        res = x
        x = self.conv5c_2a(x)
        x = self.bn5c_2a(x)
        x = self.activation5c_2a(x)
        x = self.conv5c_2b(x)
        x = self.bn5c_2b(x)
        x = self.activation5c_2b(x)
        x = self.conv5c_2c(x)
        x = self.bn5c_2c(x)
        x = x[:,:,:-1] + res
        x = self.activation5c(x)
        
        x = self.AvgPool(x)
        x = x.view(x.size(0), -1)
        
        return x
    
    
model_CNN = CNN_1D()


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn = model_CNN
        self.fc1 = nn.Linear(6144,4096)
        self.activation_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096,2048)
        self.activation_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(2048,1024)
        self.activation_fc3 = nn.ReLU()
        self.fc4 = nn.Linear(1024,2)
        
    def forward(self, x1, x2):
        x1 = self.cnn(x1)
        x2 = self.cnn(x2)
        
        x_add = x1+x2
        x_minus = x1-x2
        x_multiply = x1*x2
        x = torch.cat((x_add, x_minus, x_multiply),-1)
        x = self.fc1(x)
        x = self.activation_fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.activation_fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.activation_fc3(x)
        x = self.fc4(x)
        return x
        
model = MyModel()


def random_sampling_and_normalization(sample,sampling_length=256*256,padding=10):
    length = sample.size(1)
    if length<sampling_length:
        pad = int((sampling_length-length)/2)
        sample = torch.cat((torch.zeros((1,pad)),sample,torch.zeros((1,pad))),-1)
    sample = torch.cat((torch.zeros((1,padding)),sample,torch.zeros((1,padding))),-1)
    length = sample.size(1)
    random_num = np.random.randint(low=0, high=(length-sampling_length-1))
    sample = sample[:,random_num:random_num+sampling_length]
    
    #normalization
    #channel=（channel-mean）/std
    mean = torch.mean(sample)
    std = torch.std(sample)
    sample = (sample-mean)/std
    return sample
    
def center_sampling_and_normalization(sample,sampling_length=256*256):
    length = sample.size(1)
    if length<sampling_length:
        pad = int(sampling_length-length)
        sample = torch.cat((sample,torch.zeros((1,pad))),-1)
    sample = sample[:,:sampling_length]
    #normalization
    #channel=（channel-mean）/std
    mean = torch.mean(sample)
    std = torch.std(sample)
    sample = (sample-mean)/std
    return sample


class my_dataset(Dataset):
    def __init__(self, df_path, train = False):
        self.df = pd.read_csv(df_path)
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        sample_1_name = self.df.iloc[idx]['sample 1']
        sample_1_path = '/scratch/cz2064/myjupyter/Time_Series/Data/data_VoxCeleb/wav/'+sample_1_name
        sample_1,_ = torchaudio.load(sample_1_path)
        
        sample_2_name = self.df.iloc[idx]['sample 2']
        sample_2_path = '/scratch/cz2064/myjupyter/Time_Series/Data/data_VoxCeleb/wav/'+sample_2_name
        sample_2,_ = torchaudio.load(sample_2_path)
        
        if self.train:
            sample_1_tensor = random_sampling_and_normalization(sample_1)
            sample_2_tensor = random_sampling_and_normalization(sample_2)
        else:
            sample_1_tensor = center_sampling_and_normalization(sample_1)
            sample_2_tensor = center_sampling_and_normalization(sample_2)
            
            
        label = self.df.loc[idx,'True or False']
        label = torch.tensor(label, dtype=torch.long)
        
        sample = {'x1': sample_1_tensor, 'x2': sample_2_tensor, 'y': label}
        
        return sample
        
        
train_df_path = '/scratch/cz2064/myjupyter/Time_Series/notebook/train.csv'
val_df_path = '/scratch/cz2064/myjupyter/Time_Series/notebook/val.csv'
test_df_path = '/scratch/cz2064/myjupyter/Time_Series/notebook/test.csv'
BATCH_SIZE = 32
#train_loader = DataLoader(my_dataset(train_df_path,train = True), batch_size=BATCH_SIZE, shuffle=True)
train_sampler = torch.utils.data.sampler.RandomSampler(my_dataset(train_df_path,train = True)\
                                                       ,num_samples=50000,replacement=True)
train_loader = DataLoader(my_dataset(train_df_path,train = True), batch_size=BATCH_SIZE, \
                          sampler = train_sampler,num_workers=16)
val_loader = DataLoader(my_dataset(val_df_path), batch_size=BATCH_SIZE, shuffle=True,num_workers=16)
test_loader = DataLoader(my_dataset(test_df_path), batch_size=BATCH_SIZE, shuffle=True)


def train(model, train_loader=train_loader, val_loader=val_loader, learning_rate=5e-5, num_epoch=100):
    start_time = time.time()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    
    train_loss_return = []
    train_acc_return = []
    val_loss_return = []
    val_acc_return = []
    best_acc = 0
    
    for epoch in range(num_epoch):
        # Training steps
        correct = 0
        total = 0
        predictions = []
        truths = []
        model.train()
        train_loss_list = []
        for i, (sample) in enumerate(train_loader):
            sample_1 = sample['x1'].to(device)
            sample_2 = sample['x2'].to(device)
            labels = sample['y'].to(device)
            outputs = model(sample_1,sample_2)
            pred = outputs.data.max(-1)[1]
            predictions += list(pred.cpu().numpy())
            truths += list(labels.cpu().numpy())
            total += labels.size(0)
            correct += (pred == labels).sum()
            model.zero_grad()
            loss = loss_fn(outputs, labels)
            train_loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        # report performance
        acc = (100 * correct / total)
        train_acc_return.append(acc)
        train_loss_every_epoch = np.average(train_loss_list)
        train_loss_return.append(train_loss_every_epoch)
        print('----------Epoch{:2d}/{:2d}----------'.format(epoch+1,num_epoch))
        print('Train set | Loss: {:6.4f} | Accuracy: {:4.2f}% '.format(train_loss_every_epoch, acc))
        
        # Evaluate after every epochh
        correct = 0
        total = 0
        model.eval()
        predictions = []
        truths = []
        val_loss_list = []
        with torch.no_grad():
            for i, (sample) in enumerate(val_loader):
                sample_1 = sample['x1'].to(device)
                sample_2 = sample['x2'].to(device)
                labels = sample['y'].to(device)
                outputs = model(sample_1,sample_2)
                loss = loss_fn(outputs, labels)
                val_loss_list.append(loss.item())
                pred = outputs.data.max(-1)[1]
                predictions += list(pred.cpu().numpy())
                truths += list(labels.cpu().numpy())
                total += labels.size(0)
                correct += (pred == labels).sum()
            # report performance
            acc = (100 * correct / total)
            val_acc_return.append(acc)
            val_loss_every_epoch = np.average(val_loss_list)
            val_loss_return.append(val_loss_every_epoch)
            if acc > best_acc:
                best_acc = acc
                best_model_wts = model.state_dict()
            save_model(model,train_loss_return,train_acc_return,val_loss_return,val_acc_return,best_model_wts)
            elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
            print('Test set | Loss: {:6.4f} | Accuracy: {:4.2f}% | time elapse: {:>9}'\
                  .format(val_loss_every_epoch, acc,elapse))
    return model,train_loss_return,train_acc_return,val_loss_return,val_acc_return,best_model_wts

def save_model(model,train_loss_return,train_acc_return,val_loss_return,val_acc_return,best_model_wts):
    state = {'best_model_wts':best_model_wts, 'model':model, \
             'train_loss':train_loss_return, 'train_acc':train_acc_return,\
             'val_loss':val_loss_return, 'val_acc':val_acc_return}
    torch.save(state, 'checkpoint_CNN.pt')
    return None
    
model = MyModel().to(device)
train(model)
    