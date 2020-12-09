import warnings
warnings.filterwarnings('ignore')

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
import torchvision.models as models


device = torch.device('cuda')


def train_transform(sample,padding=200):
    pad_num = np.random.randint(padding)
    sample = torch.cat((torch.zeros((1,pad_num)),sample),-1)
    specgram = torchaudio.transforms.Spectrogram()(sample)
    specgram = (specgram+1e-10).log2()[0,:,:]
    specgram = transforms.Normalize((0.5),(0.5))(specgram.unsqueeze(0)).squeeze(0)
    specgram = torch.stack((specgram,specgram,specgram),dim=0)
    return specgram

def test_transform(sample):
    specgram = torchaudio.transforms.Spectrogram()(sample)
    specgram = (specgram+1e-10).log2()[0,:,:]
    specgram = transforms.Normalize((0.5),(0.5))(specgram.unsqueeze(0)).squeeze(0)
    specgram = torch.stack((specgram,specgram,specgram),dim=0)
    return specgram
    
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
            sample_1_tensor = train_transform(sample_1)
            sample_2_tensor = train_transform(sample_2)
        else:
            sample_1_tensor = test_transform(sample_1)
            sample_2_tensor = test_transform(sample_2)
            
            
        label = self.df.loc[idx,'True or False']
        
        sample = (sample_1_tensor, sample_2_tensor, label)
        return sample

def pad_collate(batch):
    (xx1,xx2, yy) = zip(*batch)
    x1_lens = [len(x[0][0]) for x in xx1]
    x2_lens = [len(x[0][0]) for x in xx2]
    x1_max_len = np.max(x1_lens)
    x2_max_len = np.max(x2_lens)
    
    xx1_new = torch.zeros([len(xx1),xx1[0].size(0),xx1[0].size(1),x1_max_len],dtype=torch.float)
    for i in range(len(xx1)):
        xx1_new[i,:,:,:len(xx1[i][0][0])] = xx1[i]
        
    xx2_new = torch.zeros([len(xx2),xx2[0].size(0),xx2[0].size(1),x2_max_len],dtype=torch.float)
    for i in range(len(xx2)):
        xx2_new[i,:,:,:len(xx2[i][0][0])] = xx2[i]
    
    yy_new = torch.tensor(yy, dtype=torch.long)
    
    sample = {'x1':xx1_new,'x2':xx2_new,'y':yy_new}
    return sample
    
    
    
train_df_path = '/scratch/cz2064/myjupyter/Time_Series/notebook/train.csv'
val_df_path = '/scratch/cz2064/myjupyter/Time_Series/notebook/val.csv'
test_df_path = '/scratch/cz2064/myjupyter/Time_Series/notebook/test.csv'
BATCH_SIZE = 8
train_sampler = torch.utils.data.sampler.RandomSampler(my_dataset(train_df_path,train = True)\
                                                       ,num_samples=50000,replacement=True)
train_loader = DataLoader(my_dataset(train_df_path,train = True), batch_size=BATCH_SIZE, \
                          sampler = train_sampler,num_workers=16,collate_fn = pad_collate)

val_loader = DataLoader(my_dataset(val_df_path), batch_size=BATCH_SIZE, shuffle=True,\
                        num_workers=16,collate_fn = pad_collate)

test_loader = DataLoader(my_dataset(test_df_path), batch_size=1, shuffle=True)


# resnet18 and remove the final two layers
resnet18 = models.resnet18(pretrained=True)
CNN_model = nn.Sequential(*(list(resnet18.children())[:-1]))
CNN_model[8] = nn.AvgPool2d(7,stride=1)

class LSTM(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128):
        super(LSTM, self).__init__()
        self.lstm_1 = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        x = x[:, -1, :]
        return x
RNN_model = LSTM()


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn = CNN_model
        self.rnn = RNN_model
        self.fc1 = nn.Linear(128*3,1024)
        self.activation_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(1024,128)
        self.activation_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(128,2)
    
    def forward(self, x1, x2):
        x1 = self.cnn(x1)
        x2 = self.cnn(x2)
        
        x1 = x1.squeeze(2)
        x1 = torch.transpose(x1,1,2)
        x2 = x2.squeeze(2)
        x2 = torch.transpose(x2,1,2)
        
        x1 = self.rnn(x1)
        x2 = self.rnn(x2)
        
        x_add = x1+x2
        x_minus = x1-x2
        x_multiply = x1*x2
        x = torch.cat((x_add, x_minus, x_multiply),-1)
        x = self.fc1(x)
        x = self.activation_fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.activation_fc2(x)
        x = self.fc3(x)
        return x
        
        
        
def train(model, train_loader=train_loader, val_loader=val_loader, learning_rate=1e-4, num_epoch=100):
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
    torch.save(state, 'checkpoint_CNN_LSTM.pt')
    return None
    
    
model = MyModel().to(device)
train(model)