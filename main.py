import random

import datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

#make sure everything is deterministic
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu")

df, load_data = datasets.megc("cropped")

uv_frames = np.load("data/megc_uv_frames.npy")
uv_frames = resize(uv_frames, (uv_frames.shape[0], 3, 60, 60))


le = LabelEncoder()
labels = le.fit_transform(df["emotion"])
dataset = le.fit_transform(df["dataset"])

class MEData(Dataset):
    def __init__(self, frames, labels, dataset, transform=None):
        self.frames = frames
        self.labels = labels
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return self.frames.shape[0]
    
    def __getitem__(self, idx):
        sample = self.frames[idx, ...]
        if self.transform:
            sample = self.transform(sample)
        label = self.labels[idx]
        dataset = self.dataset[idx]
        return sample, label, dataset

class Net(nn.Module):
    def __init__(self, output_size, dropout):
        super(Net, self).__init__()
        h1 = 14
        h2 = 28
        h3 = 128
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=h1, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bn1 = nn.BatchNorm2d(h1)
        self.drop1 = nn.Dropout2d(dropout)
        
        self.conv2 = nn.Conv2d(in_channels=h1, out_channels=h2, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(h2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(dropout)

        self.fc1 = nn.Linear(8 ** 2 * h2, h3)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(h3, 3)
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        x = self.drop1(self.bn1(self.pool(F.relu(self.conv1(x)))))
        x = self.drop2(self.bn2(self.pool2(F.relu(self.conv2(x)))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(self.drop(x))
        x = self.softmax(x)
        return x


def LOSO(features, df, epochs=200, lr=0.01, batch_size=128, dropout=0.5, weight_decay=0.001,
         verbose=True):
    outputs_list = []
    #groupby reorders elements, now the labels are in same order as outputs
    df_groupby = pd.concat([i[1] for i in df.groupby("subject")])
    dataset_groupby = df_groupby["dataset"]
    
    le = LabelEncoder()
    labels = le.fit_transform(df["emotion"])
    labels_groupby = le.transform(df_groupby["emotion"])

    #loop over each subject
    for group in df.groupby("subject"):
        subject = group[0]
        #split data to train and test based on the subject index
        train_index = df[df["subject"] != subject].index
        X_train = features[train_index, :]
        y_train = labels[train_index]
        dataset_train = dataset[train_index]
        
        test_index = df[df["subject"] == subject].index
        X_test = features[test_index, :]
        y_test = labels[test_index]
        dataset_test = dataset[test_index]

        #create pytorch dataloaders from the split
        megc_dataset_train = MEData(X_train, y_train, dataset_train, None)
        dataset_loader_train = torch.utils.data.DataLoader(megc_dataset_train,
                                                             batch_size=batch_size, shuffle=True,
                                                             num_workers=0)

        megc_dataset_test = MEData(X_test, y_test, dataset_test, None)
        dataset_loader_test = torch.utils.data.DataLoader(megc_dataset_test,
                                                         batch_size=100, shuffle=False,
                                                         num_workers=0)

        
        net = Net(df["emotion"].nunique(), dropout).float().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        net.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in dataset_loader_train:
                data_batch, labels_batch = batch[0].to(device), batch[1].to(device)

                optimizer.zero_grad()
                
                outputs = net(data_batch.float())
                loss = criterion(outputs, labels_batch.long())
                loss.backward()
                optimizer.step()

        #Test model
        net.eval()
        data_batch_test, labels_batch_test, _ = dataset_loader_test.__iter__().__next__()
        data_batch_test, labels_batch_test = data_batch_test.to(device), labels_batch_test.to(device)
        outputs = net(data_batch_test.float())
        _, prediction = outputs.max(1)
        prediction = prediction.cpu().data.numpy()
        outputs_list.append(prediction)
        
        train_outputs = net(data_batch.float())
        _, train_prediction = train_outputs.max(1)
        train_prediction = train_prediction.cpu().data.numpy()
        train_f1 = f1_score(labels_batch.cpu().data.numpy(), train_prediction, average="macro")
        test_f1 = f1_score(labels_batch_test.cpu().data.numpy(), prediction, average="macro")
        
        
        #Print statistics
        if verbose:
            print("Subject: {}, n={} | train_f1: {:.5f} | test_f1: {:.5}".format(
                subject, str(labels_batch_test.shape[0]).zfill(2), train_f1, test_f1))
            
    outputs = np.concatenate(outputs_list)
    f1_total = f1_score(labels_groupby, outputs, average="macro")
    idx = dataset_groupby == "smic"
    f1_smic = f1_score(labels_groupby[idx], outputs[idx], average="macro")
    idx = dataset_groupby == "casme2"
    f1_casme2 = f1_score(labels_groupby[idx], outputs[idx], average="macro")
    idx = dataset_groupby == "samm"
    f1_samm = f1_score(labels_groupby[idx], outputs[idx], average="macro")
    print("Total f1: {}, SMIC: {}, CASME2: {}, SAMM: {}".format(f1_total, f1_smic, f1_casme2, f1_samm))



LOSO(uv_frames, df, epochs=200, lr=0.01, weight_decay=0.001,
     dropout=0.5, batch_size=128)
