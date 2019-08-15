import sys
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import functools
from functools import partial
import numpy as np
from imageio import imread
import glob
import cv2
import os
import torch.nn as nn
import math

from model import save_checkpoint, Hdf5Dataset, RRDBNet

def adjust_learning_rate(initial_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 10))
    return lr

num_workers = 1
batch_size = 16
initial_lr = 0.0001
start_epoch = 0
nEpochs = 10000

train_set = Hdf5Dataset()
training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("===> Building model")
model = RRDBNet()#.to(device, dtype=torch.float)
model = nn.DataParallel(model,device_ids=[0,1,2])
model.to(device)
criterion = nn.MSELoss()

for epoch in range(start_epoch, nEpochs + 1):
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    lr = adjust_learning_rate(initial_lr, optimizer, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    model.train()
    

    for iteration, batch in enumerate(training_data_loader, 1):
        x_data, z_data = Variable(batch[0].float()).cuda(), Variable(batch[1].float()).cuda()
        output = model(z_data)
        loss = criterion(output, x_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.item()))
            save_checkpoint(model, epoch, 'simple')
    save_checkpoint(model, epoch, 'simple')