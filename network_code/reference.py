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

def cwrotate(img):
    out=cv2.transpose(img)
    out=cv2.flip(out,flipCode=0)
    return out

def ccwrotate(img):
    out=cv2.transpose(img)
    out=cv2.flip(out,flipCode=1)
    return out

model_path = '../model.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("===> Building model")
model = RRDBNet()
model = nn.DataParallel(model, device_ids=[0])
model.to(device)#.to(device, dtype=torch.float)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'].state_dict())

lr_image = glob.glob('../dataset/testLR/*.png')
result_dir = ('../results/RRDB/')

for i in range(len(lr_image)):
    print(i)
    lrimg = cv2.imread(lr_image[i])
    (w,h,_) = np.shape(lrimg)
    output = np.zeros((w*16,h*16,3))
    x = 0; y = 0
    while x<w:
        y = 0
        while y<h:
            lr_img1 = lrimg[x:x+100,y:y+100,:]
            lr_img1 = np.swapaxes(lr_img1, 0, 2)/255.0
            lr_img1 = torch.tensor([lr_img1], device=device).float()
            sr_img1 = model(lr_img1)
            sr_img1 = sr_img1.cpu().data.numpy()
            sr_img1 = np.swapaxes(sr_img1[0], 0, 2)
            
            lr_img2 = lrimg[x:x+100,y:y+100,:]
            lr_img2 = cwrotate(lr_img2)
            lr_img2 = np.swapaxes(lr_img2, 0, 2)/255.0
            lr_img2 = torch.tensor([lr_img2], device=device).float()
            sr_img2 = model(lr_img2)
            sr_img2 = sr_img2.cpu().data.numpy()
            sr_img2 = np.swapaxes(sr_img2[0], 0, 2)
            sr_img2 = ccwrotate(sr_img2)
            
            lr_img3 = lrimg[x:x+100,y:y+100,:]
            lr_img3 = cwrotate(cwrotate(lr_img3))
            lr_img3 = np.swapaxes(lr_img3, 0, 2)/255.0
            lr_img3 = torch.tensor([lr_img3], device=device).float()
            sr_img3 = model(lr_img3)
            sr_img3 = sr_img3.cpu().data.numpy()
            sr_img3 = np.swapaxes(sr_img3[0], 0, 2)
            sr_img3 = ccwrotate(ccwrotate(sr_img3))
            
            lr_img4 = lrimg[x:x+100,y:y+100,:]
            lr_img4 = cwrotate(cwrotate(cwrotate(lr_img4)))
            lr_img4 = np.swapaxes(lr_img4, 0, 2)/255.0
            lr_img4 = torch.tensor([lr_img4], device=device).float()
            sr_img4 = model(lr_img4)
            sr_img4 = sr_img4.cpu().data.numpy()
            sr_img4 = np.swapaxes(sr_img4[0], 0, 2)
            sr_img4 = ccwrotate(ccwrotate(ccwrotate(sr_img4)))
            
            sr_img = np.clip((sr_img1+sr_img2+sr_img3+sr_img4)/4, 0, 1)
            sr_img = sr_img * 255
            
            output[(x+5)*16:(x+100)*16,(y+5)*16:(y+100)*16] = sr_img[5*16:,5*16:]
            
            if (x==0):
                output[x*16:(x+5)*16,(y+5)*16:(y+100)*16] = sr_img[:5*16,5*16:]
            if y==0:
                output[(x+5)*16:(x+100)*16,y*16:(y+5)*16] = sr_img[5*16:,:5*16]
            if (x==0)and(y==0):
                output[x*16:(x+5)*16,y*16:(y+5)*16] = sr_img[:5*16,:5*16]
            y = y + 90
        x = x + 90
    cv2.imwrite(lr_image[i].replace('../dataset/validationLR/','../results/RRDB_simple_new/'),output)
