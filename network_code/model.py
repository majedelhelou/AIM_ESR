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
import random
from imageio import imread
import glob
import cv2
import os
import torch.nn as nn
import math

# checkpoint
def save_checkpoint(model, epoch, name):
    model_dir = "../model/"
    model_out_path = "%s/%s_epoch_%d.pth" % (model_dir, name, epoch)
    state = {"epoch": epoch ,"model": model, "name": name}
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    
# dataset
class Hdf5Dataset(data.Dataset):
    def __init__(self, lrname='../dataset/lr.h5', hrname='../dataset/hr.h5'):
        super(Hdf5Dataset, self).__init__()      
        self.hr_dataset = h5py.File(hrname)['/data']
        self.lr_dataset = h5py.File(lrname)['/data']
        #self.transform = transforms.Compose([
        #transforms.RandomVerticalFlip(),
        #transforms.RandomHorizontalFlip()
        #transforms.RandomRotation([0, 90, 180, 270])
        #])

    def __getitem__(self, index):
        hr_img = (self.hr_dataset[index]/255.0)
        lr_img = (self.lr_dataset[index]/255.0)
        
        #seed = np.random.randint(2147483647) # make a seed with numpy generator 
        #random.seed(seed) # apply this seed to img tranfsorms
        #if self.transform is not None:
        #    hr_img = self.transform(hr_img)
            
        #random.seed(seed) # apply this seed to target tranfsorms
        #if self.transform is not None:
        #    lr_img = self.transform(lr_img)
        return hr_img.astype('float'), lr_img.astype('float')

    def __len__(self):
        return self.hr_dataset.shape[0]

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=128, nb=20, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv4(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out    