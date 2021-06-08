#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from torch import nn

from model.cycleGan import Generator
from model.datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./data/images/test', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--x_size', type=int, default=720, help='x size of the data ')
parser.add_argument('--y_size', type=int, default=1280, help='y size of the data ')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/weight/netG_A2B.pth', help='A2B generator checkpoint file')
# parser.add_argument('--generator_B2A', type=str, default='output/weight/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
# netG_B2A = Generator(opt.output_nc, opt.input_nc)

if torch.cuda.device_count() > 1:
    netG_A2B=nn.DataParallel(netG_A2B)
#     netG_B2A=nn.DataParallel(netG_B2A)
    
netG_A2B.to(device)
# netG_B2A.to(device)
# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
# netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
# netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.x_size, opt.y_size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.x_size, opt.y_size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('output/images/cycleGAN/%d_%d_%d_%s'%(opt.batchSize,opt.x_size,opt.y_size,opt.generator_A2B.split('_')[-1][0])):
    os.makedirs('output/images/cycleGAN/%d_%d_%d_%s/real_A/'%(opt.batchSize,opt.x_size,opt.y_size,opt.generator_A2B.split('_')[-1][0]))
    os.makedirs('output/images/cycleGAN/%d_%d_%d_%s/fake_B/'%(opt.batchSize,opt.x_size,opt.y_size,opt.generator_A2B.split('_')[-1][0]))

for i, batch in enumerate(dataloader):
    # Set model input
    real_A_name=batch['clear_name'][0]
    real_A_orig=Variable(input_A.copy_(batch['clear_orig']))
    real_A = Variable(input_B.copy_(batch['clear']))

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)

    # Save image files
    save_image(real_A_orig.data, 'output/images/cycleGAN/%d_%d_%d_%s/real_A/%s.png' % (opt.batchSize,opt.x_size,opt.y_size,opt.generator_A2B.split('_')[-1][0],real_A_name))
    save_image(fake_B, 'output/images/cycleGAN/%d_%d_%d_%s/fake_B/%s.png' % (opt.batchSize,opt.x_size,opt.y_size,opt.generator_A2B.split('_')[-1][0],real_A_name))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
