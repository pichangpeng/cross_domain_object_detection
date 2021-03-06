#!/usr/bin/python3

import argparse
import itertools
import sys,os
from PIL import Image
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from model.cycleGan import Generator
from model.cycleGan import Discriminator
from model.utils import ReplayBuffer
from model.utils import LambdaLR
from model.utils import Logger
from model.utils import weights_init_normal
from model.datasets import ImageDatasetGAN,collate_fn_GAN

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--labelroot', type=str, default='', help='root directory of the label')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=5, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--random_crop', type=bool, default=False, help='True if random crop on origin image,False if first resize then random crop ')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)
fasterRcnn=fasterrcnn_resnet50_fpn(num_classes=2,trainable_backbone_layers=5)


if torch.cuda.device_count() > 1:
    netG_A2B=nn.DataParallel(netG_A2B,device_ids=[2,3,1])
    netG_B2A=nn.DataParallel(netG_B2A,device_ids=[2,3,1])
    netD_A=nn.DataParallel(netD_A,device_ids=[2,3,1])
    netD_B=nn.DataParallel(netD_B,device_ids=[2,3,1])
    fasterRcnn=nn.DataParallel(fasterRcnn,device_ids=[2,3,1])
netG_A2B.to(device)
netG_B2A.to(device)
netD_A.to(device)
netD_B.to(device)
fasterRcnn.to(device)

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G_fasterRcnn = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters(),fasterRcnn.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# optimizer_fasterRcnn= torch.optim.Adam(fasterRcnn.parameters(), lr=opt.lr)

lr_scheduler_G_fasterRcnn = torch.optim.lr_scheduler.LambdaLR(optimizer_G_fasterRcnn, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0).unsqueeze(dim=1), requires_grad=False).to(device)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0).unsqueeze(dim=1), requires_grad=False).to(device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
if opt.random_crop:
    transforms_ = [ transforms.RandomCrop(opt.size), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
else:
    transforms_ = [ transforms.Resize(int(opt.size*1.12)), 
                    transforms.RandomCrop(opt.size), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDatasetGAN(opt.dataroot,opt.labelroot,transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, drop_last=True,collate_fn=collate_fn_GAN(device))

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader),opt.batchSize,"GF")
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['day'][0])).to(device)
        real_A_orig = batch['day_orig']
        real_B = Variable(input_B.copy_(batch['night'][0])).to(device)
        target=batch["targets"]

        ###### Generators A2B and B2A and fasterRcnn######
        optimizer_G_fasterRcnn.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
        
        # fasterRcnn loss
        fake_B_orig = 0.5*(netG_A2B(real_A_orig[0].unsqueeze(0)).data + 1.0)
        output=fasterRcnn([fake_B_orig.squeeze(0)], batch["targets"])
        loss_fasterRcnn=0
        for loss_name in output:
            loss_fasterRcnn+=output[loss_name]
        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB+sum(loss_fasterRcnn)
        loss_G.backward()
        
        optimizer_G_fasterRcnn.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        
        ######fasterRcnn##########
#         optimizer_fasterRcnn.zero_grad()
#         with torch.no_grad():
#             fake_B = 0.5*(netG_A2B(real_A_orig[0].unsqueeze(0)).data + 1.0)
#         output=fasterRcnn([fake_B.squeeze(0)], batch["targets"])
#         loss_fasterRcnn=0
#         for loss_name in output:
#             loss_fasterRcnn+=output[loss_name]
#         loss_fasterRcnn.backward(loss_fasterRcnn.clone().detach())
#         optimizer_fasterRcnn.step()
        ###################################
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B),'loss_fasterRcnn':sum(loss_fasterRcnn)}, 
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
    # Update learning rates
    lr_scheduler_G_fasterRcnn.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output/weight/netG_A2B_GF.pth')
    torch.save(netG_B2A.state_dict(), 'output/weight/netG_B2A_GF.pth')
    torch.save(netD_A.state_dict(), 'output/weight/netD_A_GF.pth')
    torch.save(netD_B.state_dict(), 'output/weight/netD_B_GF.pth')
    torch.save(fasterRcnn.state_dict(), 'output/weight/fasterRcnn_GF.pth')
###################################
logger.close()


