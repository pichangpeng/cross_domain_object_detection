import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')


from torchvision.models.detection import fasterrcnn_resnet50_fpn
from model.datasets import ImageDatasetFasterRcnn,collate_fn_fasterRcnn

import argparse
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import time
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--experiment_name', type=str,default='day', help='the name of the experiment')
parser.add_argument('--imagesRoot', type=str,nargs='+', default=['output/images/cycleGAN/1_720_1280_1/fake_night'], help='set of root directory of the images')
parser.add_argument('--labelsRoot', type=str,nargs='+', default=['data/labels/train/day.json'], help='root directory of the labels')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--n_experiment', type=int, default=1, help='experimental batch ')
opt = parser.parse_args()
print(opt)

fasterRcnn=fasterrcnn_resnet50_fpn(num_classes=2,trainable_backbone_layers=5)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model=nn.DataParallel(fasterRcnn)
fasterRcnn.to(device)
optimizer_fasterRcnn= torch.optim.Adam(fasterRcnn.parameters(), lr=opt.lr)


transforms_ = [ transforms.ToTensor()]
dataloader = DataLoader(ImageDatasetFasterRcnn(opt.imagesRoot,opt.labelsRoot, transforms_=transforms_),
                        batch_size=opt.batchSize, shuffle=False,drop_last=True,collate_fn=collate_fn_fasterRcnn(device))

txt=open('./output/log/fasterRcnn_%s_%d.txt'%(opt.experiment_name,opt.n_experiment), 'w')

batches_epoch=len(dataloader)
mean_period=0
start_time=time.time()
prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        optimizer_fasterRcnn.zero_grad()
        mean_period += (time.time() - prev_time)
        prev_time = time.time()
        
        optimizer_fasterRcnn.zero_grad()
        output=fasterRcnn(batch["images"], batch["targets"])
        loss=0
        for loss_name in output:
            loss+=output[loss_name]
        loss.backward()
        optimizer_fasterRcnn.step()
        
        result={}
        for name in output:
            result[name]=output[name].item()
           
        batches_done = batches_epoch*epoch + i+1
        batches_left = batches_epoch*(opt.n_epochs - epoch+1) + batches_epoch - i-1
        print('\rEpoch %03d/%03d [%04d/%04d] -- ' % (epoch+1, opt.n_epochs, i+1, batches_epoch))
        print(result)
        print('ETA: %s' % (datetime.timedelta(seconds=batches_left*mean_period/batches_done))+"\n")
        txt.write('\rEpoch %03d/%03d [%04d/%04d] --' % (epoch+1, opt.n_epochs, i+1, batches_epoch))
        txt.write(str(result))
        txt.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*mean_period/batches_done))+"\n")
    torch.save(fasterRcnn.state_dict(), 'output/weight/fasterRcnn_%s_%d.pth'%(opt.experiment_name,opt.n_experiment))
end_time=time.time()
print("all used time:%s"%datetime.timedelta(seconds=end_time-start_time))
txt.write("all used time:%s"%datetime.timedelta(seconds=end_time-start_time))
txt.close()
        