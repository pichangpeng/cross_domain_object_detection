from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from model.datasets import ImageDatasetSSD,collate_fn_SSD

import argparse
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import time
import datetime

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--imagesRoot', type=str, default='output/images/cycleGAN/1_720_1280_1/fake_B', help='root directory of the images')
parser.add_argument('--labelsRoot', type=str, default='data/labels/test/clear.json', help='root directory of the labels')
opt = parser.parse_args()
print(opt)

anchor_sizes = ((32),(64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

ssd=fasterrcnn_resnet50_fpn(num_classes=2,trainable_backbone_layers=5,rpn_nms_thresh=0.7, box_detections_per_img=100,rpn_anchor_generator=rpn_anchor_generator)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model=nn.DataParallel(ssd)
ssd.to(device)

transforms_ = [ transforms.ToTensor()]
dataloader = DataLoader(ImageDatasetSSD(opt.imagesRoot,opt.labelsRoot, transforms_=transforms_),
                        batch_size=opt.batchSize, shuffle=False,drop_last=True,collate_fn=collate_fn_SSD(device))

txt=open('./output/log/ssd.txt', 'w')
batches_epoch=len(dataloader)
mean_period=0
start_time=time.time()
prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        mean_period += (time.time() - prev_time)
        prev_time = time.time()
        output=ssd(batch["images"], batch["targets"])
        
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
    torch.save(ssd.state_dict(), 'output/weight/ssd.pth')
end_time=time.time()
print("all used time:%s"%datetime.timedelta(seconds=end_time-start_time))
txt.write("all used time:%s"%datetime.timedelta(seconds=end_time-start_time))
txt.close()
        