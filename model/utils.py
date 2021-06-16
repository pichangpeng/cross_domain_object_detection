import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch,batchSize,size,random_crop):
        # self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.batchSize=batchSize
        self.size=size
        self.epoch = 1
        self.batch = 1
        self.start_time=time.time()
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        self.txt=open('./output/log/cycleGAN_%d_%d_%d_%d.txt'%(self.n_epochs,self.batchSize,self.size,random_crop), 'w')


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))
        self.txt.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))
        
        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        
        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name]
            else:
                self.losses[loss_name] += losses[loss_name]

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/batches_done))
                self.txt.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/batches_done))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/batches_done))
                self.txt.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/batches_done))

        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))
        self.txt.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done))+"\n")

        if (self.batch % self.batches_epoch) == 0:
            self.epoch += 1
            self.batch=1
        else:
            self.batch += 1
        sys.stdout.write('\n')
        
    def close(self):
        sys.stdout.write("all used time:%s"%datetime.timedelta(seconds=time.time()-self.start_time))
        self.txt.write("all used time:%s"%datetime.timedelta(seconds=time.time()-self.start_time))
        self.txt.close()

        

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

    
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

