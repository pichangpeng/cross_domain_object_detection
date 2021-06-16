import glob
import random
import os
import json
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate


class collate_fn_GAN():
    def __init__(self,device):
        self.device=device
    def __call__(self,batch):
        day=[]
        day_orig=[]
        night=[]
        targets=[]
        day_name=[]
        for data in batch:
            day.append(data["day"].to(self.device))
            day_orig.append(data["day_orig"].to(self.device))
            targets.append({"boxes":data["targets"]["boxes"].to(self.device),"labels":data["targets"]["labels"].to(self.device)})
            day_name.append(data["day_name"])
            night.append(data["night"])
        return {"day":day,"day_orig":day_orig,"targets":targets,"day_name":day_name,"night":night}
    

class ImageDatasetGAN(Dataset):
    def __init__(self, imagetRoot,labelRoot=None, transforms_=None, unaligned=False):
        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.transform2 = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(imagetRoot, 'day' ) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(imagetRoot, 'night' ) + '/*.*'))
        self.labelRoot=labelRoot
        if self.labelRoot:
            with open(labelRoot, 'r') as f:
                self.lab = json.load(f)
    def __getitem__(self, index):
        filename_A=self.files_A[index % len(self.files_A)]
        item_A=Image.open(filename_A)
        item_A_orig=self.transform1(item_A)
        item_A_trans = self.transform2(item_A)
        if self.unaligned:
            filename_B=self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            filename_B=self.files_B[index % len(self.files_B)]
        item_B = self.transform2(Image.open(filename_B))
        if self.labelRoot:
            target=self.lab[filename_A.split("/")[-1][:-4]]
            boxes=torch.FloatTensor(target["boxes"])
            labels=torch.tensor(target["labels"])
            return {'day': item_A_trans, 'night': item_B,"day_name":filename_A.split("/")[-1][:-4],"day_orig":item_A_orig,"targets":{"boxes":boxes,"labels":labels}}
        else:
            return {'day': item_A_trans, 'night': item_B,"day_name":filename_A.split("/")[-1][:-4],"day_orig":item_A_orig}

    def __len__(self):
        return len(self.files_A)


# transforms_ = [ transforms.ToTensor(),
#                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
# a=ImageDataset("./data/images/train",transforms_ =transforms_ )
# b=a.__getitem__(1)
# print(b["day_orig"])
# print(b['day'])

class ImageDatasetFasterRcnn(Dataset):
    def __init__(self,imagesRoot,labelRoot,transforms_=None):
        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.transform2 = transforms.Compose(transforms_)
        self.files=[]
        self.labs=[]
        for imageroot in imagesRoot:
            self.files+=glob.glob(imageroot+'/*.*')
        self.files = sorted(self.files)
        for labelroot in labelRoot:
            with open(labelroot, 'r') as f:
                lab = json.load(f)
            self.labs.append(lab)
        
    def __getitem__(self,index):
        filename=self.files[index % len(self.files)]
        item_A=Image.open(filename)
        item_A_orig=self.transform1(item_A)
        item_A_trans=torch.FloatTensor(self.transform2(item_A))
        name=filename.split("/")[-1][:-4]
        for lab in self.labs:
            if name in lab:
                target=lab[name]
                boxes=torch.FloatTensor(target["boxes"])
                labels=torch.tensor(target["labels"])
                break
            continue    
        return {"images":item_A_trans,"images_orig":item_A_orig,"targets":{"boxes":boxes,"labels":labels},"name":name}
    
    def __len__(self):
        return len(self.files)

class collate_fn_fasterRcnn():
    def __init__(self,device):
        self.device=device
    def __call__(self,batch):
        images=[]
        images_orig=[]
        targets=[]
        name=[]
        for data in batch:
            images.append(data["images"].to(self.device))
            images_orig.append(data["images_orig"].to(self.device))
            targets.append({"boxes":data["targets"]["boxes"].to(self.device),"labels":data["targets"]["labels"].to(self.device)})
            name.append(data["name"])
        return {"images":images,"images_orig":images_orig,"targets":targets,"name":name}
    
# transforms_ = [ transforms.ToTensor()]
# a=ImageDatasetSSD('output/images/cycleGAN/1_720_1280_1/fake_B','data/labels/test/day.json',transforms_ =transforms_ )
# b=a.__getitem__(1)
# print(b)
# dataloader = DataLoader(a,batch_size=2, shuffle=False,drop_last=True,collate_fn=collate_fn_SSD())
# for i, batch in enumerate(dataloader):
#     print(batch)
#     break
    
