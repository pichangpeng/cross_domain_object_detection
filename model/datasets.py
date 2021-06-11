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

class ImageDatasetGAN(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False):
        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.transform2 = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, 'day' ) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'night' ) + '/*.*'))
#         root_split=root.split("/")
#         root_split[2]="labels"
#         self.labl_root="/".join(root_split)
#         with open(os.path.join(self.labl_root, 'day.json'), 'r') as f:
#             self.item_A_lab = json.load(f)
#         with open(os.path.join(self.labl_root, 'night.json'), 'r') as f:
#             self.item_B_lab = json.load(f)

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
    def __init__(self,imagetRoot,labelRoot,transforms_=None):
        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.transform2 = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(imagetRoot+'/*.*'))
        with open(labelRoot, 'r') as f:
            self.lab = json.load(f)
        
    def __getitem__(self,index):
        filename=self.files[index % len(self.files)]
        item_A=Image.open(filename)
        item_A_orig=self.transform1(item_A)
        item_A_trans=torch.FloatTensor(self.transform2(item_A))
        target=self.lab[filename.split("/")[-1][:-4]]
        boxes=torch.FloatTensor(target["boxes"])
        labels=torch.tensor(target["labels"])
        return {"images":item_A_trans,"images_orig":item_A_orig,"targets":{"boxes":boxes,"labels":labels},"name":filename.split("/")[-1][:-4]}
    
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
    
