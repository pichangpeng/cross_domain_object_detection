import glob
import random
import os
import json
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

class ImageDatasetGAN(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False):
        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.transform2 = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, 'clear' ) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'rainy' ) + '/*.*'))
#         root_split=root.split("/")
#         root_split[2]="labels"
#         self.labl_root="/".join(root_split)
#         with open(os.path.join(self.labl_root, 'clear.json'), 'r') as f:
#             self.item_A_lab = json.load(f)
#         with open(os.path.join(self.labl_root, 'rainy.json'), 'r') as f:
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
        return {'clear': item_A_trans, 'rainy': item_B,"clear_name":filename_A.split("/")[-1][:-4],"clear_orig":item_A_orig}
#         return {"clear_name":filename_A.split("/")[-1][:-4],"clear_orig":item_A_orig,'clear': item_A_trans, 'rainy': item_B,"clear_label":self.item_A_lab[filename_A.split("/")[-1][:-4]],"rainy_label":self.item_B_lab[filename_B.split("/")[-1][:-4]]}

    def __len__(self):
        return len(self.files_A)


# transforms_ = [ transforms.ToTensor(),
#                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
# a=ImageDataset("./data/images/train",transforms_ =transforms_ )
# b=a.__getitem__(1)
# print(b["clear_orig"])
# print(b['clear'])

class ImageDatasetSSD(Dataset):
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
        return {"images":item_A_trans,"images_orig":item_A_orig,"targets":{"boxes":boxes,"labels":labels},"name":filename}
    
    def __len__(self):
        return len(self.files)

    
