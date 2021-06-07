import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False):
        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.transform2 = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, 'clear' ) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'rainy' ) + '/*.*'))

    def __getitem__(self, index):
        item_A=Image.open(self.files_A[index % len(self.files_A)])
        item_A_orig=self.transform1(item_A)
        item_A_trans = self.transform2(item_A)
        
        if self.unaligned:
            item_B = self.transform2(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform2(Image.open(self.files_B[index % len(self.files_B)]))

        return {"clear_orig":item_A_orig,'clear': item_A_trans, 'rainy': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))