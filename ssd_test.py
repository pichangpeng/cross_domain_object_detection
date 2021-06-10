from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from model.datasets import ImageDatasetSSD,collate_fn_SSD

import argparse
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--imagesRoot', type=str, default='data/images/test/rainy', help='root directory of the images')
parser.add_argument('--labelsRoot', type=str, default='data/labels/test/rainy.json', help='root directory of the labels')
parser.add_argument('--ssd_weight', type=str, default='output/weight/ssd.pth', help='root directory of weight')
parser.add_argument('--confient_thresh', type=float, default=0.5, help='the thresh of confient')
opt = parser.parse_args()
print(opt)

ssd=fasterrcnn_resnet50_fpn(num_classes=2,trainable_backbone_layers=5)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model=nn.DataParallel(ssd)
ssd.to(device)
ssd.load_state_dict(torch.load(opt.ssd_weight))
ssd.eval()

transforms_ = [ transforms.ToTensor()]
dataloader = DataLoader(ImageDatasetSSD(opt.imagesRoot,opt.labelsRoot, transforms_=transforms_),
                        batch_size=opt.batchSize, shuffle=False,drop_last=True,collate_fn=collate_fn_SSD(device))

if not os.path.exists("./output/metric/ssd/ground_truth"):
    os.makedirs("./output/metric/ssd/ground_truth")
if not os.path.exists("./output/metric/ssd/detection_results"):
    os.makedirs("./output/metric/ssd/detection_results")
if not os.path.exists("./output/metric/ssd/images_optional"):
    os.makedirs("./output/metric/ssd/images_optional")

for i, batch in tqdm(enumerate(dataloader),desc='Processing',total=len(dataloader)):
    predictions = ssd(batch["images"])
    for j in range(len(batch["images"])):
        name=batch["name"][j]
        ture_boxes=batch["targets"][j]["boxes"].cpu().detach().numpy()
        pred_boxes=predictions[j]["boxes"].cpu().detach().numpy()
        scores=predictions[j]["scores"].cpu().detach().numpy()
    
        save_image(batch["images_orig"][j].data, './output/metric/ssd/images_optional/%s.jpg' % name)
        txt1=open('./output/metric/ssd/ground_truth/%s.txt'%name, 'w')
        txt2=open('./output/metric/ssd/detection_results/%s.txt'%name, 'w')

        for k in range(len(ture_boxes)):
            txt1.write("car %.2f %.2f %.2f %.2f"%(ture_boxes[k][0],ture_boxes[k][1],ture_boxes[k][2],ture_boxes[k][3])+"\n")
        txt1.close()

        for k in range(len(pred_boxes)):
            if scores[k]>=opt.confient_thresh:
                txt2.write("car %.4f %.2f %.2f %.2f %.2f"%(scores[k],pred_boxes[k][0],pred_boxes[k][1],pred_boxes[k][2],pred_boxes[k][3])+"\n")
        txt2.close()
        