from torchvision.models.detection import fasterrcnn_resnet50_fpn
from model.datasets import ImageDatasetGan

import argparse
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--imagesRoot', type=str, default='data/images/test/rainy', help='root directory of the images')
parser.add_argument('--labelsRoot', type=str, default='data/labels/test/rainy.json', help='root directory of the labels')
parser.add_argument('--ssd_weight', type=str, default='output/weight/ssd.pth', help='root directory of weight')
opt = parser.parse_args()
print(opt)

# class BatchCollator:
#     def __call__(self, batch):
#         transposed_batch = list(zip(*batch))
#         images = default_collate(transposed_batch[0])
#         name = default_collate(transposed_batch[2])
#         list_targets = transposed_batch[1]
#         targets = Container(
#             {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
#         )
#         return images, targets, img_ids

    
ssd=fasterrcnn_resnet50_fpn(num_classes=2,trainable_backbone_layers=5,pretrained_backbone=False)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model=nn.DataParallel(ssd)
ssd.to(device)
ssd.load_state_dict(torch.load(opt.ssd_weight))
ssd.eval()

transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
dataloader = DataLoader(ImageDatasetGan(opt.imagesRoot,opt.labelsRoot, transforms_=transforms_),
                        batch_size=opt.batchSize, shuffle=False)

if not os.path.exists("./metric/ssd/ground_truth"):
    os.makedirs("./metric/ssd/ground_truth")
if not os.path.exists("./metric/ssd/detection_results"):
    os.makedirs("./metric/ssd/detection_results")

for i, batch in enumerate(dataloader):
    predictions = ssd(batch["images"].to(device))
    
    name=batch["name"][0].split("/")[-1][:-4]
    ture_boxes=batch["targets"]["boxes"].squeeze(0).numpy()
    pred_boxes=predictions[0]["boxes"].cpu().detach().numpy()
    scores=predictions[0]["scores"].cpu().detach().numpy()
    
    txt1=open('./metric/ssd/ground_truth/%s.txt'%name, 'w')
    txt2=open('./metric/ssd/detection_results/%s.txt'%name, 'w')
    
    for j in range(len(ture_boxes)):
        txt1.write("car %.2f %.2f %.2f %.2f"%(ture_boxes[j][0],ture_boxes[j][1],ture_boxes[j][2],ture_boxes[j][3])+"\n")
    txt1.close()
    
    for j in range(len(pred_boxes)):
        txt2.write("car %.4f %.2f %.2f %.2f %.2f"%(scores[j],pred_boxes[j][0],pred_boxes[j][1],pred_boxes[j][2],pred_boxes[j][3])+"\n")
    txt2.close()
    
    print(i)
    print(predictions)
    if i==5:
        break
        