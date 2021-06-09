from torchvision.models.detection import fasterrcnn_resnet50_fpn
from model.datasets import ImageDatasetSSD

import argparse
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--imagesRoot', type=str, default='data/images/test/rainy', help='root directory of the images')
parser.add_argument('--labelsRoot', type=str, default='data/labels/test/rainy.json', help='root directory of the labels')
parser.add_argument('--ssd_weight', type=str, default='output/weight/ssd.pth', help='root directory of weight')
opt = parser.parse_args()
print(opt)
  
ssd=fasterrcnn_resnet50_fpn(num_classes=3,trainable_backbone_layers=5)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model=nn.DataParallel(ssd)
ssd.to(device)
ssd.load_state_dict(torch.load(opt.ssd_weight))
ssd.eval()

transforms_ = [ transforms.ToTensor()]
dataloader = DataLoader(ImageDatasetSSD(opt.imagesRoot,opt.labelsRoot, transforms_=transforms_),
                        batch_size=opt.batchSize, shuffle=False,drop_last=True)

# if not os.path.exists("./output/metric/ssd/ground_truth"):
#     os.makedirs("./output/metric/ssd/ground_truth")
# if not os.path.exists("./metric/ssd/detection_results"):
#     os.makedirs("./output/metric/ssd/detection_results")
# if not os.path.exists("./output/metric/ssd/images_optional"):
#     os.makedirs("./output/metric/ssd/images_optional'")

for i, batch in enumerate(dataloader):
    predictions = ssd(batch["images"].to(device))
    name=batch["name"][0].split("/")[-1][:-4]
    ture_boxes=batch["targets"]["boxes"].squeeze(0).numpy()
    pred_boxes=predictions[0]["boxes"].cpu().detach().numpy()
    scores=predictions[0]["scores"].cpu().detach().numpy()
    
#     save_image(batch["images_orig"].data, './output/metric/ssd/images_optional/%s.jpg' % name)
#     txt1=open('./output/metric/ssd/ground_truth/%s.txt'%name, 'w')
#     txt2=open('./output/metric/ssd/detection_results/%s.txt'%name, 'w')
    
#     for j in range(len(ture_boxes)):
#         txt1.write("car %.2f %.2f %.2f %.2f"%(ture_boxes[j][0],ture_boxes[j][1],ture_boxes[j][2],ture_boxes[j][3])+"\n")
#     txt1.close()
    
#     for j in range(len(pred_boxes)):
#         txt2.write("car %.4f %.2f %.2f %.2f %.2f"%(scores[j],pred_boxes[j][0],pred_boxes[j][1],pred_boxes[j][2],pred_boxes[j][3])+"\n")
#     txt2.close()
    
    print(i)
    print(predictions)
    if i==5:
        break
        