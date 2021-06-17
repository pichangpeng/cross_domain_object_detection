import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from model.datasets import ImageDatasetFasterRcnn,collate_fn_fasterRcnn

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
parser.add_argument('--imagesRoot', type=str, nargs='+',default=['data/images/test/night'], help='root directory of the images')
parser.add_argument('--labelsRoot', type=str, nargs='+',default=['data/labels/test/night.json'], help='root directory of the labels')
parser.add_argument('--fasterRcnn_weight', type=str, default='output/weight/fasterRcnn.pth', help='root directory of weight')
parser.add_argument('--n_experiment', type=int, default=1, help='experimental batch ')
parser.add_argument('--train_name', type=str,default='day', help='the name of the train set')
parser.add_argument('--test_name', type=str,default='day', help='the name of the test set')
opt = parser.parse_args()
print(opt)

fasterRcnn=fasterrcnn_resnet50_fpn(num_classes=2,trainable_backbone_layers=5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model=nn.DataParallel(fasterRcnn)
fasterRcnn.to(device)
fasterRcnn.load_state_dict(torch.load(opt.fasterRcnn_weight))
fasterRcnn.eval()

transforms_ = [ transforms.ToTensor()]
dataloader = DataLoader(ImageDatasetFasterRcnn(opt.imagesRoot,opt.labelsRoot, transforms_=transforms_),
                        batch_size=opt.batchSize, shuffle=False,drop_last=True,collate_fn=collate_fn_fasterRcnn(device))

if not os.path.exists("./output/metric/fasterRcnn/%s_%d_to_%s/ground_truth"%(opt.train_name,opt.n_experiment,opt.test_name)):
    os.makedirs("./output/metric/fasterRcnn/%s_%d_to_%s/ground_truth"%(opt.train_name,opt.n_experiment,opt.test_name))
if not os.path.exists("./output/metric/fasterRcnn/%s_%d_to_%s/detection_results"%(opt.train_name,opt.n_experiment,opt.test_name)):
    os.makedirs("./output/metric/fasterRcnn/%s_%d_to_%s/detection_results"%(opt.train_name,opt.n_experiment,opt.test_name))
if not os.path.exists("./output/metric/fasterRcnn/%s_%d_to_%s/images_optional"%(opt.train_name,opt.n_experiment,opt.test_name)):
    os.makedirs("./output/metric/fasterRcnn/%s_%d_to_%s/images_optional"%(opt.train_name,opt.n_experiment,opt.test_name))

for i, batch in tqdm(enumerate(dataloader),desc='Processing',total=len(dataloader)):
    predictions = fasterRcnn(batch["images"])
    for j in range(len(batch["images"])):
        name=batch["name"][j]
        ture_boxes=batch["targets"][j]["boxes"].cpu().detach().numpy()
        pred_boxes=predictions[j]["boxes"].cpu().detach().numpy()
        scores=predictions[j]["scores"].cpu().detach().numpy()
    
        save_image(batch["images_orig"][j].data, './output/metric/fasterRcnn/%s_%d_to_%s/images_optional/%s.jpg' %(opt.train_name,opt.n_experiment,opt.test_name,name))
        txt1=open('./output/metric/fasterRcnn/%s_%d_to_%s/ground_truth/%s.txt'%(opt.train_name,opt.n_experiment,opt.test_name,name), 'w')
        txt2=open('./output/metric/fasterRcnn/%s_%d_to_%s/detection_results/%s.txt'%(opt.train_name,opt.n_experiment,opt.test_name,name), 'w')

        for k in range(len(ture_boxes)):
            txt1.write("car %.2f %.2f %.2f %.2f"%(ture_boxes[k][0],ture_boxes[k][1],ture_boxes[k][2],ture_boxes[k][3])+"\n")
        txt1.close()

        for k in range(len(pred_boxes)):
            txt2.write("car %.4f %.2f %.2f %.2f %.2f"%(scores[k],pred_boxes[k][0],pred_boxes[k][1],pred_boxes[k][2],pred_boxes[k][3])+"\n")
        txt2.close()
        