# cross domain object detection
[TOC]

针对目标检测任务中训练集和测试集图像分布不一致，训练集训练的模型在测试集上泛化性能不强的问题。一个直观的解决方案就是通过对测试集打标签并进行训练，但这种方法费时费力，在很大程度上不具备可实施性。

>该项实验首先通过对训练集的图像进行风格迁移，利用风格迁移后的图像进行训练，以求提高模型泛化性能。

## 数据集
利用公开数据集[BDD100k](https://bair.berkeley.edu/blog/2018/05/30/bdd/)，由于数据集需通过国外网站下载，速度较慢，故提供[百度云盘下载](https://pan.baidu.com/s/1QqpkOAlsx75YMiBLJGnohw)，密码为：bhnu；该数据集图片大小为1280\*720pixels，包含一天不同时间段的汽车驾驶路况图片，并且每张图片包括如下几个属性：
* 标注：bus,traffic light,traffic sign,person,bike,truck,motor,car,train,rider
* 时间段：daytime,night,dawn/dusk
* 天气：rainy,snowy,clear,overcast,partly cloudy,foggy
* 场景：tunnel,residential,parking lot,city street,gas stations,highway
> 本项实验选择的标注对象为car，以car为过滤条件，选择6000张包含car对象的图片，其中3000张为daytime时间段，另3000张为night时间段，以1500张图片为一个集合，将数据集划分为$day_{train}$、$day_{test}$、$night_{train}$、$night_{test}$四个数据集。

## 实验
实验分为3项：
1. 实现cycleGan模型，以$day_{train}$为源域，$night_{train}$为目标域，对$day_{train}$进行风格迁移生成$fakeNight_{train}$。由于分割迁移并不改变其内容位置信息，故其对应的标注信息与$day_{train}$一致；
2. 实现fasterRcnn模型，划分5个不同训练集，分别为：$day_{train}$、$day_{train}+fakeNight_{train}$、$day_{train}+night_{train}$、$fakeNight_{train}$、$night_{train}$，每个训练集重复训练10个模型，一共50个模型。测试集分为$day_{test}+night_{test}$、$night_{test}$，利用训练好的模型对测试集进行测试，指标为$mAP$；
3. 实验效果评估与展示；
   
### cycleGan
1. 训练cycleGan
````
bash workflow/cycleGAN_train.sh
````
> 由于cycleGan模型是一个高密集参数训练模型，较一般的深度模型而言，GPU的使用量高1～2个数量级，故对原始1280\*720的图片采用随机裁减到256\*256，加快训练速度。训练记录可见**output/log/cycleGAN_100_20_256_1.txt**
2. 生成$fakeNight_{train}$
````
bash workflow/cycleAN_test.sh
````
> 在测试阶段，由于速度较快，故图片以1280\*720的大小输入，这样输出的图片大小与输入一致，原数据的标注信息也可直接使用。**此阶段图片能以训练阶段不同的大小输入，是由于cycleGan模型采用全卷积**；

