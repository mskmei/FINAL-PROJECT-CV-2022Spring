# <div align="center">Comparison of Three Different Ways of Training Faster RCNN</div>

## Introduction
We have taken on three ways to train Faster RCNN:

(a) train the network with all the parameters randomly initialized;

(b) use the parameters of Resnet pretrained on ImageNet to initialize the backbone, and then use VOC to fine tune;

(b) use the backbone network parameters of Mask R-CNN pretrained on COCO, 
initialize the backbone network of Faster R-CNN, and then use VOC for fine tune;

## Installation

**1.** Clone the FasterRCNN repository.

git clone [https://github.com/mskmei/FINAL-PROJECT-CV-2022Spring.git](https://github.com/mskmei/FINAL-PROJECT-CV-2022Spring.git)

**2.** Build a new environment with conda.
Our environment is a Linux system with CUDA version == 11.4.
```bash
conda create -n faster python==3.8
conda activate faster
```

**3.** Install the needed packages.
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tensorflow
pip install opencv-python
cd FasterRCNN
pip install -r requirements.txt
```


## Dataset Preparation
 **1.** Download the training, validation, test data and VOCdevkit

	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

**2.** Extract all of these tars into one directory named `VOCdevkit`

	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar

**3.** It should have this basic structure

  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...

   
   
   
## Train
Before trainig the model, make sure you have changed the root_path in train.py.
Train the model with this line of command:
```bash
python train.py
```
The code is default to use GPU when training, predicting, and testing, so if you use CPU to train the model, use this line of command:
```bash
python train.py --cuda False
```
Besides, considering that our implementation has two backbone to choose, the default is Resnet50, so if you want to train the model with VGG as the backbone (with GPU), use this line of command:
```bash
python train.py --backbone vgg
```

## Test
Before testing the model, make sure you have changed the root_path in test.py.
Test the trained model with this line of command:
```bash
python test.py --weights --path-to/your trained model
```
The default model is our pretrained model **model_data/resnet50_faster.pth** with the Resnet50 as the backbone.

## Predict
Also, before predicting, change the root_path in predict.py.
You can use the image **FasterRCNN/img/street.jpg** to have a look of your trained model using
```bash
python predict.py --weights --path-to/your trained model --img --path-to/FasterRCNN/img/street.jpg
```

## Pretrained Model
There are four pretrained model you can download through Google Drive:
Our pretrained model:

vgg_faster.pth https://drive.google.com/file/d/1niTR2cD5di8_-JfQyPEpgeyHfGGyCiUr/view?usp=sharing

resnet50_faster https://drive.google.com/file/d/1Ujds2mvsNLc8cXHH6827S-K_SfmV0Ssg/view?usp=sharing

The pretrained model of Resnet50 and VGG for training:

resnet50-19c8e357.pth https://drive.google.com/file/d/1Rq9Qk5xaphA56VTD2ysNvps6Ij2SPQbl/view?usp=sharing

vgg16-397923af.pth https://drive.google.com/file/d/1Dq3F3lrX4Wwk-QEy-nNH1K7ECPVOrum8/view?usp=sharing
