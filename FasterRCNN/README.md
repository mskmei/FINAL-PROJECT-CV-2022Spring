# <div align="center">Comparison of Three Different Ways of Training Faster RCNN</div>

## Introduction
We have taken on three ways to train Faster RCNN:

(a) train the network with all the parameters randomly initialized;

(b) use the parameters of Resnet pretrained on ImageNet to initialize the backbone, and then use VOC to fine tune;

(b) use the backbone network parameters of Mask R-CNN pretrained on COCO, 
initialize the backbone network of Faster R-CNN, and then use VOC for fine tune.

And the performances are shown below:

| model | mAP | mIOU | acc |
|-------|-----|------|-----|
|ImageNet-based|0.649|0.608|0.806|
|COCO-based|0.676|0.597|0.800|
|Random-Initialized|0.125|0.161|0.356|


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
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install tensorflow
cd FINAL-PROJECT-CV-2022Spring/FasterRCNN
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

   
We use the training data and validation data of VOC2007 as our training set and use the testing data of VOC2007 as our validation set. When testing the model, we use the validation data of VOC2012 as our testing set.   
   
## Train
Before training and other operations, run **evaluation.py** first to get the ground-truth so that you can get the evaluation along the way, and run **voc_annotation.py** to get *trainSet.txt*, *valSet.txt*, and *testSet.txt*.


The code is default to use GPU when training, predicting, and testing, so if you use CPU to train the model, use this line of command:
```bash
python train.py --cuda False
```

For the three training ways:
if you want to train the network from scratch (randomly initialize the whole network)
```bash
python train.py --pretrained=False
```

or if you want to train the network with pretrained backbone on ImageNet
```bash
python train.py --pretrained=True
```

or if you want to use the pretrained backbone of Mask RCNN for initialization, uncomment the three lines (123 - 125) and comment the 119 and 120 lines in **nets/resnet50.py**, and then use the same command above.


## Test
Test the trained model with this line of command:
```bash
python test.py --weights --path-to/your trained model
```
The default model is our pretrained model **model_data/resnet50_faster.pth** with the Resnet50 as the backbone.

## Predict
You can use the image **FasterRCNN/img/street.jpg** to have a look of your trained model using
```bash
python predict.py --weights --path-to/your trained model --img --path-to/FasterRCNN/img/street.jpg
```

## Trained Model
Since our code can not load the pretrained weights downloaded from Pytorch'official directly, we have match the corresponding layers manually and create **mask.pth** to store it.

**mask.pth**  https://drive.google.com/file/d/1IuGINX7m8GgM3GkDhOXa9k92Bi20E9Pi/view?usp=sharing

There are three pretrained model you can download through Google Drive:

**resnet50_faster.pth** https://drive.google.com/file/d/1Ujds2mvsNLc8cXHH6827S-K_SfmV0Ssg/view?usp=sharing

**mask_faster.pth**     https://drive.google.com/file/d/1icZWvFdUXmKHTi719KJD_4gbl-Y73DUI/view?usp=sharing

**scratch_faster.pth**  https://drive.google.com/file/d/1BNturJuVtnZBV4J_pvCtrUSJeZrJM7aS/view?usp=sharing
