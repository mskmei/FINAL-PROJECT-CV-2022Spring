## <div align="center">Introduction</div>
This project is aimed to test and compare several pretrained semantic segmentation models. We will use the pretrained model to perform semantic segmentation on a specific video and output the semantically segmented video. In this project, we will first extract the video frame by frame. Then perform semantic segmentation processing on each frame, and finally use the obtained results to generate the resulting video.

Note that the experiments in this project are absolutely carried out on **colab**, and we will provide you with links that connects to our work on colab. You can see the specific output and view videos for example on colab. We provide example videos to help you have a better understanding of our work, the videos are available in the directory [here](https://drive.google.com/drive/folders/1pVmLX6EhfBikLpH_pUjXWYzlMmM7s8Ny?usp=sharing).

## <div align="center">About models</div>
In this project, we used three well-known models in the field of semantic segmentation: PSPNet, PSANet, and Deeplabv3. If you want to understand the network structure of these models and other information, you can read our experimental report or refer to related papers.

As for code, the code used in this experiment is mainly forked and modified from some open-source code. We mainly realize the frame-by-frame extraction of images and the re-splicing of videos. In the process of testing, we mainly used the "git clone+link" command to obtain the required files. For example, you can use the following command to get the relevant files needed to build PSPNet and PSANet:
```bash
git clone https://github.com/mskmei/pspnet-pasnet.git
```

Since multiple models are used in this project, and the focus of the project is on testing, rather than discussing complex network architectures, loss functions used in training, etc., it is not necessary to list complicated codes here. If you want to see the detailed code for building these models, you can click on the following link: [PSPNet](https://github.com/mskmei/pspnet-pasnet.git),  [PSANet](https://github.com/mskmei/pspnet-pasnet.git), [Deeplabv3](https://github.com/mskmei/DeepLabV3Plus-Pytorch.git). Note that these links point to my own repo, they all have some differences from the original code. 

Furthermore, these three models are all pretrained on the Cityscapes dataset. You can get the pretrained weights from following links:[pretrained pspnet](https://drive.google.com/file/d/1KaM0XLX60awJ6VzEPel84jr5AouNoLxi/view?usp=sharing), [pretrained psanet](https://drive.google.com/file/d/1qqktI7aIEM4Vucqk7XVi-L2FFDLJEQtO/view?usp=sharing), [pretrained deeplabv3](https://drive.google.com/file/d/1XlP8CzbkVkv8UZ2f6_0wtCT8P0Pu4nmD/view?usp=sharing). We tested the three models on a video that is not part of the Cityscapes dataset to observe and compare the transferability of the three pretrained models on different datasets. Of course, due to the choice of this test method, the effect will naturally not be very good. We speculate that its segmentation effect will be much worse than its segmentation effect on the Cityscapes dataset.

## <div align="center">Test</div>
<details open>
 <summary>Quick Start</summary>
  The 3 "ipynb" files in current directory can help you get a quick start. These three "ipynb" files are carried out on colab so you are strongly recommended to open them on colab. The files will provide you with a view of our work and you can view the example videos online(video view can only be done on colab). To open them on colab, you can simply click:
 
 PSP101:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mskmei/FINAL-PROJECT-CV-2022Spring/blob/main/Semantic%20Segamentation/PSP101_pytorch.ipynb)  
 
 PSA101:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mskmei/FINAL-PROJECT-CV-2022Spring/blob/main/Semantic%20Segamentation/PSA101.ipynb)  
 
 Deeplabv3:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mskmei/FINAL-PROJECT-CV-2022Spring/blob/main/Semantic%20Segamentation/deeplabv3.ipynb)  
 
 If you only need to test the example video we provide, then you only need to execute each line in colab in turn. If you want to test your own video or use your own weight, you just need to change the "gdown" function. It should look like:
 ```python
import gdown
gdown.download('https://drive.google.com/uc?id=your_share_id_for_video', 'use.mp4', quiet=False)
gdown.download('https://drive.google.com/uc?id=your_share_id_for_weight', 'psp101.pth', quiet=False)
 ```
Note that share id can be got from Google Drive. The following steps will introduce you how to build network to test figures but not videos.
 </details>
 
 <details open>
 <summary>PSP101 & PSA101</summary>
To build PSP101 or PSA101, first of all, let's get files from github:
 
 ```bash
 git clone https://github.com/mskmei/pspnet-pasnet.git
 ```
 
 Based on the requirements of the original open source code, we need to intall ninja. You may run code like:
 
 ```bash
 wget  https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
 sudo unzip ninja-linux.zip -d /usr/local/bin/
 sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
 ```
 
 Then make sure you have already have a pretraned weight file and a directory with images. If you don't carry out the experiment on colab, you may need to change the parameter "model_path" in the file "cityscapes_pspnet101.yaml" and "cityscapes_psanet101.yaml". It should be similar to:
 
 ```
  model_path: /path/to/psp101.pth
 ```
 
 Next run the code to test on pspnet101:
 
 ```bash
 cd pspnet-pasnet
 python testdemo.py --config=/content/pspnet-pasnet/config/cityscapes/cityscapes_pspnet101.yaml --image=/path/to/your_images TEST.scales '[1.0]'
 # if you want test on psanet101, just run the following code
 # python testdemo.py --config=/content/pspnet-pasnet/config/cityscapes/cityscapes_psanet101.yaml --image=/path/to/your_images TEST.scales '[1.0]'
 ```
 
 The results will be stored in a directory named out by default.(This can be changed by modifying testdemo.py)
 </details>
 
 <details open>
 <summary>Deeplabv3</summary>
 The same as PSP101 and PSA101, we need to getfilesfrom GitHubb first:
 
 ```bash
 git clone https://github.com/mskmei/DeepLabV3Plus-Pytorch.git
 ```
 
 Then we need to install some required packages, you may run:
 
 ```bash
 cd DeepLabV3Plus-Pytorch
 pip install -r requirements.txt
 ```
 And import required packages as:
 
 ```python
 import network
 import torch
 ```
 
 Ensure that youhaveprepared thee  needed pretrained weights and video, then load the model through:
 
 ```python
 model.load_state_dict(torch.load('/path/to/DeepLabV3Plus-Pytorch/weights.pth')['model_state'])
 ```
 
 Then simply test using the following code:
 
 ```bash
 python predict.py --inp /path/to/input_images/ --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt /path/to/weights.pth --save_val_results_to output_dir
 ```
 
 We also provide you with some example images. The following pictures are produced by PSP101, where the first one is only the result of semantic segmentation, and the second one is the image mixed with the original image in a ratio of 7:3.
<div>
<img src="samples/1psp.png"   width="40%">
<img src="samples/1mixpsp.png"  width="40%">
</div>
 
 Comparing to PSA101:
<div>
<img src="samples/1psa.png"   width="40%">
<img src="samples/1mixpsa.png"  width="40%">
</div>
 </details>
 
 ## <div align="center">Video Processing</div>
 This part will give you a brief introduction of our method to process video. First import these packages:
 
 ```python
 import os
 import shutil
 import cv2
 
 ```
 
 The next function will help you to extract the video frame by frame as a picture：
 
 ```python
 def video_2_images(video_file= './road.mp4',   # declare the target video
                   image_dir='./images/', 
                   image_file='%s.jpg'):  
 
    # Initial setting
    i = 0
    interval = 3
    length = 600  # the max of frames
    
    cap = cv2.VideoCapture(video_file)
    while(cap.isOpened()):
        flag, frame = cap.read()  
        if flag == False:  
                break
        if i == length*interval:
                break
        if i % interval == 0: 
           cv2.imwrite(image_dir+image_file % str(int(i/interval)).zfill(6), frame)
        i += 1 
    cap.release()  

# images folder reset
if os.path.isdir('images'):
    shutil.rmtree('images')
os.makedirs('images', exist_ok=True)

video_2_images()
 ```
 And the following code can help you stitch the processed pictures into a video named "output.mp4"：
 
 ```python
cd /your/directory
if os.path.exists('./output.mp4'):
   os.remove('./output.mp4')
!ffmpeg -r 10 -i /path/to/iamges/%06d.png -vcodec libx264 -pix_fmt yuv420p output.mp4
 ```
