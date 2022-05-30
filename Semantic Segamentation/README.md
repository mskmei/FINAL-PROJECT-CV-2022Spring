## <div align="center">Introduction</div>
This project is aimed to test and compare several pretrained semantic segmentation models. We will use the pretrained model to perform semantic segmentation on a specific video and output the semantically segmented video. In this project, we will first extract the video frame by frame. Then perform semantic segmentation processing on each frame, and finally use the obtained results to generate the resulting video.

Note that the experiments in this project are absolutely carried out on **colab**, and we will provide you with links that connects to our work on colab. You can see the specific output and view videos for example on colab.

## <div align="center">About models</div>
In this project, we used three well-known models in the field of semantic segmentation: PSPNet, PSANet, and Deeplabv3. If you want to understand the network structure of these models and other information, you can read our experimental report or refer to related papers.

As for code, the code used in this experiment is mainly forked and modified from some open-source code. We mainly realize the frame-by-frame extraction of images and the re-splicing of videos. In the process of testing, we mainly used the "git clone+link" command to obtain the required files. For example, you can use the following command to get the relevant files needed to build PSPNet and PSANet:
```bash
git clone https://github.com/mskmei/pspnet-pasnet.git
```

Since multiple models are used in this project, and the focus of the project is on testing, rather than discussing complex network architectures, loss functions used in training, etc., it is not necessary to list complicated codes here. If you want to see the detailed code for building these models, you can click on the following link: [PSPNet](https://github.com/mskmei/pspnet-pasnet.git),  [PSANet](https://github.com/mskmei/pspnet-pasnet.git), [Deeplabv3](https://github.com/mskmei/DeepLabV3Plus-Pytorch.git). Note that these links point to my own repo, they all have some differences from the original code. 

Furthermore, these three models are all pretrained on the Cityscapes dataset. You can get the pretrained weights from following links:[pretrained pspnet](https://drive.google.com/file/d/1KaM0XLX60awJ6VzEPel84jr5AouNoLxi/view?usp=sharing), [pretrained psanet](https://drive.google.com/file/d/1qqktI7aIEM4Vucqk7XVi-L2FFDLJEQtO/view?usp=sharing), [pretrained deeplabv3](https://drive.google.com/file/d/1XlP8CzbkVkv8UZ2f6_0wtCT8P0Pu4nmD/view?usp=sharing). We tested the three models on a video that is not part of the Cityscapes dataset to observe and compare the transferability of the three pretrained models on different datasets. Of course, due to the choice of this test method, the effect will naturally not be very good. We speculate that its segmentation effect will be much worse than its segmentation effect on the Cityscapes dataset.
