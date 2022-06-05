## `<div align="center">`Introduction`</div>`

This project is about reproducing the vision transformer and training on CIFAR-100 dataset from scratch. The vision transformer is
**handcrafted**.

Reference: https://arxiv.org/abs/2010.11929

## `<div align="center">`Dataset`</div>`

In this experiment, we choose CIFAR-100 dataset to experiment on. It contains 50,000 natural images in the training dataset while 10,000 for testing. Each of them is of size 32 x 32, categorized in one of the 100 classes.

## `<div align="center">`A Quick Start`</div>`

Here we offer two notebooks on different platforms to play around. In AIStudio we implement with paddlepaddle while in Colab we implement with PyTorch. The training is light as a simple illustration of how things work. Make sure to run the notebook in GPU environment!

<div align="center">

[Open In AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/4025759?contributionType=1&shared=1)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mskmei/FINAL-PROJECT-CV-2022Spring/blob/main/VisionTransformer/notebooks/CIFAR100_VisionTransformer_torch.ipynb)`</div>`

## `<div align="center">`Training`</div>`

We have offered two entrances for training, **PaddlePaddle** and **PyTorch**. One can train with either package that he or she is familiar with.

To start with, you need to get access to our files. You can try either

```bash
git clone https://github.com/mskmei/FINAL-PROJECT-CV-2022Spring.git
cd FINAL-PROJECT-CV-2022Spring/VisionTransformer
```

or download our project as a zip and unzip it locally and turn to "VisionTransformer" directory. Then we come up with two choices.

<h2>Choice A. PaddlePaddle</h2>
<h3>Dependencies</h3>

Python 3.x

1. paddlepaddle-gpu >= 2
2. visualdl == 2.2.3
3. tqdm
4. PIL
5. numpy

<h3>Configuration</h3>

 Open **configs.py** and configure the paths, GPUs, training parameters, etc.

<h3>Train</h3>

Run **train_paddle.py**, or

```bash
python train_paddle.py
```

<br>

<h2>Choice B. PyTorch</h2>
 <h3>Dependencies </h3>

Python 3.x

1. pytorch + cuda
2. torchvision
3. tensorboard == 2.9.0
4. tqdm
5. PIL
6. numpy

<h3>Configuration</h3>

 Open **configs.py** and configure the paths, CUDAs, training parameters, etc.

<h3>Train</h3>

Run **train_torch.py**, or

```bash
python train_torch.py
```

## `<div align="center">`Testing`</div>`

We have prepared **test.py** for one to easily test a vision transformer built with out code on CIFAR-100! Just type in

```bash
python test.py
```

or manually execute the file, and the program will ask for inputting the path to your model. Both **.pdparams** and **.pth** files are supported. Then, it will ask for the index of cuda to run your model on (the process will be omitted if no cuda is available) and several network settings. But do not be afraid, the default will handle!

## `<div align="center">`Models`</div>`

 Here we also offer four already-trained models for testing, which can be downloaded at
 [https://pan.baidu.com/s/1Fkca1pHfBrN6YPAKHE8oZw?pwd=l56w](https://pan.baidu.com/s/1Fkca1pHfBrN6YPAKHE8oZw?pwd=l56w). 

| Model        | Environment  | Top-1 Acc (%) | Top-5 Acc(%) |
| ------------ | ------------ | ------------- | ------------ |
| VITsmall     | PaddlePaddle | 57.85         | 81.89        |
| VITlarge     | PaddlePaddle | 57.33         | 81.10        |
| ResVIT       | PaddlePaddle | 63.33         | 85.79        |
| ResVIT_torch | PyTorch      | 66.30         | 87.62        |

## `<div align="center">`Data Preparation`</div>`

We place this part in the final because this is NOT a must-do. The program will automatically fetch the CIFAR-100 dataset if you do not have any.

However,  if you are struck with the automatic download of the CIFAR-100 dataset (i.e. bad network connection), or you want to use a already-existed CIFAR-100 file,  this part might be helpful. The dataset can be accessed elsewhere, for example at [CIFAR](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz).

Open the **configs.py** and customize the ``data_file``. If you are working with PaddlePaddle, set `paddle_data_file` the path to **cifar-100-python.tar.gz**. If you are working with PyTorch, set `torch_data_file` to the folder that contains **cifar-100-python.tar.gz**.

 Leave `data_file   = None` as default and for automatic downloads.
