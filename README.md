# DA-Net
This is a Pytorch implementation of IEEE Access paper "DA-Net: Learning the fine-grained density distribution with deformation aggregation network". 

<!-- ![](https://github.com/BigTeacher-xyx/DA-Net/blob/master/pictures/whole.gif) -->
## Enviroment
[![python](https://img.shields.io/badge/python-2.7.12-brightgreen.svg)]()
[![pytorch](https://img.shields.io/badge/pytorch-0.3.1-blue.svg)]()

## Getting Started
### Data Preparation
| Datasets | Method | 
| :----:   | :----: |
| ShanghaiTech Part A | Geometry-adaptive kernels |
| ShanghaiTech Part B | Normal Fixed kernel: σ = 4|
|    UCSD   | Normal Fixed kernel: σ = 4|
|    The WorldExpo’10 | Perspective |
|    UCF_CC_50 | Geometry-adaptive kernels| 
|    TRANCOS   | Normal Fixed kernel: σ = 4 |

For ShanghaiTech Part A and UCF_CC_50, use the code in "data_preparation/geometry-kernel"; For The WorldExpo’10, use the code in "data_preparation/perspective"; For UCSD and TRANCOS, use the code in "data_preparation/normal". In geometry-kernel, we augment the data by cropping 100 patches that each of them is 1/4 size of the original image. In perpective, we augment the data by cropping 10 patches that each of them is size of 256*256. In normal, data enhancement is not performed.

### Run
* Train: python train.py
a. Set pretrained_vgg16 = False
b. Set fine_tune = False
* Test: python test.py
a. Set save_output = True to save output density maps