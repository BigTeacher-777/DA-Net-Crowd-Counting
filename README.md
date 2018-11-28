# DA-Net
This is a Pytorch implementation of IEEE Access paper [DA-Net: Learning the fine-grained density distribution with deformation aggregation network](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8497050). 

<!-- ![](https://github.com/BigTeacher-xyx/DA-Net/blob/master/pictures/whole.gif) -->
## Enviroment
[![python](https://img.shields.io/badge/python-2.7.12-brightgreen.svg)]()
[![pytorch](https://img.shields.io/badge/pytorch-0.3.1-blue.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-8.0-orange.svg)]()

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
1. Train: python train.py
	```Shell
	a. Set pretrained_vgg16 = False
	b. Set fine_tune = False
	```
2. Test: python test.py
	```Shell
	a. Set save_output = True to save output density maps
	```
3. pretrained model:<br>
	[[Shanghai Tech A](https://www.dropbox.com/s/h9tl5rl8gotwb5o/DA-Net_shtechA_80.h5?dl=0)]<br>
	[[Shanghai Tech B](https://www.dropbox.com/s/4c3pkha3vpw0nrg/DA-Net_shtechB_20.h5?dl=0)]<br>

 

## Cite
If you use the code, please cite the following paper:
```
@ARTICLE{8497050, 
author={Z. Zou and X. Su and X. Qu and P. Zhou}, 
journal={IEEE Access}, 
title={DA-Net: Learning the Fine-Grained Density Distribution With Deformation Aggregation Network}, 
year={2018}, 
volume={6}, 
number={}, 
pages={60745-60756}, 
keywords={Feature extraction;Strain;Kernel;Adaptation models;Diamond;Switches;Training;Crowd counting;deformable convolution;adaptive receptive fields;fine-grained density distribution}, 
doi={10.1109/ACCESS.2018.2875495}, 
ISSN={2169-3536}, 
month={},}
```
