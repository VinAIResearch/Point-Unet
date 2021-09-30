#### Table of contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Training & Evaluation ](#Training_Evaluation)
4. [Result ](#result)
5. [Notes](#notes)

# <a name="introduction"></a> Point-Unet: A Context-Aware Point-Based Neural Network for Volumetric Segmentation

Point-Unet is a point-based volumetric segmentation frame- work with three main modules: the saliency attention, the context-aware sam- pling, and the point-based segmentation module. This is an implematation of the MICCAI 2021 paper Point-Unet: A Context-Aware Point-Based Neural Network for Volumetric Segmentation in Tensorflow.

![DETR](figure/flowchart.jpg)


Details of the Point-Unet model architecture and experimental results can be found in our [following paper](https://rdcu.be/cyhME). Please cite our paper when Point-Unet is used to help your research.

```
@inproceedings{ho2021point,
  title={Point-Unet: A Context-Aware Point-Based Neural Network for Volumetric Segmentation},
  author={Ho, Ngoc-Vuong and Nguyen, Tan and Diep, Gia-Han and Le, Ngan and Hua, Binh-Son},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={644--655},
  year={2021},
  organization={Springer}
}
```
## Requirements
This implenmemntation conduct on machine NVIDIA V100
* Install required python packages

```bash
$ pip install -r requirements.txt
```
* Download and organize [BraTS18](https://www.med.upenn.edu/sbia/brats2018/data.html), [BraTS19](https://www.med.upenn.edu/cbica/brats2019/data.html), [BraTS20](https://www.med.upenn.edu/cbica/brats2020/data.html) and [Pancreas](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT) dataset from its official website.
* This implementation conduct on machine NVIDIA V100

## Training code
### Context-Aware Sampling
Run command
```bash
$ python3 PointSegment/utils/data_prepare_pancreas.py
```
* generated results as format *.ply, *pkl, *.npy.s
### PointSegment
Training model
```bash
$ python3 -B  main_pancreas_step.py --gpu 0 --mode train
```

Evaluation model
```bash
$ python3 -B  main_pancreas_step.py --gpu 0 --mode test 
```

## <a name="result"></a> Result

* Offline validation set
    | Dataset | Dice ET | Dice ET | Dice ET | Average Dice | Average HD95 |
    |---------|:-------:|:-------:|:-------:|:------------:|:-------------:|
    | BraTS18 |  80.76  |  90.55  |  87.09  |     86.13    |     6.01     |
    | BraTS19 |  85.56  |  91.18  |  90.10  |     88.98    |     4.92     |
    | BraTS20 |  76.43  |  89.67  |  82.97  |     83.02    |     8.26     |

* Online validation set
    | Dataset | Dice ET | Dice ET | Dice ET | Average Dice | Average HD95 |
    |---------|:-------:|:-------:|:-------:|:------------:|:------------:|
    | BraTS18 |  80.97  |  90.50  |  84.11  |     85.19    |     6.30     |
    | BraTS19 |  79.01  |  87.63  |  79.70  |     82.11    |     10.39    |
    | BraTS20 |  78.98  |  89.71  |  82.75  |     83.81    |     11.73    |

## <a name="notes"></a> Notes
This repository is still continues update!




## License
    
    MIT License

    Copyright (c) 2021 VinAI Research

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.