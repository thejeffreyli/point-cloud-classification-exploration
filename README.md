# Exploration and Assessment of Parametric and Non-Parametric Networks for 3D Point Cloud Classification

By Jeffrey Li and Ning-Yuan Li

Last Updated: December 16, 2023

## Project Summary

Processing 3D point cloud data is important for applications in self-driving cars, robotics, and virtual and augmented reality. Qi et al. (2017) introduced the PointNet architecture which performs classification tasks after being trained on point cloud data. The group based their design decisions on three properties of point clouds, namely permutation invariance, transformation invariance, and interactions among points. On the other hand, Zhang et al. (2023) offers a newer non-parametric approach in solving the same problem. Non-parametric building blocks are stacked across multiple stages to construct a pyramid hierarchy. We will be exploring and evaluating both models. Firstly, we focused on implementing both networks and testing them on a smaller point cloud dataset based off ModelNet40. Secondly, we performed a series of tests on robustness on PointNet, as we augment the data during inference. Lastly, we will visually assess the internal encodings of the multistep hierarchical layers inside the non-parametric encoder of the Point-NN to understand how it captures spatial representations for classification tasks.

## Content

- **papers:** contains the original papers that we base our project from
    - [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](./papers/PointNet.pdf)
    - [Parameter is Not All You Need: Starting from Non-Parametric Networks for 3D Point Cloud Analysis](./papers/PointNN.pdf)

- **pointnet:** contains relevant code for PointNet model
    - data_loader.py: load point cloud data
    - eval_cls.py: assess model accuracy on validation set
    - models.py: implementation of PointNet in PyTorch, including variants
    - utils.py: used for saving models during checkpoints
    - train.py: training and validation loops 
    - Visualize-PointNet-Results.ipynb: visualizing plots and results from robustness testing

- **pointnn:** contains relevant code for PointNN model
    - fps.py: performs farthest point sampling on point cloud data
    - knn.py: performs k-nearest neighbors classification on point cloud data

- **report:** contains final report submission

## Program and System Requirements

We developed this primarily using Python, PyTorch, and CUDA GPU.

Packages to Install:

```
pip install matplotlib
pip install numpy
pip install torch
pip install argparse
pip install torchsummary
pip install torchviz
pip install scikit-learn
pip install tensorboard
```

## Model Architecture

![PointNet](/assets/img/pointnet.jpg)


<div style="display: flex; justify-content: space-between;">
    <img src="/assets/img/pointnn01.png" alt="NPE" width="300" style="margin-right: 10px;">
    <img src="/assets/img/pointnn02.png" alt="PMB" width="300"/>

</div>

## Sample Point Clouds for Common Household Objects 

| Image| SKU | Model |
|----------|----------|------------------------------|
| ![Image 1](/assests/img/sample/555088-105.jpg) | 555088-105 | Jordan 1 Retro High Dark Mocha |
| ![Image 2](/assests/img/sample/555088-126.jpg) | 555088-126 | Jordan 1 Retro High Light Smoke Grey |
| ![Image 3](/assests/img/sample/555088-500.jpg) | 555088-500 | Jordan 1 Retro High Court Purple White |
| ![Image 4](/assests/img/sample/555088-711.jpg) | 555088-711 | Jordan 1 Retro High OG Taxi
| ![Image 5](/assests/img/sample/DC1788-100.jpg) | DC1788-100 | Jordan 1 Retro High CO.JP Midnight Navy (2020) |
| ![Image 6](/assests/img/sample/DD9335-641.jpg) | DD9335-641 | Jordan 1 Retro High OG Atmosphere (Women's) |
| ![Image 7](/assests/img/sample/DO7097-100.jpg) | DO7097-100 | Jordan 1 Retro High OG A Ma Maniére |
| ![Image 8](/assests/img/sample/DZ5485-031.jpg) | DZ5485-031 | Jordan 1 Retro High OG Lucky Green |
| ![Image 9](/assests/img/sample/DZ5485-400.jpg) | DZ5485-400 | Jordan 1 Retro High OG UNC Toe |
| ![Image 10](/assests/img/sample/DZ5485-612.jpg) | DZ5485-612 | Jordan 1 Retro High OG Chicago Lost and Found |

## Results

![augmentation](/assests/img/augmentation.png)

Training image samples following augmentation.

![acc_curve](/assests/img/acc_curve.png)

![loss_curve](/assests/img/loss_curve.png)

Validation accuracy curves for three experiments:
- Experiment 1 (67.4%): only RELU activation and MaxPool layer following convolutional layer
- Experiment 2 (78.1%): introduced Batch Normalization prior to each RELU activation layer
- Experiment 3 (86.0%): introduced data augmentation techniques for training set, namely Normalization, RandomRotation, and RandomHorizontalFlip 

Validation loss curves for three experiments:
- Experiment 1 (1.26)
- Experiment 2 (0.750)
- Experiment 3 (0.585)

![augmentation](/assests/img/confusion_matrix.png)

Confusion Matrix of classification results on validation set.