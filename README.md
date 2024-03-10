# Exploration and Assessment of Parametric and Non-Parametric Networks for 3D Point Cloud Classification

By Jeffrey Li and Ning-Yuan Li

Last Updated: December 16, 2023

## Project Summary

Processing 3D point cloud data is important for applications in self-driving cars, robotics, and virtual and augmented reality. Qi et al. (2017) introduced the PointNet architecture which performs classification tasks after being trained on point cloud data. The group based their design decisions on three properties of point clouds, namely permutation invariance, transformation invariance, and interactions among points. On the other hand, Zhang et al. (2023) offers a newer non-parametric approach in solving the same problem. Non-parametric building blocks are stacked across multiple stages to construct a pyramid hierarchy. We will be exploring and evaluating both models. Firstly, we focused on implementing both networks and testing them on a smaller point cloud dataset based off ModelNet40. Secondly, we performed a series of tests on robustness on PointNet, as we augment the data during inference. Lastly, we will visually assess the internal encodings of the multistep hierarchical layers inside the non-parametric encoder of the Point-NN to understand how it captures spatial representations for classification tasks.

## Sample Point Clouds for Common Household Objects 

![chairs](/assets/img/chairs.png)

![lamps](/assets/img/lamps.png)

![vases](/assets/img/vases.png)


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

## PointNet Architecture

![PointNet](/assets/img/pointnet.jpg)

The classification network takes n points as input, applies input and feature transformations, and then aggregates point features by max pooling. The output is classification scores for k classes.

## PointNN Architecture

<div style="display: flex; justify-content: space-between;">
    <img src="/assets/img/pointnn01.png" alt="NPE" width="400" style="margin-right: 10px;">
    <img src="/assets/img/pointnn02.png" alt="PMB" width="400"/>
</div>

Non-Parametric Encoder (Left). Zhang et al. (2023) utilized trigonometric functions to encode raw Point Clouds points into high-dimensional vectors in PosE. The vectors then pass through a series of hierarchical non-parametric operations, namely FPS, k-NN, local geometric aggregation, and pooling, where they will be encoded into a global feature $f_G$. Point-Memory Bank (Right). The training set features are passed through the Point-Memory Bank outputting $F_{mem}$, which is later used to classify using similarity matching.


## Results

### Accuracy and Loss Curves

![acc_curve](/assests/img/pointnetplotacc250.png)

![loss_curve](/assests/img/pointnetplotloss250.png)

PointNet Training and testing curves over 250 epochs.

### Robustness Testing for Best PointNet Model

| Number of Samples | Accuracy | Degree of Rotation | Accuracy |
|-------------------|----------|---------------------|----------|
| 10000             | 98.3%    | 0°                  | 98.3%    |
| 7500              | 98.3%    | 5°                  | 98.1%    |
| 5000              | 98.3%    | 30°                 | 73.6%    |
| 2500              | 98.1%    | 45°                 | 43.2%    |
| 1000              | 97.5%    | 90°                 | 25.3%    |
| 500               | 97.0%    |                     |          |
| 100               | 94.6%    |                     |          |


### Visualization of Point Cloud Data Encoding by FPS

![fps](/assests/img/fps.png)

### Visualization of Point Cloud Data Encoding by k-NN

![knn](/assests/img/knn.png)