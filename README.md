# Exploration and Assessment of Parametric and Non-Parametric Networks for 3D Point Cloud Classification

By Jeffrey Li and Ning-Yuan Li

Last Updated: December 16, 2023

## Project Summary

Processing 3D point cloud data is important for applications in self-driving cars, robotics, and virtual and augmented reality. Qi et al. (2017) introduced the PointNet architecture which performs classification tasks after being trained on point cloud data. The group based their design decisions on three properties of point clouds, namely permutation invariance, transformation invariance, and interactions among points. On the other hand, Zhang et al. (2023) offers a newer non-parametric approach in solving the same problem. Non-parametric building blocks are stacked across multiple stages to construct a pyramid hierarchy. We will be exploring and evaluating both models. Firstly, we focused on implementing both networks and testing them on a smaller point cloud dataset based off ModelNet40. Secondly, we performed a series of tests on robustness on PointNet, as we augment the data during inference. Lastly, we will visually assess the internal encodings of the multistep hierarchical layers inside the non-parametric encoder of the Point-NN to understand how it captures spatial representations for classification tasks.

## Content

- **create_dataset.py:** creates image datasets in your local directory
    - Each type of sneaker model will have its own directory named after the SKU.
    - Each image in the SKU directory will be numbered.
    - The images are then divided into training and testing files. The default is 70:30 split.  

- **dataloader.py:** creates a separate dataloader for the training or testing image datasets using the [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html) class

- **image-scraping-driver.py:** image-scraping Selenium driver for Google Images 
    - You will need to download the correct [ChromeDriver](https://chromedriver.chromium.org/downloads) for your version of Google Chrome for this driver to work. 
    - Resources/References:
        - [Automating Google Chrome to Scrape Images with Selenium and Python](https://www.youtube.com/watch?v=7KhuEsq-I8o)
        - [A Beginner’s Guide to Image Scraping with Python and Selenium](https://medium.com/@nithishreddy0627/a-beginners-guide-to-image-scraping-with-python-and-selenium-38ec419be5ff)

- **model-training-assessment.ipynb:** contains training and validation loops and qualitative and quantiative assessments 

- **model.py:** contains the architecture of CNN 

## Program and System Requirements

We developed this primarily on Google Colab, using Python, PyTorch, and CUDA GPU.

Packages to Install:

```
pip install torch
pip install torchvision
pip install pillow
pip install matplotlib
pip install numpy
pip install pandas
pip install seaborn
pip install scikit-learn
pip install tqdm
pip install selenium
pip install requests
pip install opencv-python
```

## Model Architecture

| Layer  | Operation                                 | Input Size          | Output Size         |
|--------|-------------------------------------------|---------------------|---------------------|
| Conv1  | Conv2d(3, 32, kernel_size=3, stride=1, pad=1)| (224, 224, 3)       | (224, 224, 32)      |
| BN1    | BatchNorm2d(32)                            | (224, 224, 32)      | (224, 224, 32)      |
| ReLU   | ReLU()                                     | (224, 224, 32)      | (224, 224, 32)      |
| Pool1  | MaxPool2d(kernel_size=2, stride=2)         | (224, 224, 32)      | (112, 112, 32)      |
| Conv2  | Conv2d(32, 64, kernel_size=3, stride=1, pad=1)| (112, 112, 32)     | (112, 112, 64)      |
| BN2    | BatchNorm2d(64)                            | (112, 112, 64)      | (112, 112, 64)      |
| ReLU   | ReLU()                                     | (112, 112, 64)      | (112, 112, 64)      |
| Pool2  | MaxPool2d(kernel_size=2, stride=2)         | (112, 112, 64)      | (56, 56, 64)        |
| Conv3  | Conv2d(64, 128, kernel_size=3, stride=1, pad=1)| (56, 56, 64)       | (56, 56, 128)       |
| BN3    | BatchNorm2d(128)                           | (56, 56, 128)       | (56, 56, 128)       |
| ReLU   | ReLU()                                     | (56, 56, 128)       | (56, 56, 128)       |
| Pool3  | MaxPool2d(kernel_size=2, stride=2)         | (56, 56, 128)       | (28, 28, 128)       |
| Flatten| Flatten()                                  | (28, 28, 128)       | 100352              |
| FC1    | Linear(100352, 256)                       | 100352              | 256                 |
| ReLU   | ReLU()                                     | 256                 | 256                 |
| FC2    | Linear(256, 10)                           | 256                 | 10                  |

## Example Search Queries on Google Images

- Query Example #1: 
    - Jordan 1 Retro High OG Chicago Lost and Found 
    - DZ5485-612
    - Air Jordan 1 Lost and Found Ebay
- Query Example #2: 
    - Jordan 1 Retro High Dark Mocha
    - 555088-105
    - Air Jordan 1 Dark Mocha Ebay

## Sample Images 

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