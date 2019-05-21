# Self-Driving Car Engineer Nanodegree

This repository contains projects from Self-Driving Car Engineer Nanodegree course on Udacity. 

## Projects

- Lane Lines Detector
- Traffic Light Classifier
- Behavioral Cloning
- Advanced Lane Finding




## Lane Lines Detector

The goal of this first project was to **create a simple pipeline to detect road lines** in a frame taken from a roof-mounted camera.
A short demo [video](http://www.youtube.com/watch?feature=player_embedded&v=KlQ-8iD1EFM)  can be found here.

This project does not include any machine learning, just **old school computer vision techniques**. 

Image processing pipeline goes something like this:

1. Convert images to grayscale.
2. Darken the images with **Gamma correction** method.
3. Convert the original image to HSL and isolate white and yellow mask.
4. Combine masks using OR operation and then combine them with original image using AND operation.
5. Apply **Gaussian blur** to an image with kernel size 5.
6. Use **Canny Edge Detection** algorithm to detect edges.
7. Define the area of interest on the image. (Lower part due to camera position)
8. Apply **Hough Transform** technique to extract lines.

![Final resault](https://i.ytimg.com/vi/EZcHGsPX55Y/maxresdefault.jpg)


## Traffic Light Classifier

The goal of this project was to build a **CNN** in [TensorFlow](https://www.tensorflow.org/) to **classify traffic sign images** from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

For image preprocessing, I used standard methods such as grayscaling and normalization. I also used **CLAHE (Contrast Limited Adaptive Histogram Equalization)** because the distribution of sign images was not ideal. 
 
As proposed in the lab I implemented two network architectures, [LeNet5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) and [VGG16](https://arxiv.org/pdf/1409.1556.pdf), and compared their results. 

#### LeNet5
I trained LeNet architecture for **50 epochs**, with **batch size 128** and it gave **97% accuracy** on validation set. 

<img src="https://github.com/mmilunovic/sdc-udacity/blob/master/images/lenet_acc.png" width="400"/> <img src="https://github.com/mmilunovic/sdc-udacity/blob/master/images/lenet_loss.png" width="400"/> 

#### VGG16ish
