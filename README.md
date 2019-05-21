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
