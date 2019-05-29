
## Traffic Light Classifier

The goal of this project was to build a **CNN** in [TensorFlow](https://www.tensorflow.org/) to **classify traffic sign images** from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

For image preprocessing, I used standard methods such as grayscaling and normalization. I also used **CLAHE (Contrast Limited Adaptive Histogram Equalization)** because the distribution of sign images was not ideal. 
 
As proposed in the lab I implemented two network architectures, [LeNet5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) and [VGG16](https://arxiv.org/pdf/1409.1556.pdf), and compared their results. 

#### LeNet5
I trained LeNet architecture for **50 epochs**, with **batch size 128** and it gave **97% accuracy** on validation set. 

<img src="https://github.com/mmilunovic/sdc-udacity/blob/master/images/lenet_acc.png" width="400"/> <img src="https://github.com/mmilunovic/sdc-udacity/blob/master/images/lenet_loss.png" width="400"/> 

#### VGG16ish
