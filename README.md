# Clarifruit Unet

A framework for experimenting with semantic Segmentation using 
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) and
performing color based classification on the resulting masks.
Used in this instance to perform semantic segmentation on images of cherries to
isolate the stems and classify the images into 4 quality levels after
transforming the maskes into hsv color space histograms and using them with an
Xgboost algorithm for final classification.
Part of my internship at Clarifruit


## Getting Started


### Prerequisites

Installed python 3

### Installing

Install venv requirments from requirements.txt
first create a new Virtual Environment in the same
folder containing the "Work" folder
```
py -m venv venv
```
activate the Environment:
```
venv\Scripts\activate
```
and than use pip install:
```
pip install -r requirements.txt
```
## Overview

### Data
Used on an private data set, unavailable at the moment
but in order to use the framework, the data set must have a root
folder containing one folder the images, and another for the segmentation masks
e.g
![](md_images\root_structure.PNG)

### Unet

Using a slightly modified implementation from https://github.com/zhixuhao/unet

#### Data augmentation
Using the options of the keras ImageDataGenerator augmentations.
e.g 
```
ImageDataGenerator(rotation_range=180,
                   brightness_range=[0.2, 1.],
                   shear_range=5,
                   zoom_range=0.5,
                   horizontal_flip=True,
                   vertical_flip=True,
                   fill_mode='nearest')
```

#### usage

For an example on usage see the model_training notebook
```
Give an example
```
#### Results
![drawing](md_images\67260-70372.png.jpg)


## Segmentation Augmentation

Also implemented is an approach to augment the given\ resulting segmentation
masks using Computer Vision algorithems, such as [felzenszwalb](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf)
implemented with the [skimage module](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb),

### Usage
for usage see the --------


### Results
before:
![before](md_images\38360-02397.png.jpg)
after:
![after](md_images\38360-02397.png.seg_ontop.jpg)

##Classification

The segmentation masks are used to extract the "stems" of the cherries,
which are converted to hsv histograms which are used as input to an [Xgboost]() 
classifier (on a new train test data) which results in ranked classification
###Results
An Example:
![before](md_images\stems\38360-02397.png.jpg)
after:
![after](md_images\stems\38360-02397-stem.png.jpg)
 initial classification Results:
 '''
 (              precision    recall  f1-score   support

           A       0.78      0.87      0.83        79
           B       0.65      0.62      0.64        55
           C       0.80      0.23      0.36        35
           D       0.68      0.96      0.80        47

    accuracy                           0.72       216
   macro avg       0.73      0.67      0.65       216
weighted avg       0.73      0.72      0.69       216)
'''
 
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Mentoring and guidance [Roman Mirochnik](https://www.linkedin.com/in/mrroman/)

