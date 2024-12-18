# Multiclass classification model using a custom convolutional neural network in TensorFlow

The project is part of the Module Assignment for the Deep Learning Module. 
The project's intent is to utilize Deep Learning concepts, particularly Neural Networks, to solve a problem with a real-world dataset.
The assignment is developed in Python using **Keras Framework** which provides a rich library of functions to create custom **Convolutional Neural Networks**. 


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

## General Information

In this assignment, we will build a multiclass classification model using TensorFlow's custom convolutional neural network. 

### Background 
The project aims to build a CNN-based model that can accurately detect melanoma. 
Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. 
A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

### Dataset
The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed by the International Skin Imaging Collaboration (ISIC).
All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, except for melanomas and moles, whose images are slightly dominant.
The data set contains the following diseases:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

### CNN Specification - Baseline

- Layers
    - Convolution with Features Size = 32
    - MaxPooling with Filter Size = 2 and Stride Size = 2
    - Convolution with Features Size = 64
    - MaxPooling with Filter Size = 2 and Stride Size = 2
    - Dense with Neurons = 128
- Activation Functions
    - RELU for Convolutional Layers
    - SOFTMAX for Output Layer
- Optimizer Function
    - adam
- Loss Function
    - sparse_categorical_crossentropy  

## Conclusions
### Baseline performance

The initial baseline performance for the network mentioned above (i.e. without any hyperparameter tunings).
The model indicates **overfitting** with a wide gap between the training and validation accuracies.

|Training|Validation|
|-|-|
|90 %| 50 %|
### Effect of Dropout Layers

Introducing a dropout of **25%** after each of the Convolutional Layer and the Output Layer.
The dropout **reduces the overfitting** by way of reducing the training accuracy. 
However, there is no increase in the validation accuracy.

|Training|Validation|
|-|-|
|73 %| 53 %|

### Effect of Batch Normalisation Layers

Introducing a Batch Normalisation Layer after each of the Convolutional Layer and the Output Layer.
There is **no significant impact** of Batch Normalisation on either the training or validation accuracies as compared to the baseline model.

|Training|Validation|
|-|-|
|92 %| 45 %|

### Effect of Augmentation Layer

Introducing an Augmentation Layer at the beginning of the Network to introduce more noise or diversified datasets to limit overfitting.
The augmentation layer performs:
    - Randomized horizontal and vertical flips
    - Randomized rotation 
    - Randomized zoom
    - Randomized Contrast

The model seems to be **converging for both the Training and Validation Accuracies**. 
However, on the downside both accuracies drop drastically.

|Training|Validation|
|-|-|
|56 %| 54 %|

### Effect of Class Balancing via Augmentation

- Analysis of the data reveals class imbalance between the 9 classes:

| Disease                     | Count |
|-----------------------------|-------|
| Pigmented Benign Keratosis  | 462   |
| Melanoma                    | 438   |
| Basal Cell Carcinoma        | 376   |
| Nevus                       | 357   |
| Squamous Cell Carcinoma     | 181   |
| Vascular Lesion             | 139   |
| Actinic Keratosis           | 114   |
| Dermatofibroma              | 95    |
| Seborrheic Keratosis        | 77    |

| Training | Validation | Total |
|-|-|-|
|1792| 447 | 2239 |

After performing generating 500 images of each class using the Augmentor framework.

| Condition                   | Count |
|-----------------------------|-------|
| Melanoma                    | 500   |
| Pigmented Benign Keratosis  | 500   |
| Nevus                       | 500   |
| Basal Cell Carcinoma        | 500   |
| Actinic Keratosis           | 500   |
| Squamous Cell Carcinoma     | 500   |
| Vascular Lesion             | 500   |
| Seborrheic Keratosis        | 500   |
| Dermatofibroma              | 500   |

| Training | Validation | Total |
|-|-|-|
|5392| 1347 | 6379 |


#### Baseline model with augmented dataset

Running the baseline model without any tuning with the augmented dataset boosts the Validation Accuracy significantly.
The boost due to augmentation narrows the gap between the Training and Validation Accuracies too.
However, there is still some overfitting of the model which can be brought down by dropouts.

|Training|Validation|
|-|-|
|95 %| 73 %|

#### Baseline model with augmented dataset and dropouts

The dropout ratio of (0.25) further reduces the gap between the Training and Validation Accuracies.

**Overall this model seems to be a good fit.**

|Training|Validation|
|-|-|
|87 %| 77 %|


## Technologies Used
- tensorflow
- keras
- augmentor

## Contact
Created by [@mundrapiyush] - feel free to contact me!
