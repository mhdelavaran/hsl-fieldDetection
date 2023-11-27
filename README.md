# Image Landmark Detection

This project implements an image landmark detection model using TensorFlow and OpenCV.

## Overview

The goal is to detect landmark points on input images. The model is trained on a dataset of images with labeled landmark coordinates. 

The main components are:

- `cfg.py`: Configuration file containing hyperparameters and paths
- `dataManager.py`: Handles loading images, preprocessing, and batching data
- `network.py`: Defines the CNN architecture 
- `teacher.py`: Trains the model 
- `detector.py`: Loads a trained model and makes predictions on new images

## Usage

To train the model:

```
python teacher.py
```

This will automatically load data, train the model, and save the weights.

To make predictions using a trained model:

```python
from detector import Detector

model = Detector('models/my_model.h5')
predictions = model.detect('test_image.jpg') 
```

## Data

The data consists of images and corresponding landmark coordinates. The images are preprocessed to a standard square size and normalized. 

The coordinates are normalized to the range 0.0 - 1.0 to represent positions within the image.

## Model

The model is a convolutional neural network defined in `network.py`. It consists of convolutional layers, max pooling, flatten, and dense layers.

The model is trained with an MSE loss to predict the landmark coordinate values.

## Results

The model achieves a validation loss of `X` after training for `Y` epochs. Some example predictions are shown below:

![Example 1](examples/ex1.png)
![Example 2](examples/ex2.png)

## Credits

- OpenCV for image processing
- TensorFlow for model definition and training
