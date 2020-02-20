import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# This data set contains 60,000 images of different clothing/apparel items.
# Our goal for our network is to look at these images and classify them appropriately.
data = keras.datasets.fashion_mnist


# Split the data into training and testing data.
# By doing this, we can test the accuracy of the model of data
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Define a list off the class names and pre-process images.
# Divide each image by 255.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0