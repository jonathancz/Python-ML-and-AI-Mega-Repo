import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# Loading the data sets
# Use the scale function to scale our data down.
# Convert the large values that are contained as features into
# range between -1 and 1 to simplify calculations and make training
# easier and more accurate
digits = load_digits()
data = scale(digits.data)
y = digits.target

k = 10
samples, features = data.shape

# Scoring
# To score the model use a function from sklearn to compute a

