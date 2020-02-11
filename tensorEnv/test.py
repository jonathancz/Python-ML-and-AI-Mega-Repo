import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Read csv
data = pd.read_csv("student-mat.csv", sep=";")

# Print the first 5 rows of the data
print(data.head())

# Only save the attributes we are interested in
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

# What are the values we are trying to predict
# What are the values that we actually want
# We want G3 which stands for their final grade
predict = "G3"
# Remember, labels are the things we want to predict

# Returns a new data frame that does not have G3
# We are going to reserve G3 to assess how accurate our predictions are based
# on our training set
X = np.array(data.drop([predict], 1))

# Save G3 into another numpy array
y = np.array(data[predict])

# Split into 4 variables
# x test
# y test
# x train
# y train

# Create 4 different arrays
# x_train will be a section of X's array
# y_train will be a section of y's array
# x_test and y_test are going to be the test data that we are going to use
# to test the accuracy of our model or algorithm
# The question is Why are we splitting the data?
# If we train the model with every single bit of data that we have, then
# it will simply just memorize patterns
x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

