import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

# With non numerical data, we have to convert them to numerical data
# If we examine the car data set, you can realize that most of the data
# is non numerical. So with Sklearn, we can use preprocessing to convert
# these non numerical values to integer values.
# For example, low = 0, medium = 1, high = 2

# Use Sklearn to preprocess
le = preprocessing.LabelEncoder()

# Create list from the array (preprocessing takes in a list as a parameter)
# All these return a numpy array
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
safety = le.fit_transform(list(data["safety"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
cls = le.fit_transform(list(data["class"]))
# Display example
# print(buying)

predict = "class"

# Create one big list
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
print(x_train, y_test)





