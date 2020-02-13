import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
# print(data.head())

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
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# print(x_train, y_test)

# K Nearest Neighbor:
# K is known as a hyper parameter. It stands for the amount of neighbors
# that we're going to look for.
numNeighbor = 9
model = KNeighborsClassifier(n_neighbors = numNeighbor) # Takes in the number of neighbors; The K value

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)
predicted =  model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(x_test)):
    print("Predicted: ", names[predicted[x]], " Data: ", x_test[x], " Actual: ", names[y_test[x]])
    # Get the distance between data points
    n = model.kneighbors([x_test[x]], numNeighbor, True)
    print("N: ", n)



