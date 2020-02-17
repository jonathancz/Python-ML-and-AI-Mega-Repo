import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#print(x_train, y_train)
classes = ['malignant', 'benign']

classifier = svm.SVC(kernel='linear', C=1) # Apply soft margin
classifier.fit(x_train, y_train)

# Predict some data before scoring
y_pred = classifier.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

