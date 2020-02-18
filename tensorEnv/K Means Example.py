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
# To score the model use a function from sklearn to compute many
# different scores from different parts of our model
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

# Training the model
# Create the model with K Means classifier
clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)

# Using Matplotlib to visualize

