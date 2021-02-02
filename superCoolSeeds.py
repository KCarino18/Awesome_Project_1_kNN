from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import cm
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d   # must keep
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

seeds = pd.read_csv('seeds_dataset.csv')
seedNames = dict(zip(seeds.seedType.unique(),[ "Kama", "Rosa", "Canadian"]))
print(seedNames)
X = seeds[["area","perimeter","compactness","lengthOfKernel","widthOfKernel","asymmetryCoefficient",]]
y = seeds["seedType"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['area'], X_train['asymmetryCoefficient'], X_train['compactness'], c = y_train, marker = '$f$', s=100)
ax.set_xlabel('area')
ax.set_ylabel('asymmetryCoefficient')
ax.set_zlabel('compactness')
plt.show()

knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')

knn.fit(X_train, y_train)

knn.score(X_test, y_test)

k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k, weights = 'distance')
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20])
plt.show()

t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
knn = KNeighborsClassifier(n_neighbors=5, weights = 'distance')
plt.figure()
for s in t:
    scores = []
    for i in range(1, 1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy')
plt.show()