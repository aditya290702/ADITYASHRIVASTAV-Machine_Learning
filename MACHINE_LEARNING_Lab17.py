import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def feature_mapping(x1, x2):
    feature1 = x1**2
    feature2 = np.sqrt(2 * x1 * x2)
    feature3 = x2**2
    return feature1, feature2, feature3

# Data points
data_points = [
    (1, 13),    (1, 18),    (2, 9),    (3, 6),    (6, 3),    (9, 2),    (13, 1),    (18, 1),    (3, 15),    (6, 6),    (6, 11),    (9, 5),    (10, 10),    (11, 5),    (12, 6),    (16, 3)]

# Transform data points into 3D space
transformed_points = [feature_mapping(x1, x2) for x1, x2 in data_points]
transformed_points = np.array(transformed_points)

# Labels
labels = ['Blue'] * 8 + ['Red'] * 8

# Fit SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(transformed_points, labels)

# Get separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-100, 100)
yy = a * xx - (clf.intercept_[0]) / w[1]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = {'Blue': 'blue', 'Red': 'red'}
for point, label in zip(transformed_points, labels):
    ax.scatter(point[0], point[1], point[2], c=colors[label])

# Plot separating plane
ax.plot(xx, yy, color='green', linestyle='--')

# Set labels
ax.set_xlabel('X1^2')
ax.set_ylabel('sqrt(2 * X1 * X2)')
ax.set_zlabel('X2^2')

plt.show()
