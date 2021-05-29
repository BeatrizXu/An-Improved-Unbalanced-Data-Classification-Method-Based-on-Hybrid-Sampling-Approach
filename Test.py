samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]

from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy import linalg as LA
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances
m = np.arange(8).reshape(2,2,2)
n = LA.norm(m)
v_hat = m / n
a = np.array([[1.16601855]])
b = np.array([0.41520116])
d = euclidean_distances(a[0], b)
c = np.random.rand(a)
print(c)


