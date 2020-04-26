# The Simpliest Neural Network
import numpy as np

x = np.array([[0,0,0,1], [1,0,1,0], [1,1,0,1], [1,1,1,1], [0,1,0,1]])
y = np.array([[0,1,1,0,1]]).T #[:, None]

w1 = np.random.random((4,5)) - 1
w2 = np.random.random((5,1)) - 1

for i in range(1000):
    h1 = 1/(1 + np.exp(-np.dot(x, w1)))
    h2 = 1/(1 + np.exp(-np.dot(h1, w2)))

    delta_h2 = (y - h2) * h2 * (1 - h2)
    delta_h1 = delta_h2.dot(w2.T) * h1 * (1 - h1)

    w2 += w2.T.dot(delta_h2)
    w1 += x.T.dot(delta_h1)

print(h2)
print(y)