import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([0,1,1,0])


W1 = np.random.randn(2, 2) * 0.1
b1 = np.zeros(2)
W2 = np.random.randn(1, 2) * 0.1
b2 = np.zeros(1)

print (W1.shape,W2.shape)

def sigmo (x):
    return 1 / (1 + np.exp(-x))


z1 = W1 @ X[0] + b1
a1 = sigmo(z1)
z2 = W2 @ a1 + b2
a2 = sigmo(z2)
print (a2)


