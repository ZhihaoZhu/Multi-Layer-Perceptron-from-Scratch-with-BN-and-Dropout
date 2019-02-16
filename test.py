import numpy as np
from nn import *
import matplotlib.pyplot as plt

W1 = np.load("./saved_model2/W1_30.npy")
W1 = W1.T

W1 = W1.reshape(100,28,56)
fig=plt.figure()
for i in range(1, 101):
    img = W1[i-1,:,:]
    fig.add_subplot(10, 10, i)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
plt.show()