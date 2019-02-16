import numpy as np
import os
import matplotlib.pyplot as plt


test = np.loadtxt('../data/data/test.txt',delimiter= ',',unpack= False)
image = test[:4,:-1]
image = image.reshape(4,28,56)
a = image[:,:,28:]*255
fig=plt.figure()
for i in range(1, 5):
    img = a[i-1,:,:]
    fig.add_subplot(1, 4, i)
    plt.imshow(img)
plt.show()


