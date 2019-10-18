import cv2
import matplotlib.pyplot as plt


a = plt.imread('/home/yotamg/pred.jpg')
b = cv2.bilateralFilter(a, 90, 175,75)
plt.subplot(211)
plt.imshow(a)
plt.subplot(212)
plt.imshow(b)
plt.show()
print ("A")