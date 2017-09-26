import numpy as np
import cv2

mat = np.load("test_logs/jaco_pixel/w0.npz")
print(mat.shape)

low = np.min(mat)
high = np.max(mat)
print(low, high)
for i in range(mat.shape[3]):
    filt = mat[:,:,:,i]
    filt = (filt - low ) / (high - low)
    #filt *= 255
    cv2.imshow("win", filt[:,:,0])
    cv2.imshow("win1", filt[:,:,1])
    cv2.waitKey(0)
