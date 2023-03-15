import cv2
import numpy as np

img = cv2.imread('coins.png')
g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
thresh = cv2.threshold(g_img, 145, 255, cv2.THRESH_BINARY_INV)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))
dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
closing = cv2.morphologyEx(dilate, cv2.MORPH_ERODE, kernel)
cv2.imshow("Grayscale segment",closing)

img_sat = hsv_img[:,:,1]
hsvTh = cv2.inRange(img_sat,38,255)
n_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))
hsv_opening = cv2.morphologyEx(hsvTh, cv2.MORPH_OPEN, kernel)
hsv_closing = cv2.morphologyEx(hsv_opening, cv2.MORPH_CLOSE, kernel)

cv2.imshow("HSV segment",hsv_closing)

mask = cv2.bitwise_and(hsv_closing,closing)
mask = cv2.blur(mask,(2,2))
cv2.imshow("Mask",mask)

result = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("Result",result)
cv2.imwrite("coin_mask.png",result)
cv2.waitKey(0)
cv2.destroyAllWindows()