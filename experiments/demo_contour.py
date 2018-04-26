import numpy as np
import cv2

# load the image
image = cv2.imread("C:\\Users\\Goutham\\Documents\\SakaLabs\\Imageclassification\\tests\\image14.jpeg", 1)
image = cv2.resize(image, (600, 800))

# red color boundaries (R,B and G)
lower = [100, 100, 100]
upper = [255, 255, 255]

# create NumPy arrays from the boundaries
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)

ret,thresh = cv2.threshold(mask, 40, 255, 0)
im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
	# draw in blue the contours that were founded
	cv2.drawContours(output, contours, -1, 255, 3)

	#find the biggest area
	c = max(contours, key = cv2.contourArea)
	max_contour = np.subtract(contours, c)
	c = max(max_contour, key = cv2.contourArea)

	x,y,w,h = cv2.boundingRect(c)
	# draw the book contour (in green)
	cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

# show the images
cv2.imshow("Result", output)
cv2.imshow("Original", image)

cv2.waitKey(0)