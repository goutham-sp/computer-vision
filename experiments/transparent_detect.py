import cv2
import numpy as np
from matplotlib import pyplot as plt


def if_high_resolution(image):
	lower_red = np.array([245,0,0])
	upper_red = np.array([255,5,5])

	mask = cv2.inRange(image, lower_red, upper_red)
	res = cv2.bitwise_or(image, image, mask= mask)

	_, thresh_grey = cv2.threshold(res, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

	cv2.imshow('mask',thresh_grey)

	kernel = np.ones((3,3),np.uint8)
	# dilation = cv2.dilate(erosion,kernel,iterations = 1)
	opening = cv2.morphologyEx(thresh_grey, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	kernel = np.ones((8,8), np.uint8)
	erosion = cv2.erode(closing, kernel, iterations = 1)

	points = np.argwhere(erosion == 0)
	points = np.fliplr(points)
	x, y, w, h = cv2.rectangle(points)
	
	crop2 = erosion[y:y+h, x:x+w]
	# detect = DetectTransparent(im)
	# detect.detect_edges(thresh)
	# print(detect.find_avg_region(0,0,30,30))
	cv2.imshow('Final Crop', crop2)
	if cv2.waitKey(0) and 0xFF:
		cv2.destroyAllWindows()


def if_low_resolution(image):
	lower_red = np.array([245, 0, 0])
	upper_red = np.array([255, 5, 5])

	mask = cv2.inRange(image, lower_red, upper_red)
	res = cv2.bitwise_or(image, image, mask = mask)

	_, thresh_grey = cv2.threshold(res, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

	kernel = np.ones((3,3), np.uint8)

	cv2.imshow('mask', thresh_grey)

	dilation = cv2.dilate(thresh_grey, kernel, iterations = 1)
	opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

	kernel = cv2.ones((7,7), np.uint8)
	
	erosion = cv2.erode(closing, kernel, iterations = 1)

	points = np.argwhere(erosion == 0)
	points = np.fliplr(points)
	x, y, w, h = cv2.boundingRect(points)

	crop = erosion[y:y+h, x:x+w]

	cv2.imshow("Final crop", crop)
	if cv2.waitKey(0) and 0xFF:
		cv2.destroyAllWindows()


def main():

	image = cv2.imread(".\\tests\\image17.png")
	if_high_resolution(image)


if __name__ == "__main__":
	main()