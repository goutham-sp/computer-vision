import cv2
import numpy as np
import math
import scipy.ndimage.measurements as measurements
import os

# Can be improved later. Know how to do it.

def get_foreground(image, x, y, w, h):
	"""
		Get Foreground of the image say bottle or a box eliminating the background noise.
		
		args:
			image : The image from which the forground needs to be extracted
			x, y, w, h : The coordinates of top left point and the width and height from where the foreground
							should be extracted.
		returns : An image where only the foreground is extracted.
	"""

	mask = np.zeros(image.shape[:2], dtype=np.uint8)
	bgdModel = np.zeros((1,65), dtype=uint8)
	fgdModel = np.zeros((1,65), dtype=uint8)

	rect = (x, y, w, h)

	cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
	mask2 = np.where((mask==2)|(mask==0), 0, 1)
	foreground = image*mask2[:,:,np.newaxis].astype('uint8')

	return foreground


def get_object(image, mask, crop_extra = False):
	"""
		Get the image croped from the masked image

		args:
			image : Image from where the object needs to be croped out
			mask : Mask of the image to crop out
			crop_extra : Crop an extra 15 pixels on all the sides

		returns : A croped image of the image given.
	"""

	points = np.argwhere(mask != 0)
	points = np.fliplr(points)
	print(points)

	# contours,hierarchy = cv2.findContours(res, 1, 2)

	# contours = cv2.findContours(mask, 1, 2)
	# contour = max(contours)

	x, y, w, h = cv2.boundingRect(points)

	# foreground = get_foreground(image, x, y, w, h)
	if crop_extra:
		return image[y+15:y+h-15, x+15:x+w-15]
	else:
		return image[y:y+h, x:x+w]


def crop_image(image, x1, x2, y1, y2):
	"""
		Crop with two coordinates, width and height

		args :
			image : Image which needs to be croped
			x1, x2, y1, y2 : coordinates to crop the image

		returns : A croped image which needs to processed further
	"""
	print(y1-y2)
	print(x1-x2)
	if x1 < 0:
		x1 = 0
	if x2 < 0:
		x2 = 0
	if y1 < 0:
		y1 = 0
	if y2 < 0:
		y2 = 0
	return np.rot90(image[y1:y2, x1:x2], k=3)


def crop_from_cordinates(image, top_left, top_right, bottom_left, bottom_right):
	"""
		Crop an image from the coordinates specified from the app

		args:
			image : Image which needs to croped
			top_left, top_right, bottom_left, bottom_right : Coordinates to crop the image

		returns : A croped image which needs to processed more for getting the object only
	"""
	mask = np.zeros(image.shape, dtype=np.uint8)
	roi_corners = np.array([[top_left, top_right, bottom_left, bottom_right]], dtype=np.int32)
	channel_count = image.shape[2]
	ignore_mask_color = (255,)*channel_count
	cv2.fillPoly(mask, roi_corners, ignore_mask_color)
	# from Masterfool: use cv2.fillConvexPoly if you know it's convex
	# apply the mask
	# cv2.imshow("Mask", mask)
	# cv2.imshow("Image", image)
	# cv2.waitKey(0)
	cv2.distroyAllWindows()
	return image, mask
	# masked_image = cv2.bitwise_and(image, mask)


def save_image(image_name, image):
	"""
		Saves image in a seperate Storage folder.

		args :
			image_name : Name of the image to save.
			image : Image needs to saved.

		returns : Boolean value specifing if the image is save or not
	"""
	d = os.getcwd()
	try:
		os.mkdir("Storage")
		os.chdir(os.path.join(d,"Storage"))
		# os.chdir(".\\Storage\\")

	except Exception as e:
		print("This Exception happened\n", e)
		os.chdir(os.path.join(d,"Storage"))
		# os.chdir(".\\Storage\\")

	try:
		cv2.imwrite(image_name, image)
		os.chdir(d)
		# os.chdir("..\\")
		return True
	except Exception as e:
		os.chdir(d)
		# os.chdir("..\\")
		print("This Exception happened\n", e)
		return False


def area_of_object(mask):
	"""
		Find the area of the object with the croped image so that the area of the liquid can be found.

		args:
			mask : Masked object to out the area of the masked pixels

		returns : An float value of area of the mask
	"""
	# image_temp = np.zeros(image.shape, dtype=np.uint8)
	(_, im_bw) = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
	(_, cnts, _) = cv2.findContours(im_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key = cv2.contourArea)

	area_of_object_ = cv2.contourArea(c)
	print(area_of_object_)

	return area_of_object_


def rgb2hex(r,g,b):
    hex = "#{:02x}{:02x}{:02x}".format(r,g,b)
    return hex


def find_object_only(image):
	"""
		Retrive the object only from the croped image

		args:
			image : Croped image

		returns : 
	"""
	w,h,d = image.shape
	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# image.shape = (w*h, 0, d)
	image = np.reshape(image, (w*h, d))
	# print("Image")
	# print(image)
	B, G, R = tuple(np.mean(image, axis = (0)))
	B = int(B)
	G = int(G)
	R = int(R)

	hex_code = rgb2hex(R,G,B)
	print("Hex Code", hex_code)

	# lower_object_color = higher_object_color = [B, G, R]

	# lower = higher = cv2.cvtColor(np.uint8([[lower_object_color]]), cv2.COLOR_BGR2HSV)[0,0]
	
	# temp_uint8 = np.array([50], dtype=np.uint8)

	# lower_object_color = np.array([lower[0] - temp_uint8, lower[1] - temp_uint8, lower[2] - temp_uint8], dtype=np.uint8)
	# higher_object_color = np.array([higher[0] + temp_uint8, 255, 255], dtype=np.uint8)
	# print("Boundaries")
	# print(lower_object_color)
	# print(higher_object_color)

	lower = np.array([60,50,50])
	higher = np.array([190,255,255])
	# mask_object = cv2.inRange(hsv_image, lower_object_color, lower_object_color)
	mask_object = cv2.inRange(hsv_image, lower, higher)
	# print("Mask shape", mask_object.shape)
	# res_object = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_object)
	# res_object = after_masking(image, mask=mask_object)
	area_of_object_ = area_of_object(mask_object)

	# cv2.imshow("Mask", mask_object)
	# cv2.waitKey(0)

	# print(mask_object)
	return mask_object, hex_code


def preprocess(image, hsv_image, lower_thresh, upper_thresh):
	"""
		Preprocess image to get only the object that is needed.

		args:
			image - Original image in BGR
			hsv_image - Original image in HSV
			lower_thresh - The lower boundary of HSV to grab from hsv_image
			upper_thresh - The upper boundary of HSV to grab from hsv_image

		returns : Preprocessed image after Gaussian Blur, Fore Ground Extraction, and other enhancements
	"""

	# cv2.imshow("Image", image)
	# cv2.waitKey(0)

	lower_thresh = np.array(lower_thresh)
	upper_thresh = np.array(upper_thresh)

	mask = cv2.inRange(hsv_image, lower_thresh, upper_thresh)
	res = cv2.bitwise_and(image, image, mask = mask)

	gray_img = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
	ret, thresh = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)

	# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

	# cv2.imshow("Mask", mask)
	# cv2.imshow("threshold", thresh)

	cv2.waitKey(0)

	img = get_object(image, thresh)
	print(img.shape)

	blur = cv2.GaussianBlur(img,(5,5),0)
	# cv2.imshow("Blur", blur)

	processed_image = blur

	find_object_only(blur, cv2.cvtColor(blur, cv2.COLOR_BGR2HSV))

	# cv2.imshow("Mask", mask)

	# cv2.waitKey(0)
	cv2.destroyAllWindows()

	return processed_image