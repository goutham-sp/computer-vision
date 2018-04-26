import cv2
import numpy as np
from matplotlib import pyplot as plt


def preprocess(image):
	
	# Image Thresholding
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	# Noise Removal
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

	return ret, thresh, kernel, opening


def back_ground(opening, kernel):

	sure_bg = cv2.dilate(opening, kernel, iterations=3)
	return sure_bg


def fore_ground(opening):
	
	dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
	ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
	return ret, sure_fg


def unknown_region(sure_fg_, sure_bg):

	# sure_fg = np.uint8(sure_fg_)
	unknown = cv2.subtract(sure_bg, sure_fg_)
	return sure_fg, unknown


def load_image(image_):

	try:
		image = cv2.imread(image_)
		return image
	except Exception as e:
		print(e)
		return None


def save_image(image):
	try:	
		cv2.imwrite("testimage.png", image)
		return True
	except Exception as e:
		print(e)
		return False


def remove_background(image):

	fgbg = cv2.createBackgroundSubtractorMOG2()
	fgmask = fgbg.apply(image)

	cv2.imshow('frame',fgmask)
	cv2.waitKey()
	cv2.destroyAllWindows()


def foreground_extraction(img):

	mask = np.zeros(img.shape[:2],np.uint8)

	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)

	rect = (50,50,450,290)
	cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img = img*mask2[:,:,np.newaxis]

	plt.imshow(img),plt.colorbar(),plt.show()
	return img
"""

"""
class Detect(object):
	def __init__(self):
		pass

	def load_image(self, image):
		image_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
		image_y = np.zeros(image_yuv.shape[0:2],np.uint8)
		image_y[:,:] = image_yuv[:,:,0]
		return image_y

	def reduce_noise(self, image, kernel):
		dilation = cv2.dilate(image,kernel,iterations = 1)
		opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
		closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
		image_blurred = cv2.GaussianBlur(closing,(3,3),0)
		return image_blurred

	def detect_edges(self, image):
		edges = cv2.Canny(image,100,300,apertureSize = 3)
		return edges

	def contours_(self, edges):
		contours = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		return contours

	def show_image(self, image):
		plt.imshow(image)
		plt.show()

def main():
	# image = load_image("C:\\Users\\Goutham\\Documents\\SakaLabs\\Imageclassification\\tests\\image13.jpeg")
	# ret, thresh, kernel, opening = preprocess(image)
	# print(type(ret))
	# print(type(thresh))
	# print(type(kernel))
	# print(type(opening))	
	# sure_bg = back_ground(opening = opening, kernel = kernel)
	# print(type(sure_bg))
	# ret, sure_fg = fore_ground(opening = opening)
	# print(type(ret))
	# print(type(sure_fg))
	# sure_fg, unknown = unknown_region(sure_fg_ = sure_fg, sure_bg = sure_bg)
	# sure_fg = remove_background(image)
	# img = foreground_extraction(image)
	# print(save_image(img))
	d = Detect()
	original = cv2.imread("C:\\Users\\Goutham\\Documents\\SakaLabs\\Imageclassification\\tests\\image13.jpeg")
	noisy_image = d.load_image(original)
	kernel = np.ones((3,3),np.uint8)
	print("Noisy Image")
	d.show_image(noisy_image)
	image = d.reduce_noise(noisy_image, kernel)
	print("Noise Removed")
	d.show_image(image)
	edges = d.detect_edges(image)
	print("Edges")
	d.show_image(edges)
	contours = d.contours_(edges)
	print("contours")
	print(contours)


if __name__ == "__main__":
	main()