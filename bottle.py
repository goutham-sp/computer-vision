import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import medial_axis, skeletonize, skeletonize_3d
from skimage.data import *
from skimage import data
from skimage import img_as_bool
from skimage.data import binary_blobs
import skimage.io
import math
from skimage.util.colormap import magma
import os


class iterator:
	def __init__(self, max = 0):
		self.max = max
	def __iter__(self):
		self.n = 0
		return self
	def __next__(self):
		if self.n <= self.max:
			res = 1 + self.n
			self.n +=1
			return res
		else:
			raise StopIteration

def print_image(image_name):
	cv.imshow("Print",image_name)
	cv.waitKey(0)
	cv.destroyAllWindows()

def write_image(image,image_name,num):
	cv.imwrite(os.path.join('Storage',image_name+str(num)+'.png'),image)
	print ("Written..")

def read_image(image_name,num):
	return cv.imread(os.path.join('Storage', image_name+str(num)+'.png'))

def traverse_blocks_x(grid,i,image,width,height,x,x1,mul):
	print("Dividing grid", i)
	ratio = math.floor((width/height)*10)
	i = i + 1
	y = mul * math.floor(height/ratio)

	if i < ratio:
		x = x1
		y1 = y + math.floor(height/ratio)
		l1 = y1-y
		l2 = (x1 + math.floor(width/ratio)) - x1

		# grid[mul, i] = np.zeros((l1, l2, 3), dtype=int)

		# np.copyto(grid[mul, i], image[y:y1, x:(x1 + math.floor(width/ratio))])
		grid[mul][i] = np.array(image[y:y1, x:(x1 + math.floor(width/ratio))])

		#replace by HSV/RGB Calculator/Gauss avr
		traverse_blocks_x(grid,i,image,width,height,x,x1+math.floor(width/ratio),mul)


def lowest_difference(difference_list_white):
	lowest = difference_list_white[0]
	for i in difference_list_white:
		if i < lowest:
			lowest = i
	return lowest

def count_non_black(array_):
	count = 0
	a = np.zeros((0, 3))

	for item in array_:
		# if item != [0 0 0]:
		if (item!=array_).all():
			count+=1
	return count

def find_water_level(image):
	# sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
	sobely = cv.Sobel(image,cv.CV_64F,0,1,ksize=5)
	most = 0
	index = 0
	# print(sobely.shape[0])
	# ret,th1 = cv2.threshold(sobely,255,255,cv2.THRESH_BINARY)
	ret,th1 = cv.threshold(sobely,255,255,cv.THRESH_BINARY)

	for i in range(th1.shape[0]):
		# print(th1[i])
		cur_array = count_non_black(th1[i])
		if cur_array > most:
			most = cur_array
			index = i
	print(index)
	return (th1.shape[0] - index) / 100

def calculate_area_liquid(object_area, image, level):
	level_line = find_water_level(image)
	if level == 2:
		area_of_liquid = object_area + ((object_area/3) - ((object_area/3) * level_line))

	elif level == 1:
		area_of_liquid = (2 * (object_area/3)) - ((object_area/3) * level_line)

	elif level == 0:
		area_of_liquid = (object_area/3) - ((object_area/3) * level_line)

	print(area_of_liquid)

	return area_of_liquid

def liquid_level(image, area_of_object):
	white_buffer_list = []
	black_buffer_list = []
	difference_list_black = []
	difference_list_white = []

	height = 0
	width = 0
	value = 30

	cv_image = cv.imread(os.path.join('Storage', image))
	print("Opened image")
	# cv_image = cv.GaussianBlur(cv_image,(5,5),0)
	hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV) #convert it to hsv
	h, s, v = cv.split(hsv)
	lim = 255 - value
	v[v > lim] = 255
	v[v <= lim] += value

	final_hsv = cv.merge((h, s, v))
	img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
	kernel = np.ones((5,5),np.uint8)
	erosion = cv.erode(img,kernel,iterations = 1)
	opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
	closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

	# print_image(closing)
	write_image(closing,"closing",1)
	iter_two = read_image("closing",1)

	gray_image = cv.cvtColor(closing, cv.COLOR_BGR2GRAY)
	gray_image = cv.GaussianBlur(gray_image,(5,5),0)
	write_image(gray_image,"gray_image",2)

	print ("GrayScale Sequence Completed ..")
	gray_image = read_image("gray_image",2)

	height,width,_ = gray_image.shape
	ratio = math.floor((width/height)*10)

	print('Ratio='+str(ratio))


	# grid = np.zeros((ratio,ratio),dtype=np.uint8)
	grid = []
	for i in range(0, ratio):
		new = []
		for j in range(0, ratio):
			new.append(0)
		grid.append(new)

	for mul in range(0, ratio):
		traverse_blocks_x(grid,-1,gray_image,width,height,0,0,mul)
	for postition in range(0, ratio):
		img1 = grid[postition][1]
		# print_image(img1)
		write_image(img1,'selected',postition)

		height,width,channels = img1.shape
		red = list()
		green = list()
		blue = list()
		r1,g1,b1 = img1[0,0]

		red.append(r1)
		green.append(g1)
		blue.append(b1)
		white_buffer = 0
		for w in range(0,height):
			for h in range(0,width):
				r,g,b  = (img1[w,h])
				if r > 100 and g > 100 and b > 100:
					white_buffer = white_buffer + 1
				else:
					black_buffer = (w*h) - white_buffer
		white_buffer_list.append(white_buffer)
		black_buffer_list.append(black_buffer)
		difference_list_white.append(white_buffer-black_buffer)
		difference_list_black.append(black_buffer-white_buffer)

	high_white = max(white_buffer_list)
	high_black = max(black_buffer_list)
	print (difference_list_black)

	sentence = ""

	if difference_list_black.index(lowest_difference(difference_list_black)) == 0:
		img = read_image("selected", 0)
		area_liquid = calculate_area_liquid(float(area_of_object), img, level=2)
		sentence = "Bottle is nearly Full. "

	elif difference_list_black.index(lowest_difference(difference_list_black)) == 1:
		img = read_image("selected", 1)	
		area_liquid = calculate_area_liquid(float(area_of_object), img1, level=1)
		sentence = "Bottle is not empty yet. "

	else:
		img = read_image("selected", 2)	
		area_liquid = calculate_area_liquid(float(area_of_object), img1, level=0)
		sentence = "Bottle is nearly empty. "

	sentence += str(math.ceil((area_liquid/float(area_of_object))*100)) + "% " + "Full"
	print(sentence)
	return sentence

			# white_location = calculate_grid_RGB(0,r,red,g,green,b,blue)
			# print (white_location)