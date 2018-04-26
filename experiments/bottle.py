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
    cv.imwrite('BottleData/'+image+str(num)+'.png',image_name)
    print ("Written..")

def read_image(image,image_name,num):
    return cv.imread('BottleData/'+image+str(num)+'.png')

def traverse_blocks_x(grid,i,image,width,height,x,x1,mul):
    ratio = math.floor((width/height)*10)
    i = i + 1
    y = mul * math.floor(height/ratio)

    if i < ratio:
        x = x1
        y1 = y + math.floor(height/ratio)

        grid[mul][i] = image[y:y1,x:x1+math.floor(width/ratio)]

        #replace by HSV/RGB Calculator/Gauss avr
        traverse_blocks_x(grid,i,image,width,height,x,x1+math.floor(width/ratio),mul)

def lowest_difference(difference_list_white):
    lowest = difference_list_white[0]
    for i in difference_list_white:
        if i < lowest:
            lowest = i
    return lowest


def water_level(total_area, image):
    white_buffer_list = []
    black_buffer_list = []
    difference_list_black = []
    difference_list_white = []

    height = 0
    width = 0
    value = 30

    img_list = ['pepsi']#,'WaterBottle1','WaterBottle2','WaterBottle3']

    for images in img_list:
        cv_image = cv.imread('BottleData/'+images+'.png')
        cv_image = cv.GaussianBlur(cv_image,(5,5),0)
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

        print_image(closing)
        write_image(images,closing,1)
        iter_two = read_image(images,closing,1)

        gray_image = cv.cvtColor(iter_two, cv.COLOR_BGR2GRAY)
        gray_image = cv.GaussianBlur(gray_image,(5,5),0)
        print_image(gray_image)
        write_image(images,gray_image,2)

        print ("GrayScale Sequence Completed ..")
        gray_image_read = read_image(images,gray_image,2)


        height,width,_ = gray_image_read.shape
        ratio = math.floor((width/height)*10)

        grid = np.zeros((ratio,ratio),dtype=object)
        for mul in range(0,3):
            traverse_blocks_x(grid,-1,gray_image_read,width,height,0,0,mul)
        for postition in range(0,3):
            img1 = grid[postition][1]
            print_image(img1)
            write_image('selected',img1,postition)

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
        if difference_list_black.index(lowest_difference(difference_list_black)) == 0:
            print ("Bottle is almost full")
            print ("Calculating Area of liquid")

        elif difference_list_black.index(lowest_difference(difference_list_black)) == 1:
            print ("Bottle is half full")

        else:
            print ("Bottle is empty")



                # white_location = calculate_grid_RGB(0,r,red,g,green,b,blue)
                # print (white_location)