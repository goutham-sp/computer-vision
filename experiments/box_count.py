import cv2
import numpy as np
from matplotlib import pyplot as plt
from image_preprocess import preprocess


class BoxCounter(object):

	def __init__(self, image):

		self.hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		self.image = image
	
		lower_thresh = [150,80,50]
		upper_thresh = [210,255,255]

		self.processed_image = preprocess(self.image, self.hsv_image, lower_thresh, upper_thresh)


	def draw_boxes(self):
		img_temp = self.image


	def count_boxes(self):
		pass


	def calc_covered_area(self):
		pass



def main():
	image = cv2.imread(".\\tests\\image17.png")
	box = BoxCounter(image)
	boxes = box.draw_boxes()
	counts = box.count_boxes()
	area_covered = box.calc_covered_area()
	print("Boxes\n\n", boxes)
	print("Counts", counts)
	print("Area Covered", area_covered)


if __name__ == "__main__":
	main()