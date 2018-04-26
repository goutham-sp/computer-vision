import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


class CheckMultiple(object):

	def __init__(self, image):
		self.hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		self.image = image
		self.height, self.width, _ = self.image.shape


	def check_for_two(self):
		temp_image = self.image
		cv2.line(temp_image,(0,math.ceil(width/2)),(height,math.ceil(width/2)),(0,0,255),1)
		if np.all([self.image, temp_image]):
			return True
		else:
			return False


def main():
	c = CheckMultiple(cv2.imread(".\\tests\\image17.png"))
	c.check_with_mid()


if __name__ == "__main__":
	main()