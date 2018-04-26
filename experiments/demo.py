import cv2
import numpy as np
from matplotlib import pyplot as plt


class DetectObject(object):

	def preprocess(self, image):
		image2 = cv2.imread(image, 0)