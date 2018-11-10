import numpy as np
import sys
import time
sys.path.append("Interface/python/")
from init import NeuronLayerBox
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

if __name__ == '__main__':
	NLB=NeuronLayerBox(step_ms=1,model=1,spike=0,restore=1)

	input_src=[]
	img=cv2.imread("load_data/input.bmp")
	img=rgb2gray(img).astype(int)

	input_src.append(img)
	NLB.step(20)
	NLB.input(input_src)
	for i in range(50):
		NLB.step(5)
		a=(NLB.output()['5']/max(np.max(NLB.output()['7']),0.0000001))*255
		cv2.imshow("1.jpg",a)
		cv2.waitKey(1)
	time.sleep(10)
	NLB.save()
	NLB.exit()
