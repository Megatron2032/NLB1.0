# coding=UTF-8
from ctypes import *
import time
import numpy as np
import os
class shared_use_st(Structure):
	_fields_ = [("written", c_int),
                ("_switch_", c_int),
                ("timen", c_int),
                ("Nlines", c_int),
				("input_num", c_int),
				("output_num", c_int),
                ("step", c_int),
				("output_spike", c_int),
				("model", c_int),
				("save", c_int),
				("restore", c_int),
				("restore_file", c_char*100)];

class shared_iamge(Structure):
	_fields_ = [("written_flag", c_int),
                ("read_flag", c_int),
                ("length", c_int),
				("layer", c_int),
                ("data", c_int*921600),
                ("width", c_int),
                ("height", c_int),
				("next_frame", c_int)];

class NeuronLayerBox:
	def __init__(self,step_ms=10,model=1,spike=0,restore=0):
		mylib = CDLL("libNLB.so")
		self.NLB_init = mylib.NLB_init
		self.NLB_init.argtypes = [c_int] # 参数类型，两个int（c_int是ctypes类型，见上表）
		#self.NLB_init.restype = c_int # 返回值类型，int (c_int 是ctypes类型，见上表）
		self.NLB_init.restype = POINTER(shared_use_st)
		self.p = self.NLB_init(step_ms,model)  #p.contents.written

		self.NLB_step=mylib.NLB_step
		self.NLB_step.argtypes = [c_int,POINTER(shared_use_st)]

		#####input#####
		self.NLB_input_init = mylib.input_init
		self.NLB_input_init.restype = POINTER(shared_iamge)
		self.input_data = self.NLB_input_init(self.p.contents.input_num)  #self.input.contents.written

		self.NLB_input = mylib.input
		self.NLB_input.argtypes = [POINTER(c_int),POINTER(shared_iamge),c_int,c_int]

		#####output#####
		self.NLB_output_init = mylib.output_init
		self.NLB_output_init.restype = POINTER(shared_iamge)
		self.output_data = self.NLB_output_init(self.p.contents.output_num)  #self.input.contents.written

		if restore:
			rootdir = 'load_data/SaveData'
			list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
			if len(list)>0:
				year=[0 for i in range(len(list))]
				mon=[0 for i in range(len(list))]
				day=[0 for i in range(len(list))]
				hour=[0 for i in range(len(list))]
				minu=[0 for i in range(len(list))]
				sec=[0 for i in range(len(list))]
				all_time=[0 for i in range(len(list))]
				for i in range(0,len(list)):
					path = os.path.join(rootdir,list[i])
					if os.path.isfile(path):
						[year[i],mon[i],day[i],hour[i],minu[i],sec[i]]=list[i].strip('.txt').split('_')
						print(year[i],mon[i],day[i],hour[i],minu[i],sec[i])
						all_time[i]=((((int(year[i])*12+int(mon[i]))*30+int(day[i]))*24+int(hour[i]))*60+int(minu[i]))*60+int(sec[i])
				max_num=np.argmax(np.array(all_time))
				self.p.contents.restore_file=os.path.join("load_data/SaveData/",list[max_num])
				print("load_data in {}\n".format(list[max_num]))
			else:
				print("load_data/SaveData has no data, train a new model\n")
				restore=0

		self.p.contents.restore=restore
		self.p.contents.output_spike=spike
		self.p.contents._switch_=1
		print("switch ON\n")
		print("wait signal\n")
		while(self.p.contents.written!=1):
			time.sleep(0.01)
		print("signal come\n")

		self.out_data = {}
		for i in range(self.p.contents.output_num):
			self.out_data.update({str(self.output_data[i].layer):np.zeros((self.output_data[i].height,self.output_data[i].width))})

	def step(self,step_ms):
		self.NLB_step(step_ms,self.p)

	def save(self):
		self.p.contents.save=1
		print("saved model")

	def exit(self):
		self.p.contents._switch_=0
		self.p.contents.step=2

	def input(self,in_data):
		i=0
		for _data in in_data:
			_data=np.array(_data).reshape(-1).astype('int32')
			assert (len(_data) == self.input_data[0].length), 'input error : required {} not {}'.format(self.input_data[i].length,len(_data))
			if not _data.flags['C_CONTIGUOUS']:
				_data = np.ascontiguous(_data, dtype=_data.dtype)  # 如果不是C连续的内存，必须强制转换
			data_ctypes_ptr = cast(_data.ctypes.data, POINTER(c_int))   #转换为ctypes，这里转换后的可以直接利用ctypes转换为c语言中的int*，然后在c中使用
			#for m in range(len(_data)):
				#print(_data[m],data_ctypes_ptr[m])
			self.NLB_input(data_ctypes_ptr,self.input_data,i,self.p.contents.model)
			i=i+1

	def output(self):
		if not self.p.contents.model:
			self.output_data[0].read_flag=1
			self.output_data[0].written_flag=1
			while(self.output_data[0].written_flag==1):
				pass

		for i in range(self.p.contents.output_num):
			for j in range(self.output_data[i].height):
				for k in range(self.output_data[i].width):
					self.out_data[str(self.output_data[i].layer)][j][k]=self.output_data[i].data[j*self.output_data[i].width+k]
		return self.out_data

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

if __name__ == '__main__':
	import cv2

	NLB=NeuronLayerBox(step_ms=1,model=1,spike=0,restore=1)
	print NLB.p.contents.Nlines
	print NLB.p.contents.step

	input_src=[]
	img=cv2.imread("../../load_data/input.bmp")
	img=rgb2gray(img).astype(int)

	input_src.append(img)
	NLB.step(20)
	NLB.input(input_src)
	for i in range(50):
		NLB.step(5)
		a=(NLB.output()['5']/max(np.max(NLB.output()['5']),0.0000001))*255
		cv2.imshow("1.jpg",a)
		cv2.waitKey(1)
	time.sleep(10)
	print NLB.p.contents.step
	NLB.save()
	NLB.exit()
