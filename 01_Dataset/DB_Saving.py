import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from LSTM_Parameters import Variables
import math
from tqdm import tqdm
import torchvision.transforms.functional as TF
import os
import csv
import matplotlib.pyplot as plt
import time
import numpy as np 
import numpy

with open("", 'r') as file:
    csvreader = csv.reader(file)
    foldernamelist = list(csvreader)
    length = len(foldernamelist)

frame_limit = 25 #the number of frames extracted from each folder
folder_no = 60
width = 32
height = 32

class TrainDataset(Dataset):
	def __init__(self, number):
		self.folder_name = "" + str(foldernamelist[number])[2:22]
		file_list = sorted(glob.glob(self.folder_name + "/*.jpg"))
		self.data = []
		i = 1
		t = len(file_list)
		x = t/frame_limit
		ima = 1
		img_array = []
		for img_path in sorted(glob.glob(self.folder_name + "/*.jpg")):
			if (i<frame_limit+1 and t >= frame_limit):
				img_array.append(ima)
				ima = math.floor(1 + i*x)
				i += 1
			elif(i<frame_limit+1 and t < frame_limit):
				img_diff = frame_limit - t 
				if (ima == 1):
					for m in range(math.ceil(img_diff/2)+1):
						img_array.append(ima)
					ima = math.floor(1 + i)
					i += 1
				elif(ima > 1 and ima < t+1):
					img_array.append(ima)
					ima = math.floor(1 + i)
					i += 1
                
				if (ima == t):
					for m in range(math.floor(img_diff/2)):
						img_array.append(ima)
                        #print(m)
	
		for img in range(len(img_array)):
			image_name2 = '{:0>5}'.format(img_array[img])
			image_name2 = image_name2 + '.jpg'
			self.data.append(self.folder_name + "/" + image_name2)
		self.img_dim = (Variables.width, Variables.height)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path = self.data[idx]
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, self.img_dim)
		img_tensor = torch.from_numpy(img)
		img_tensor = img_tensor.permute(2, 0, 1)
		if (Variables.channels == 1):
			img_tensor = TF.rgb_to_grayscale(img_tensor,1)
		return img_tensor

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Saving Training Image Set")
	for group in tqdm(range(length)):
		T_data_loader = DataLoader(TrainDataset(group), shuffle=False, num_workers = 1)
		t_input_seq = torch.empty((1,len(T_data_loader),Variables.channels,Variables.height,Variables.width), dtype=torch.float32)
		t_label_seq = []
		k = 0
		for t_imgs in T_data_loader:
			t_imgs = t_imgs.unsqueeze(0)
			t_input_seq[:,k,:,:,:] = t_imgs
			k += 1
		k = 0

		class_name = str(foldernamelist[group])[10:23]
		os.makedirs('../Data/Training/Epoch_{}/'.format(math.floor(group/(12*Variables.folder_no))), exist_ok=True)
		os.makedirs('../Data/Validation/', exist_ok=True)
		if (group < 15000):
			torch.save(t_input_seq, '../Data/Training/Epoch_{}/'.format(math.floor(group/(12*Variables.folder_no))) + str(class_name))
		else:
			torch.save(t_input_seq, '../Data/Validation/' + str(class_name))

	
if __name__ == '__main__':
	main()