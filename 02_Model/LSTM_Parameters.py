import torch
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import random

class Variables():
	### Generally for training ###
	train_groups = 5 			# How many training folders are being used, 5 is max 
	val_groups = 1 				# How many validation folders are being used, 1 is max
	num_epochs =  50 			# How many epochs to train for 
	hidden_size = 128 			# Replace with the desired hidden size
	num_layers = 3 				# Replace with the desired number of LSTM layers
	learning_rate = 0.001 		# Learning rate of the model
	batch_size = 25 			# How many folders we're processing at a time with dataloader 
	words = 25 					# How many words we're predicting, 25 is min and max

	### Generally for exportation ###
	train_data_folders = 60 	# How many train folders in the data folder for exportation
	val_data_folders = 12 		# How many validation folders in the data folder for exportation
	width = 32 					# Width of pictures
	height = 32 				# Height of pictures
	frame_limit = 25 			# How many frames from each video will be used
	channels = 3 				# Number of channels in the pictures

	### Generally for plots ###
	People = 12 				# Number of people in each training folder
	folder_no = 250 			# How many folders per person, every 10 folders = 1 words

#defining a custom dataset using pytorch's dataloader
class TrainDataset(Dataset):
	def __init__(self, number):
		self.imgs_path = '../Data/Training/Epoch_{}/'.format(str(number))
		file_list = glob.glob(self.imgs_path + "*")
		self.data = []
		for class_path in file_list:
			class_name = class_path.split("_")[2]
			self.data.append([class_path, class_name])
		self.class_map = {"001": 0, "002":1 , "003" : 2, "004":3, "005":4, "006":5,"007": 6, "008":7 , "009" : 8, "010":9, "011":10, "012":11, "013": 12, 
		    				"014":13 , "015" : 14, "016":15, "017":16, "018":17,"019": 18, "020":19 , "021" : 20, "022":21, "023":22, "024":23, "025":24}
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path, class_name = self.data[idx]
		img_tensor = torch.load(img_path)
		label = self.class_map[class_name]
		img_label = torch.tensor([label])
		return img_tensor, img_label

class ValidationDataset(Dataset):
	def __init__(self, number):
		self.imgs_path = '../Data/Validation/'
		file_list = glob.glob(self.imgs_path + "*")
		self.data = []
		for class_path in file_list:
			class_name = class_path.split("_")[1]
			self.data.append([class_path, class_name])
		self.class_map = {"001": 0, "002":1 , "003" : 2, "004":3, "005":4, "006":5,"007": 6, "008":7 , "009" : 8, "010":9, "011":10, "012":11, "013": 12, 
		    				"014":13 , "015" : 14, "016":15, "017":16, "018":17,"019": 18, "020":19 , "021" : 20, "022":21, "023":22, "024":23, "025":24}
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path, class_name = self.data[idx]
		img_tensor = torch.load(img_path)
		label = self.class_map[class_name]
		img_label = torch.tensor([label])
		return img_tensor, img_label

def manual_Normalization(tensor):
	for i in range(tensor.shape[0]):
		# normalisation # z-score normalisation
		if (Variables.channels == 3):
			mean = (tensor[i,:,0].mean(),tensor[i,:,1].mean(), tensor[i,:,2].mean())
			std = (tensor[i,:,0].std(),tensor[i,:,1].std(), tensor[i,:,2].std())
		else:
			mean = (tensor[i,:,0].mean())
			std = (tensor[i,:,0].std())	
		tensor[i] = transforms.Normalize(mean, std)(tensor[i])
		# scaling to [0,1] # unnecessary as we're already using Z-score normalisation
		min = tensor[i].min()
		max = tensor[i].max()
		tensor[i] = (tensor[i] - min)/(max - min)
	return tensor

def manual_ColorJitter(tensor):
	transform = transforms.ToPILImage()
	transform_back = transforms.PILToTensor()
	r_brightness = random.uniform(0.8,1.2)
	r_everything_else = random.uniform(0.75, 1.25)
	for i in range(tensor.shape[0]):
		img = transform(tensor[i,:,:,:])
		img = tf.adjust_brightness(img, r_brightness)
		img = tf.adjust_contrast(img, r_everything_else)
		img = tf.adjust_saturation(img, r_everything_else)
		img = tf.adjust_gamma(img, r_everything_else)
		tensor[i,:,:,:] = transform_back(img)
	return tensor

def transformation(transformed):
	G_blur = transforms.GaussianBlur(kernel_size=(3,5), sigma=(0.1,5))
	R_HFlip = transforms.RandomHorizontalFlip(p=0.5)
	R_Perspective = transforms.RandomPerspective(distortion_scale = 0.2, p = 0.3)
	R_Rotation = transforms.RandomRotation(degrees=15)
	for i in range(transformed.shape[0]):
		determinant = random.randint(0,5)
		if (determinant < 3): 
			transformed[i,:,:,:,:] = manual_ColorJitter(transformed[i,:,:,:,:]) 
		determinant = random.randint(0,6)
		if (determinant < 2): 
			transformed[i,:,:,:,:] = G_blur(transformed[i,:,:,:,:])
		transformed[i,:,:,:,:] = R_HFlip(transformed[i,:,:,:,:])
		transformed[i,:,:,:,:] = R_Perspective(transformed[i,:,:,:,:])				
		determinant = random.randint(0,3)
		if (determinant < 2): 
			transformed[i,:,:,:,:] = R_Rotation(transformed[i,:,:,:,:])
	return transformed