import torch
from torch.utils.data import Dataset
import glob

class Variables():
	### Generally for training ###
	train_groups = 4 			# How many training folders are being used, 5 is max 
	val_groups = 1 				# How many validation folders are being used, 1 is max
	num_epochs = 50  			# How many epochs to train for 
	hidden_size = 128 			# Replace with the desired hidden size
	num_layers = 3 				# Replace with the desired number of LSTM layers
	learning_rate = 0.001 		# Learning rate of the model
	batch_size = 25 			# How many folders we're processing at a time with dataloader 
	words = 25 					# How many words we're predicting, 25 is min and max

	### Generally for exportation ###
	train_data_folders = 60 	# How many train folders in the data folder for exportation
	val_data_folders = 15 		# How many validation folders in the date folder for exportation
	width = 32 					# Width of pictures
	height = 32 				# Height of pictures
	frame_limit = 25 			# How many frames from each video will be used
	channels = 1 				# Number of channels in the pictures

	### Generally for plots ###
	People = 14 				# Number of people in each training folder
	folder_no = 250 			# How many folders per person, every 10 folders = 1 words

class ValidationDataset(Dataset):
	def __init__(self):
		self.imgs_path = './Saved_images/Test/'
		file_list = glob.glob(self.imgs_path + "*")
		self.data = []
		for image_path in file_list:
			self.data.append(image_path)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path = self.data[idx]
		img_tensor = torch.load(img_path)
		return img_tensor
