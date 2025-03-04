import torch
import torch.nn as nn
import torch.nn.functional as F
from Webcam_Parameters import Variables
import torchvision.transforms as transforms

def manual_Normalization(tensor):
	for i in range(tensor.shape[0]):
		# normalisation
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

#defining the CNN_LSTM model we'll be using
class CNN_LSTM(nn.Module):
	def __init__(self, hidden_size, num_layers, l1=128):
		super(CNN_LSTM, self).__init__()
		# convolution layer 1
		self.conv1 = nn.Conv2d(Variables.channels, 16, 5, padding = 'same')
		self.conv5 = nn.Conv2d(16, 16, 5, padding = 'same')
		self.bn1 = nn.BatchNorm2d(16)

		# convolution layer 2
		self.conv2 = nn.Conv2d(16, 32, 5, padding = 'same')
		self.conv6 = nn.Conv2d(32, 32, 5, padding = 'same')
		self.bn2 = nn.BatchNorm2d(32)
		self.dropout1 = nn.Dropout(0.2)

		# convolution layer 3
		self.conv3 = nn.Conv2d(32, 64, 3, padding = 'same')
		self.conv7 = nn.Conv2d(64, 64, 3, padding = 'same')
		self.bn3 = nn.BatchNorm2d(64)

		# output layer
		self.lstm1 = nn.LSTM(4096, l1, num_layers, batch_first=True, dropout = 0.4) 
		self.fc1 = nn.Linear(l1, Variables.words)

	def forward(self, input_seq):
		input_seq = input_seq.to(torch.uint8)
		z = torch.zeros((1,Variables.frame_limit,64,8,8), dtype=torch.float32)
		input_seq = manual_Normalization(input_seq.to(torch.float32))	
		for i in range(input_seq.shape[1]):
			# convolution layer 1
			y = self.conv1(input_seq[:,i,:,:]) 
			y = self.conv5(y)
			y = self.bn1(y)
			y = F.max_pool2d(y, 2)
			y = F.relu(y)			
			y = self.dropout1(y)

			# convolution layer 2 + dropout
			y = self.conv2(y)
			y = self.conv6(y) 
			y = self.bn2(y)
			y = F.relu(y)
			y = self.dropout1(y)
				
			# convolution layer 3
			y = self.conv3(y)
			y = self.conv7(y)
			y = self.bn3(y)
			y = F.max_pool2d(y, 2)
			y = F.relu(y)

			z[:,i,:,:,:] = y
		z = z.view(z.shape[0], z.shape[1], -1)

		# output layer
		lstm_out, _ = self.lstm1(z)
		final_out = self.fc1(lstm_out[:,-1,:])
		
		return final_out
