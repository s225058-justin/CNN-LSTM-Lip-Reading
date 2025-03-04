import torch
import torch.nn as nn
import torch.nn.functional as F
from LSTM_Parameters import Variables, transformation, manual_Normalization
import random

#defining the CNN_LSTM model we'll be using
class CNN_LSTM(nn.Module):
	def __init__(self, hidden_size, num_layers):
		super(CNN_LSTM, self).__init__()
		# convolution layer 1
		self.conv1 = nn.Conv2d(Variables.channels, 16, 5, padding = 'same')
		self.conv5 = nn.Conv2d(16, 16, 5, padding = 'same')
		self.bn1 = nn.BatchNorm2d(16)
		self.dropout1 = nn.Dropout(0.2)
		
		# convolution layer 2
		self.conv2 = nn.Conv2d(16, 32, 5, padding = 'same')
		self.conv6 = nn.Conv2d(32, 32, 5, padding = 'same')
		self.bn2 = nn.BatchNorm2d(32)
		self.dropout2 = nn.Dropout(0.2)

		# convolution layer 3
		self.conv3 = nn.Conv2d(32, 64, 3, padding = 'same')
		self.conv7 = nn.Conv2d(64, 64, 3, padding = 'same')
		self.bn3 = nn.BatchNorm2d(64)

		# output layer
		self.lstm1 = nn.LSTM(4096, hidden_size, num_layers, batch_first=True, dropout = 0.4) 
		self.fc1 = nn.Linear(hidden_size, Variables.words)


	def forward(self, input_seq, transform, epoch):
		input_seq = input_seq.to(torch.uint8).cuda()
		z = torch.zeros((Variables.batch_size,Variables.frame_limit,64,8,8), dtype=torch.float32).cuda()
		determinant = random.randint(0,3)
		if (determinant < 2 and transform == 1):
			input_seq = transformation(input_seq)
		input_seq = manual_Normalization(input_seq.to(torch.float32))	
		for i in range(input_seq.shape[1]):
			# convolution layer 1
			y = self.conv1(input_seq[:,i,:,:])
			y = self.conv5(y)
			y = self.bn1(y)
			y = F.max_pool2d(y, 2)
			y = F.relu(y)			
			if (epoch >= 10):
				y = self.dropout1(y)

			# convolution layer 2 + dropout
			y = self.conv2(y)
			y = self.conv6(y) 
			y = self.bn2(y)
			y = F.relu(y)
			if (epoch >= 10):
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
