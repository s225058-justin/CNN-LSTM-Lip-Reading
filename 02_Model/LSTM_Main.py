### Importing the other python files ###
from LSTM_Parameters import Variables, TrainDataset, ValidationDataset
from LSTM_Save import Drawing, Draw_graph
from LSTM_Model import CNN_LSTM

### Importing regular libraries ###
import gc
import torch
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# **********************************************
# main
# **********************************************
def main(name, mode):
	start_time = datetime.now()
	#Setting up the LSTM model
	model = CNN_LSTM(Variables.hidden_size, Variables.num_layers)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	Draw = Drawing()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), weight_decay=0.1, lr = Variables.learning_rate)
	
	# Learning loop
	best_validation_loss = 100
	best_accuracy = 0 
	best_train_loss = 100
	best_ac_tl = 100
	best_ac_vl = 100
	best_tl_ac = 0
	best_tl_vl = 100
	best_vl_ac = 0
	best_vl_tl = 100
	best_tl_model = 0
	best_vl_model = 0		
	best_ac_model = 0		
	accuracy = 0
	transformation = 0

	print("Begin Training -----> Validation")
	for epoch in tqdm(range(Variables.num_epochs)):
		if (epoch >= 5):
			transformation = 1
		model.train()
		train_loss = 0.0
		correct = 0
		total = 0
		for group in range(Variables.train_groups):
			#____________________________________________________________________________________________________________________________#
			#_______________________________________________________ training set _______________________________________________________#
			#____________________________________________________________________________________________________________________________#
			t_input_seq = torch.empty((0,Variables.frame_limit,Variables.channels*Variables.height*Variables.width), dtype=torch.float32)
			t_label_seq = torch.empty((Variables.frame_limit,0), dtype=torch.int64)

			# Loading data batch_size folders at a time
			T_data_loader = DataLoader(TrainDataset(group), batch_size=Variables.batch_size, shuffle=True, num_workers = 10)
			for t_imgs, t_labels in T_data_loader:
				t_imgs = torch.squeeze(t_imgs, dim=1)
				t_labels = torch.squeeze(t_labels)

				# To randomise the order of the tensors inputted, uncomment the following code
				#rdm = torch.randperm(Variables.batch_size)
				#t_imgs = t_imgs.view(-1,frame_limit,Variables.channels*Variables.height*Variables.width)[rdm].view(t_imgs.size())
				#t_labels = t_labels.view(Variables.batch_size)[rdm].view(t_labels.size())
				#t_length_seq = t_length_seq.view(Variables.batch_size)[rdm].view(t_length_seq.size())

				# Feeding inputs into GPU
				t_imgs = t_imgs.to(device)
				t_labels = t_labels.to(device)
				outputs = model(t_imgs, transformation, epoch)
				loss = criterion(outputs, t_labels)


				# Backward pass and optimization
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# Calculating loss and accuracy
				train_loss += loss.item() * t_labels.size(0)
				_, predicted = torch.max(outputs.data, 1)
				total += t_labels.size(0)
				correct += (predicted == t_labels).sum().item()

		# Total loss divided by total number of folders used for training at the time for loss per folder
		train_loss /= total  
		Draw.train_loss_epoch[epoch] = train_loss
		
		train_accuracy = 100.0 * correct / total
		Draw.train_accuracy_epoch[epoch] = train_accuracy
		transformation = 0
		del t_input_seq, t_label_seq, t_imgs, t_labels
		gc.collect()

		#____________________________________________________________________________________________________________________________#
		#______________________________________________________ validation set ______________________________________________________#
		#____________________________________________________________________________________________________________________________#
		model.eval()
		validation_loss = 0.0
		correct = 0
		total = 0
		temp_number_occurence = np.zeros(Variables.words)
		for group in range(Variables.val_groups):
			v_input_seq = torch.empty((0,Variables.frame_limit,Variables.channels,Variables.height,Variables.width), dtype=torch.float32)
			v_label_seq = torch.empty((Variables.frame_limit,0), dtype=torch.int64)

			V_data_loader = DataLoader(ValidationDataset(group), batch_size=Variables.batch_size, shuffle=True, num_workers = 10)
			#for tensor required for LSTM (batch, frame, data)
			for v_imgs, v_labels in V_data_loader:
				v_imgs = torch.squeeze(v_imgs, dim=1)
				v_labels = torch.squeeze(v_labels)

				# Randomising sequence of the Validation image set
				#rdm = torch.randperm(Variables.batch_size)
				#v_imgs = v_imgs.view(-1,frame_limit,Variables.channels*Variables.height*Variables.width)[rdm].view(v_imgs.size())
				#v_labels = v_labels.view(-1)[rdm].view(v_labels.size())
				#v_length_seq = v_length_seq.view(-1)[rdm].view(v_length_seq.size())

				with torch.no_grad():
					v_imgs = v_imgs.to(device)
					v_labels = v_labels.to(device)
					outputs = model(v_imgs, 0, epoch)
					loss = criterion(outputs, v_labels)

					validation_loss += loss.item() * v_labels.size(0)
					_, predicted = torch.max(outputs.data, 1)
					total += v_labels.size(0)
					correct += (predicted == v_labels).sum().item()

					i=0
					for row in outputs:
							x = torch.argmax(row)
							temp_number_occurence[x.item()] += 1
					Draw.number_occurence[:, epoch] = temp_number_occurence

		print(total)
		validation_loss /= total
		Draw.validation_loss_epoch[epoch] = validation_loss

		accuracy = 100.0 * correct / total
		Draw.validation_accuracy_epoch[epoch] = accuracy
		
		print("Train Loss: ", train_loss)
		print("Validation Loss: ", validation_loss)
		print("Train Accuracy: ", round(train_accuracy), "%")
		print ("Validation Accuracy: ", round(accuracy), "%" )

		if (train_loss <= best_train_loss):
				best_train_loss = train_loss
				best_tl_vl = validation_loss
				best_tl_ac = accuracy
				best_tl_model = model.state_dict()
		
		if (validation_loss <= best_validation_loss):
				best_validation_loss = validation_loss
				best_vl_tl = train_loss
				best_vl_ac = accuracy
				best_vl_model = model.state_dict()
		
		if (accuracy >= best_accuracy):
				best_accuracy = accuracy
				best_ac_vl = validation_loss
				best_ac_tl = train_loss
				best_ac_model = model.state_dict()

		print(Draw.number_occurence[:,epoch])
		end_time = datetime.now()
		Draw_graph(Draw.number_occurence, Draw.train_loss_epoch, Draw.validation_loss_epoch, Draw.train_accuracy_epoch, Draw.validation_accuracy_epoch, end_time - start_time, epoch, name)
		del v_input_seq, v_label_seq, v_imgs, v_labels
		gc.collect()

	torch.save(best_tl_model, './Sotsuron/{}/weights(tl).pth'.format(name))
	print('Best train loss: ', best_train_loss)
	print('Validation loss :', best_tl_vl)
	print('Accuracy during best loss :', best_tl_ac)
	print("\n")
	torch.save(best_vl_model, './Sotsuron/{}/weights(vl).pth'.format(name))
	print('Best validation loss: ', best_validation_loss)
	print('Train loss :', best_vl_tl)
	print('Accuracy during best validation :', best_vl_ac)
	print("\n")
	torch.save(best_ac_model, './Sotsuron/{}/weights(ac).pth'.format(name))
	print('Best accuracy: ', best_accuracy)
	print('Validation loss :', best_ac_vl)
	print('Train loss :', best_ac_tl)
	print("\n")

if __name__ == '__main__':
	name = input('What will be the name for the plot and CSV file?\n')
	main(name,0)
