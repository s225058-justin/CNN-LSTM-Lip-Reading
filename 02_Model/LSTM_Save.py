from LSTM_Parameters import Variables
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

class Drawing():
    plot_shape = (Variables.num_epochs)
    train_accuracy_epoch = np.empty(plot_shape)
    train_loss_epoch = np.empty(plot_shape)
    validation_loss_epoch = np.empty(plot_shape)
    validation_accuracy_epoch = np.empty(plot_shape)
    number_occurence_shape = (Variables.words, Variables.num_epochs)
    number_occurence = np.zeros(number_occurence_shape)
    train_accuracy_epoch.fill(np.nan)
    train_loss_epoch.fill(np.nan)
    validation_accuracy_epoch.fill(np.nan)
    validation_loss_epoch.fill(np.nan)
    number_occurence.fill(np.nan)

def Draw_graph(number_occurence, train_loss_epoch, validation_loss_epoch, train_accuracy_epoch, validation_accuracy_epoch, dt_string, current_epoch, name):
    fig = plt.figure(figsize=(20,10))

    # Accuracy
    ax1 = fig.add_subplot(221)
    ax1.set_title('Accuracy')
    ax1.plot(range(Variables.num_epochs),train_accuracy_epoch, '-')
    ax1.plot(range(Variables.num_epochs),validation_accuracy_epoch, '-')
    ax1.axhline(y=80, linewidth=1, linestyle = '--', color='r')
    ax1.axvline(x=5, linewidth=1, linestyle = '--', color='b')
    ax1.axis([0, Variables.num_epochs-1, 0, 100])
    ax1.set(xlabel='Epoch', ylabel='Accuracy')
    ax1.legend(['Train Accuracy','Validation Accuracy'], loc='upper left')
    
    # Loss
    ax2 = fig.add_subplot(222)
    ax2.set_title('Loss')
    ax2.plot(range(Variables.num_epochs),train_loss_epoch, '-')
    ax2.plot(range(Variables.num_epochs),validation_loss_epoch, '-')
    ax2.axhline(y=0.8, linewidth=1, linestyle = '--', color='r')
    ax2.axvline(x=5, linewidth=1, linestyle = '--', color='b')
    ax2.axis([0, Variables.num_epochs-1, 0, 5])
    ax2.set(xlabel='Epoch', ylabel='Loss')
    ax2.legend(['Train Loss','Validation Loss'], loc='upper left')

    # Number occurences
    ax3 = fig.add_subplot(223)
    ax3.set_title('Occurence')
    for i in range(Variables.words):
        ax3.plot(range(Variables.num_epochs),number_occurence[i], '-')
    ax3.axhline(y=120*Variables.val_groups, linewidth=1, linestyle = '--', color='g')
    ax3.axis([0, Variables.num_epochs-1, 0, 500])
    ax3.set(xlabel='Epoch', ylabel='Occurences of numbers')

    # Details
    ax4 = fig.add_subplot(224)
    ax4.set_axis_off()
    ax4.text(0, 0.95, 'Batch size for training: ' + str(Variables.train_groups*Variables.People))
    ax4.text(0, 0.90, 'Batch size for validation: ' + str(Variables.val_groups*(Variables.People)))
    ax4.text(0, 0.85, 'Learning rate: ' + str(Variables.learning_rate))
    ax4.text(0, 0.80, 'Number of words: ' + str(int(Variables.words)))
    ax4.text(0, 0.75, 'Picture size: ' + str(Variables.width) + '*' + str(Variables.height))
    ax4.text(0, 0.70, 'Number of pictures for training: ' + str(Variables.train_groups*Variables.frame_limit*Variables.folder_no*Variables.People))
    ax4.text(0, 0.65, 'Number of pictures for validation: ' + str(Variables.val_groups*Variables.frame_limit*Variables.folder_no*Variables.People))
    ax4.text(0, 0.60, 'Number of channels per picture: ' + str(Variables.channels))
    ax4.text(0, 0.55, 'Execution time: ' + str(dt_string))
    ax4.text(0, 0.50, 'Execution time per epoch: ' + str(dt_string/(current_epoch+1)))
    plt.tight_layout()

    plt.savefig('../Saved Plots/Current plot.png')
    if(current_epoch + 1 == Variables.num_epochs):
        os.makedirs('../{}/'.format(name), exist_ok=True)
        plt.savefig('../Saved Plots/' + name + '.png')
        plt.savefig('../{}/'.format(name) + name + '.png')
        # Saving to CSV
        data = {'Train Loss': train_loss_epoch,
            'Val Loss': validation_loss_epoch,
            'Train Accuracy': train_accuracy_epoch,
            'Val Accuracy': validation_accuracy_epoch}
        df = pd.DataFrame.from_dict(data)
        df.to_csv('../CSV_Folder/' + name + '.csv')
        df.to_csv('../{}/'.format(name) + name + '.csv')
    plt.close()
