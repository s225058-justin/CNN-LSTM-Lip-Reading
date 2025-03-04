import numpy as np
import cv2
import os
import glob
import csv
from math import trunc
from tqdm import tqdm 

folder_list = []
with open("", 'r') as file:
	csvreader = csv.reader(file)
	foldernamelist = list(csvreader)
	length = len(foldernamelist)
	
for i in tqdm(range(length)):
	folder_name = str(foldernamelist[i])[9:21]
	m = trunc(i/3000)
	if (i%3000 != 0 or i == 0,3000,6000,9000,12000,15000):
		m += 1
	word_name = folder_name[6:8]
	person_name = int(folder_name[2:4]) - 1
	if (os.path.exists('' + str(m) + '/' + folder_name) == False):
		os.makedirs('' + str(m) + '/' + folder_name, exist_ok=True)
		a = glob.glob('' + str(m) + '/' + folder_name + '/')
		array = np.loadtxt('' + folder_name + '.csv', delimiter=',' , usecols=(range(96,120)))
		i=0
		masks = 0
		for b in sorted(a):
			xl = 1000
			yl = 1000 
			xwl = 0 
			yhl = 0
			filepaths = glob.glob(b + '*.jpg')
			for f in sorted(filepaths):
				image = cv2.imread(f)
				background = np.full_like(image, (0,0,0)) 
				x_array = array[i, :]
				x_array = np.array(np.reshape(x_array, (-1,2)), dtype='int16')
				mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
				cv2.fillConvexPoly(mask, np.int32(x_array), (1.0, 1.0, 1.0)) #filling in the rest of the mask
				mask = 255*np.uint8(mask)
				kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10)) #no idea
				mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1) #closing up the mask so it's more accurate
				kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
				mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
				mask = cv2.dilate(mask, kernel, iterations=2)
				masks = masks + mask 
				
				#reading the cropped out lips so we can find the contours
				new_gray = cv2.cvtColor(masks, cv2.COLOR_BGR2GRAY) #setting the colour of the mask to gray
				ret, thresh = cv2.threshold(new_gray, 175, 255, cv2.THRESH_BINARY) #converting the image to pure black and white, anything that's not within the set threshold gets set to black and anything that 	is gets set to white
				contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #finds contours inside the image (finding contours is basically just detecting white shapes on a black 	background)
				contours = list(filter(lambda x: cv2.contourArea(x) >= 80, contours)) #filters out contours smaller than 80px
				name = os.path.basename(f)
				
				# cropping out the lips
				for cntr in contours: #for center in contours
					# get bounding boxes
					pad = 10 #padding = 10px
					x,y,w,h = cv2.boundingRect(cntr) #making a bounding rectangle that starts from the top left of the contour (x,y) and spans w long and h tall
					if (y-15 < yl): 
						yl = y-15
					if (y+h+15 > yhl):
						yhl = y+h+15
					if (x-15 < xl):
						xl = x-15
					if (x+w+15 > xwl):
						xwl = x+w+15
				i += 1
				image = cv2.resize(image[yl-10:yhl+10,xl:xwl], (128, 128),interpolation = cv2.INTER_AREA)
				cv2.imwrite('' + str(m) + '/' + folder_name + '/' + name, image) #saves the image
		i=0


		
