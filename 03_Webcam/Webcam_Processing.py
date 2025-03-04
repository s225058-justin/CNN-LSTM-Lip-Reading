import dlib
import numpy as np
import cv2
from imutils import face_utils
from Webcam_Parameters import Variables
from torch.utils.data import Dataset, DataLoader
import glob
import torch
import math
import torchvision.transforms.functional as TF
import os

#defining a custom dataset using pytorch's dataloader
frame_limit = 25 #the number of frames extracted from each folder
folder_no = 60
width = 32
height = 32

def save_tensors():
        V_data_loader = DataLoader(ValidationDataset(), batch_size=frame_limit, shuffle=False, num_workers = 10)
        v_input_seq = torch.empty((0,Variables.channels,height,width), dtype=torch.float32)
        for v_imgs in V_data_loader:
            v_input_seq = torch.cat((v_input_seq, v_imgs), dim=0)
        return v_input_seq

def lip_detection(frame, rects):
    predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks_GTX.dat') #launches the dataset used for face detector based on the argument inputed at launch
    for(i, rect) in enumerate(rects): #for i number of faces/rectangles
        shape = predictor(frame, rect)               #detecting landmarks on each face
        shape = face_utils.shape_to_np(shape)       #converting landmarks to a numpy array for processing
        return shape

def face_detection(frame):
    detector = dlib.get_frontal_face_detector() #launches the face detector
    rects = detector(frame, 1)
    global face_flag
    if len(rects) == 0: # no faces detected
        face_flag = 0
        print("No face found!")
    else:
        face_flag = 1
    return(face_flag, rects)

def get_lips():
    number = 0
    path = './Saved_images/Test'
    for images in sorted(glob.glob(path + "/*.jpg")):
        print(images)
        number += 1
        frame = cv2.imread(images)
        _, rects = face_detection(frame)
        if (face_flag != 0):
            shape = lip_detection(frame, rects)
            (x,y,w,h) =  face_utils.rect_to_bb(rects[0])
                         
            mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.float32) #no idea
            cv2.fillConvexPoly(mask, np.int32(shape[48:60]), (1.0, 1.0, 1.0)) #filling in the rest of the mask
            mask = 255*np.uint8(mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10)) #no idea
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1) #closing up the mask so it's more accurate
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            masks = 0 #initialise a global variable required later
            masks += mask

            #making a white background
            inverseMasks = cv2.bitwise_not(mask)
            # for only lips, use just the below code and images to mask_out
            mask_out = cv2.subtract(frame, inverseMasks)

            #reading the cropped out lips so we can find the contours
            new_gray = cv2.cvtColor(masks, cv2.COLOR_BGR2GRAY) #setting the colour of the mask to gray
            ret, thresh = cv2.threshold(new_gray, 175, 255, cv2.THRESH_BINARY) #converting the image to pure black and white, anything that's not within the set threshold gets set to black and anything that is gets set to white
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #finds contours inside the image (finding contours is basically just detecting white shapes on a black 	background)
            contours = list(filter(lambda x: cv2.contourArea(x) >= 80, contours)) #filters out contours smaller than 80px

            # cropping out the lips
            xl, yl, xwl, yhl = 1000, 1000, 0, 0
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
            
            image = cv2.resize(mask_out[yl-10:yhl+10,xl:xwl], (Variables.width, Variables.height),interpolation = cv2.INTER_AREA)
            path = './Cropped_images/Test'
            os.makedirs(path, exist_ok = True)
            cv2.imwrite('./Cropped_images/Test/img_{:0>3}.jpg'.format(number), image)
	
class ValidationDataset(Dataset):
    def __init__(self):
        self.imgs_path = './Cropped_images/Test/'
        file_list = sorted(glob.glob(self.imgs_path + "/*.jpg"))
        self.data = []
        i = 1
        t = len(file_list)

        x = t/frame_limit
        ima = 1
        img_array = []
        for image_file in sorted(glob.glob(self.imgs_path + "/*.jpg")):
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
        
        for img in range(len(img_array)):
            image_name2 = '{:0>3}'.format(img_array[img])
            image_name2 = 'img_' + image_name2 + '.jpg'
            self.data.append('./Cropped_images/Test/' + image_name2)
        self.img_dim = (width, height)
		
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_dim)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        if (Variables.channels == 1):
            img_tensor = TF.rgb_to_grayscale(img_tensor,1)
        return img_tensor
	
def get_tensor():
	#for tensor required for LSTM (batch, frame, data)
    print("Processing Webcam Data")
    get_lips()
    tensor = save_tensors()
    return tensor

if __name__ == '__main__':
	get_tensor()
