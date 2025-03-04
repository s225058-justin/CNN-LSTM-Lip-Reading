from datetime import datetime
from pynput import keyboard
import dlib
import torch
import cv2
import os
import shutil
from Webcam_Processing import get_tensor
from Webcam_Model import CNN_LSTM
from Webcam_Parameters import Variables

folder_list = ["Zero", "Ichi", "Ni", "San", "Yon", "Go", "Roku", "Nana", "Hachi", "Kyuu", "Arigatou", 
	           "Iie", "Ohayou", "Omedetou", "Oyasumi", "Gomennasai", "Konnichiwa", "Konbanwa", "Sayounara", "Sumimasen", 
               "Douitashimashite", "Hai", "Hajimemashite", "Matane", "Moshimoshi"]

def on_press(key):
    if(str(key) == "Key.enter"):
        global record_flag
        if(record_flag == 0):
            global start_time
            start_time = datetime.now()
        record_flag = 1

def on_release(key):
    if(str(key) == "Key.enter"):
        global record_flag
        record_flag = 2
        global i
        i = 0
        global start_time
        global end_time
        end_time = datetime.now()
        print("Seconds recorded: ", end_time - start_time)  
    if key == keyboard.Key.esc:
        # Stop listener
        return False

def face_detection(frame):
    start_time = datetime.now()
    detector = dlib.get_frontal_face_detector() #launches the face detector
    rects = detector(frame, 1)
    global face_flag
    if len(rects) == 0: # no faces detected
        face_flag = 0
    else:
        face_flag = 1
    end_time = datetime.now()
    print("Face detection: ", end_time - start_time)        
    return(face_flag, rects)

def clear_folders(folder_paths: list):
    for folder_path in folder_paths:
        os.makedirs(folder_path, exist_ok=True)
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        
        print(f"All contents of '{folder_path}' have been deleted.")

def main():
    clear_folders(['./Saved_images/Test/', './Cropped_images/Test/'])
    listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
    listener.start()
    global face_flag
    global record_flag
    face_flag = -1 # initial = -1, face not detected = 0, face detected = 1
    record_flag = 0 # not recording = 0, recording = 1
    model = CNN_LSTM(Variables.hidden_size, Variables.num_layers)
    model.load_state_dict(torch.load('./Pretrained_Model_Weights/weights.pth',map_location='cpu'))
    device = torch.device("cpu")
    model.to(device)

    model.eval()
    print("Please wait until the preview window for the webcam has opened.")
    cap = cv2.VideoCapture(0)
    if not (cap.isOpened()):
        print ("Could not open webcam")
    print("Pick any of the following words to say.")
    print(  " 0.Zero              ", "1.Ichi   ", "2.Ni             ", "3.San       ", "4.Yon         ", "5.Go          ", "6.Roku        ", "7.Nana      ", "8.Hachi      ", "9.Kyuu       \n", 
            "10.Arigatou         ", "11.Iie   ", "12.Ohayou        ", "13.Omedetou ", "14.Oyasumi    ", "15.Gomennasai ", "16.Konnichiwa ", "17.Konbanwa ", "18.Sayounara ", "19.Sumimasen \n", 
            "20.Douitashimashite ", "21.Hai   ", "22.Hajimemashite ", "23.Matane   ", "24.Moshimoshi ")
    print("Press and hold Enter to start recording and release once you're done speaking.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    global i
    i = 0
    while(True):
        ret, frame = cap.read()
        cv2.imshow('preview', frame)
        if (record_flag == 1):
            i += 1
            if (i == 1):
                print("RECORDING")
            cv2.imwrite('./Saved_images/Test/img_{:0>3}.jpg'.format(i), frame)
        elif(record_flag == 2):
            input_seq = get_tensor()
            target = [1, 25, Variables.channels,32,32]
            input_seq =  input_seq[:, :, :, :].expand(target)
            input_seq = input_seq.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(input_seq)
                x = torch.argmax(outputs).item()
                print("Predicted word = " + folder_list[x])
                Softmax = torch.nn.Softmax(dim=0)
                confidence_score = round(Softmax(outputs[0])[x].item()*100)
                print("Confidence: ", confidence_score, "%")
            record_flag = 0
            clear_folders('./Saved_images/Test/')
            print("Pick any of the following words to say.")
            print(  " 0.Zero              ", "1.Ichi   ", "2.Ni             ", "3.San       ", "4.Yon         ", "5.Go          ", "6.Roku        ", "7.Nana      ", "8.Hachi      ", "9.Kyuu       \n", 
                    "10.Arigatou         ", "11.Iie   ", "12.Ohayou        ", "13.Omedetou ", "14.Oyasumi    ", "15.Gomennasai ", "16.Konnichiwa ", "17.Konbanwa ", "18.Sayounara ", "19.Sumimasen \n", 
                    "20.Douitashimashite ", "21.Hai   ", "22.Hajimemashite ", "23.Matane   ", "24.Moshimoshi ")
            print("Press and hold Enter to start recording and release once you're done speaking.")

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
	main()