# 会話支援のための機械学習を用いた読唇システムの開発
This repository stores the programs and results from my graduation research at Yonago National Institute of Technology on developing a Japanese Lip-Reading AI System.
A demo is available and the steps to test it out are available below. 

## The demo
This repository contains the code I used for preprocessing and training, in the folders `01_Dataset` and `02_Model`. The demo is contained in `03_Webcam`. In order to run the demo, you must have a webcam. The steps are as follows:
1) Install the requirements stated in `requirements.txt`
2) Download the `shape_predictor_68_face_landmarks_GTX.dat` model from [this link](https://drive.google.com/drive/folders/1t1fRQfTaL1-XgGA1JSzuvLSXsitZ6Scj) and place it in the `dlib_shape_predictor` folder in `03_Webcam`.
3) Run the Webcam_LivePredict.py file using
```bash
python3 Webcam_LivePredict.py
```
4) Wait until the webcam preview loads and choose a word to say. Since audio is not required for recognition, it is fine to mouth it silently.
5) Press and hold the `Enter` key, say your chosen word and then release the `Enter` key.
6) Wait for the model to predict the word

## Training the model
### The dataset
I utilised the SSSD Dataset provided by Kyushu Institute of Technology's 齊藤・張 研究室, whose website can be found at https://www.saitoh-lab.com/. Due to the confidentiality of the dataset, parts of the code used for training has been deleted and that dataset is not included in this repository to maintain confidentiality. As a result, it is not intended for the training of the model to be reimplemented through this repository.

The SSSD dataset comprises of labeled short videos of 25 words, spoken 10 times by 72 speakers shot at 30fps with no audiotrack. The images are cropped to focus around the mouths of the participants. The 25 words are as follows:

| #  | 発話内容   | #  | 発話内容       | #  | 発話内容       |
|----|----------|----|--------------|----|--------------|
| 0  | ぜろ      | 10 | ありがとう     | 20 | どういたしまして |
| 1  | いち      | 11 | いいえ         | 21 | はい          |
| 2  | に        | 12 | おはよう       | 22 | はじめまして   |
| 3  | さん      | 13 | おめでとう     | 23 | またね        |
| 4  | よん      | 14 | おやすみ       | 24 | もしもし      |
| 5  | ご        | 15 | ごめんなさい   |    |              |
| 6  | ろく      | 16 | こんにちわ     |    |              |
| 7  | なな      | 17 | こんばんわ     |    |              |
| 8  | はち      | 18 | さようなら     |    |              |
| 9  | きゅう    | 19 | すみません     |    |              |

### Preprocessing
To reduce the amount of data utilised for training and lower training time, I set the frame limit of each video to 25 frames. To achieve this, videos with more than 25 frames were downsampled to 25 frames, whereas videos with less than 25 frames utilised padding to achieve 25 frames. 

A video with only 17 frames is padded to 25 frames.

![Padding](https://github.com/user-attachments/assets/22b43625-2adb-4382-a446-e2530b8fa0d7)

A video with 49 frames is downsampled to 25 frames.

![Sampling](https://github.com/user-attachments/assets/6d458d67-f9aa-4f9e-8eb1-267efc92dc53)

To reduce the amount of redundant information learned during training, I took several steps to lower the amount of data available.
1) First, I converted the images to monochrome
2) I cropped out the lips of each participants and removed any facial and background details
3) I resized the images from 300px X 300px to 32px X 32px.
These steps lowered the amount of data to a mere 0.18% of the original size, while retaining much of the relevant data required for training a model.

The original photo

![uncropped](https://github.com/user-attachments/assets/9bbd5fed-8642-4b42-99f6-980d53f82158)

The cropped black and white version used for training

![lipsonlybnw](https://github.com/user-attachments/assets/3aa91e21-fa81-405a-a7cf-8b69e3d79972)

After processing the images, I preconverted the videos into tensors, to save time from loading each video when optimising the hyperparameters of the model during training. Hence, during the training, the script would load precompiled tensors, instead of images, saving significant training time.

I also experimented with other picture sizes, color and preprocessing effects, but the black and white lips were my final choice. Please consult the full paper if interested.

### Synthetic data
To increase the robustness of my data, I also implemented synthetic data generation. I allowed the model to learn on unmodified data for the first 5 epochs, and then enabled a transformation function with a 60% chance to modify the videos. The transformations that could then affect the videos were as follows:

1. Blurring : 動画にランダムな強さでモザイク処理
2. Flipping : 動画を水平に反転
3. Perspective : 動画の観点をランダムな数字で変換
4. Rotation : ±15◦ 以内に動画を回転
5. Brightness : 動画の明るさを ±20%以内に変換
6. Contrast : 動画のコントラストを ±25%以内に変換
7. Saturation : 動画の鮮やかさを ±25%以内に変換
8. Gamma : 動画のガンマを ±25%以内に変換

After the transformation function is triggered, each transformation is assigned their own probability of triggering, meaning that multiple types of transformations can occur on the same video. This significantly increased the variety of data available for training by varying the types of transformations a video could be modified with. The table of probabilities for each transformation is listed below.

| 変換         | 変換関数確率 (%) | 変換確率 (%) | 合計確率 (%) |
|------------|--------------|----------|----------|
| Blurring   |              | 30       | 18       |
| Perspective |              |          |          |
| Rotation   |              |          |          |
| Flipping   | 60           |          |          |
| Brightness |              |          |          |
| Contrast   |              |   50       | 30       |
| Saturation |              |          |          |
| Gamma      |              |          |          |

This means that 18% of videos after the 5th epoch have the blurring and perspective transformations, whereas 30% of videos have the rotation, flipping, brightness, contrast, saturation and gamma transformations applied.

### The model
I selected a CNN-LSTM architecture, where the CNN extracts spatial features from frames, while the LSTM captures temporal patterns in the videos. Dropout layers were also strategically used in the CNN architecture to reduce overfitting. The architecture of the model is as follows:

![CNN_LSTM-model](https://github.com/user-attachments/assets/b05807a6-3214-4cb1-b8f4-aa44d74218ce)

The model was then trained for 50 epochs with a 83:17 split for training and validation.

### The results
After training for 50 epochs, the model achieved an accuracy of around 80%. Increasing the epoch count did not increase accuracy but caused the model to overfit instead. The picture below describes the results of training the model. In the occurence tab, it shows the number of times each word was predicted and an ideal result would be all words predicted 120 times as in validation there are 12 speakers.

![Results](https://github.com/user-attachments/assets/5a49fcae-44f6-4696-a6ec-1679ddf3df4c)

I further validated my findings by creating a demo version of my model and gathered video data from 10 students to test my model. Each student would record themselves saying each word 10 times for a total of 250 videos per student. The videos were then preprocessed the same way as the training data and then fed into the model using weights saved from the training. As a result, the total accuracy was 70.64% and I discovered that shorter words were harder for the system to read than longer words. In particular, the word ゴ had the least true positives and the word イチ had the most false positives. Longer words like さようなら、はじめまして and こんばんは had lowest occurences of false positives and a good ratio of true positives. The results are as below:

![total_results1](https://github.com/user-attachments/assets/40e2145c-10ba-43d3-b658-ad90da92bc13)

Furthermore, of the 10 students, 7 were native Japanese students while 3 were foreign students. I discovered that Japanese students had an accuracy of 74% whereas foreign students had an accuracy of 63%. An assumption I have is that since the model was trained on data from native Japanese speakers, the foreign students had a lower accuracy as they did not necessarily pronounce words the way an average Japanese speaker would. However, I did not test this assumption for robustness and it remains an assumption.
