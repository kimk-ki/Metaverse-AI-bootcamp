import cv2
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import serial
import time
import pandas as pd

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ans_1 = ['label', 'non_label']
ans_2 = ['non_label', 'color']
count = 0
num = []
data_label = []
data_no_label = []
data_color = []



ser = serial.Serial('COM3', 9600)
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model_1 = tensorflow.keras.models.load_model('label_detection.h5')
model_2 = tensorflow.keras.models.load_model('color_detection.h5')
data = np.ndarray(shape=(1, 270, 480, 3), dtype=np.float32)
size = (480, 270)
img_num = 1

while True and count < 1000:
    count += 1

    if ser.readable():
        val = ser.readline()
        flag = int(val.decode()[:len(val) - 1])
        
        if flag == 1:

            print("Trash detected")
            ret, frame = capture.read()
            print("캡쳐")
            # cv2.imwrite("temp.jpg", frame)
            image = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_AREA)
            # Normalize the image
            normalized_image_array = (image.astype(np.float32) / 127.0) - 1
            # Load the image into the array
            data[0] = normalized_image_array
            # run the inference
            prediction_1 = model_1.predict(data)
            prediction_2 = model_2.predict(data)
            # if prediction[0][1] * 1.5 > 0.5:
            #     index =1
            # elif prediction[0][0] > 0.5:
            #     index = 0
            # else:
            #     index = 2
            index_1 = np.argmax(prediction_1)
            index_2 = np.argmax(prediction_2)
            print(prediction_1)
            print(prediction_2)
            # cv2.imwrite(str(ans[index]) + str(img_num) + ".jpg", image)
            # img_num = img_num + 1

            resize_img = cv2.resize(image, (int(1920/4), int(1080/4)))
            cv2.imshow("a", image)
            cv2.waitKey(5000)

            cv2.destroyAllWindows()


            if index_1 == 0:
                send = '1'
                send = send.encode('utf-8')
                ser.write(send)
                print(ans_1[index_1])
                time.sleep(0.5)

            elif index_1 == 1 and index_2 == 0:
                send = '2'
                send = send.encode('utf-8')
                ser.write(send)
                print(ans_2[index_2])
                time.sleep(0.5)
            
            elif index_1 == 1 and index_2 == 1:
                send = '3'
                send = send.encode('utf-8')
                ser.write(send)
                print(ans_2[index_2])
                time.sleep(0.5)
            flag = 0
    # num.append(count)
    # data_label.append(prediction[0][0])
    # data_no_label.append(prediction[0][1])
    # data_color.append(prediction[0][2])
    
    # if count == 1000:
    #     df = pd.DataFrame({'label' : data_label,
    #                        'no label' :  data_no_label,
    #                        'color' : data_color})
    #     df.to_csv('no_label_data.csv', index=False)


capture.release()
cv2.destroyAllWindows()
