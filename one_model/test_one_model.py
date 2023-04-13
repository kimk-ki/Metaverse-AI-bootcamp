import cv2
import tensorflow.keras
import numpy as np
import serial
import time
import pandas as pd

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ans = ['label', 'non_label', 'color']
count = 0
data_label = []
data_no_label = []
data_color = []



ser = serial.Serial('COM3', 9600)
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('new_learning_11.h5')
data = np.ndarray(shape=(1, 270, 480, 3), dtype=np.float32)
size = (480, 270)
img_num = 1

while True and count < 100:
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
            prediction = model.predict(data)
            index = np.argmax(prediction)
            print(prediction)

            resize_img = cv2.resize(image, (int(1920/4), int(1080/4)))
            cv2.imshow("a", image)
            cv2.waitKey(5000)

            cv2.destroyAllWindows()


            if index == 0:
                send = '1'
                send = send.encode('utf-8')
                ser.write(send)
                print(ans[index])
                time.sleep(0.5)

            elif index == 1:
                send = '2'
                send = send.encode('utf-8')
                ser.write(send)
                print(ans[index])
                time.sleep(0.5)
            
            elif index == 2:
                send = '3'
                send = send.encode('utf-8')
                ser.write(send)
                print(ans[index])
                time.sleep(0.5)
            flag = 0
    data_label.append(prediction[0][0])
    data_no_label.append(prediction[0][1])
    data_color.append(prediction[0][2])
    
    if count == 100:
        df = pd.DataFrame({'label' : data_label,
                           'no label' :  data_no_label,
                           'color' : data_color})
        df.to_csv('label_data.csv', index=False)


capture.release()
cv2.destroyAllWindows()
