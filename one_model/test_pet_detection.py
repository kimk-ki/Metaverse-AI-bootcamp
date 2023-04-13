from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
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
ans = ['label', 'non_label', 'color']
count = 0
num = []
data_label = []
data_no_label = []
data_color = []



ser = serial.Serial('COM3', 9600)
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

net = cv2.dnn.readNet("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# Load the model
model = tensorflow.keras.models.load_model('new_learning_13.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
size = (224, 224)
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
            (h, w) = image.shape[:2]
            
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            
            for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
                confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
                if confidence > 0.01:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # ensure the bounding boxes fall within the dimensions of
                    # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # extract the face ROI, convert it from BGR to RGB channel
                    # ordering, resize it to 224x224, and preprocess it
                    face = image[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)

                    # pass the face through the model to determine if the face
                    # has a mask or not
                    (Label, withoutLabel, withcolor) = model.predict(face)[0]

                    # determine the class label and color we'll use to draw
                    # the bounding box and text
                    label = "Label" if Label > 0.5 else "No label"
                    label = "Color" if withcolor > 0.5 else "No label"
                    color = (0, 255, 0) if label == "Label" else (0, 0, 255)
                    color = (255, 0, 0) if label == "Color" else (0, 0, 255)
                    
                    if label == "Label":
                        index = 0
                    elif label == "No label":
                        index = 1
                    elif label == "Color":
                        index = 2

                    # include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(Label, withoutLabel, withcolor) * 100)

                    # display the label and bounding box rectangle on the output
                    # frame
                    cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

            
            # # Normalize the image
            # normalized_image_array = (image.astype(np.float32) / 127.0) - 1
            # # Load the image into the array
            # data[0] = normalized_image_array
            # # run the inference
            # prediction = model.predict(data)
            # # if prediction[0][1] * 1.5 > 0.5:
            # #     index =1
            # # elif prediction[0][0] > 0.5:
            # #     index = 0
            # # else:
            # #     index = 2
            # index = np.argmax(prediction)
            # print(prediction)
            # # cv2.imwrite(str(ans[index]) + str(img_num) + ".jpg", image)
            # # img_num = img_num + 1

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
