# Importing the required libraries
import json, sys
import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
from helper.preprocess import preprocess_image, face_detector


# Initializes the webcam to capture lie video
video_capture = cv.VideoCapture(1)


employee_full_name = input(" Please input your name")
employee_ID = input(" Please input your Employee ID")
count = 0

# Collect 100 samples of the face from webcam
while True:
        _, img = video_capture.read()
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)            

        face, coord = face_detector(gray_img)
        x,y,w,h = coord

        if face is not None:
            face = preprocess_image(face)
            count = count + 1 
            # Save file in specified directory
            image_name_path = "./faces/employees/" + employee_full_name + "." + employee_ID + "." + str(count) + ".jpg"
            cv.imwrite(image_name_path, face)


            # Display Live Count
            cv.putText(gray_img, str(count), (x+10,y-30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

            cv.imshow("Face Cropper", gray_img)
        
        else:
            cv.imshow("Face Cropper", gray_img)
        
        if (cv.waitKey(1) & 0xFF == ord("q")) or count == 100:
            break


if count == 100:
    print("Training Model")

    # Get the training data
    data_path = "./faces/employees/"
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    # Create list for training data and labels

    training_data, labels = [], []

    # Creating a numpy array for training data

    labels_to_name = {}

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        training_data.append(np.asarray(images, dtype=np.uint8))
        labels.append(i)
        full_name, employee_id, _, _ = files.split(".")
        labels_to_name[i] = {"full_name": full_name, "employee_id": employee_id}


    with open("labels_to_name.json", "w") as write_file:
        json.dump(labels_to_name, write_file)

    # Creating a numpy array for both training 
    labels = np.asarray(labels, dtype=np.int32)
    training_data = np.asarray(training_data, dtype=np.uint8)
    # Initialize the facial recognizer
    model = cv.face.LBPHFaceRecognizer_create()

    model.train(training_data, labels)
    model.write("trainer/model.xml")

    print("Models Trained Successfully.")
        
video_capture.release()
cv.destroyAllWindows()

print("Samples Collected.")
