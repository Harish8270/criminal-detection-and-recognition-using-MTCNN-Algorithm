import cv2
import os
from facenet_pytorch import MTCNN
import csv


video = cv2.VideoCapture(0)

count = 0
password = int(input("enter password : "))
criminal_Id = str(input("Enter criminal ID :"))
name = str(input("Enter criminal Name: ")).title()
age = str(input("Enter criminal age: "))
gender = str(input("Enter criminal gender: "))
crime_level = str(input("Enter level of criminal : "))
crime_type = str(input("Enter type of crime type : "))
identity_mark = str(input("Enter type of criminal identity : "))
address = str(input("Enter Your Address: "))

path = 'images/' + name

if os.path.exists(path):
    print("Name already exists")
    name = str(input("Enter Your Name Again: ")).title()
    path = 'images/' + name
else:
    os.makedirs(path)

# CSV file to store image paths, name, and address
csv_file = 'image_data.csv'
header = ['Criminal_ID', 'Name', 'Age', 'Gender', 'Criminal_level', 'Crime_type', 'Identity_mark', 'Address', 'Image_Path']
if password != 1:
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

# Create MTCNN detector
detector = MTCNN()

while True:
    ret, frame = video.read()

    # Detect faces in the current frame using MTCNN
    boxes, probs = detector.detect(frame)

    if boxes is not None:
        for box in boxes:
            count += 1
            x, y, width, height = box
            img_name = f"{path}/{count}.jpg"
            print("Creating Images........." + img_name)
            cv2.imwrite(img_name, frame[int(y):int(y+height), int(x):int(x+width)])
            cv2.rectangle(frame, (int(x), int(y)), (int(x+width), int(y+height)), (0, 0, 255), 3)
            # Write image path, name, and address to CSV file
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([criminal_Id, name, age, gender, crime_level, crime_type, identity_mark, address, img_name])

    cv2.imshow("Video", frame)
    cv2.waitKey(1)

    if count > 10:
        break

video.release()
cv2.destroyAllWindows()
