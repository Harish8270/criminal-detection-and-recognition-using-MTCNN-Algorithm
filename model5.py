import cv2
import pandas as pd
import geocoder
from twilio.rest import Client
from facenet_pytorch import InceptionResnetV1, MTCNN, extract_face
import torch
import numpy as np
import smtplib
from email.message import EmailMessage

# Function to extract facial features using InceptionResnetV1
def extract_features(frame, mtcnn, resnet_model):
    faces, _ = mtcnn.detect(frame)

    if faces is not None:
        features = []
        for face in faces:
            face_tensor = extract_face(frame, face, image_size=160)
            face_tensor = face_tensor.permute(1, 2, 0).numpy()
            face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
            face_tensor = face_tensor.transpose((2, 0, 1))
            face_tensor = np.expand_dims(face_tensor, axis=0)
            face_tensor = torch.from_numpy(face_tensor).float()
            features.append(resnet_model(face_tensor))
        return faces, features
    else:
        return None, []

# Function to send an email
def send_email(recipients, criminal_id, name, age, gender, criminal_level, crime_type, identity_mark, address, latitude, longitude):
    EMAIL_ADDRESS = 'hsh28643@gmail.com'  # your email address
    EMAIL_PASSWORD = 'zdyi lbjr mkdf qcih'  # your email password

    msg = EmailMessage()
    msg['Subject'] = 'Criminal Found!'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = recipients  # recipients should be a list of email addresses
    msg.set_content(f"Criminal_ID: {criminal_id}\nName: {name}\nAge: {age}\nGender: {gender}\nCriminal_level:{criminal_level}\nCrime_type: {crime_type}\nIdentity_mark:{identity_mark}\nAddress: {address}\nLive Location - Latitude: {latitude}, Longitude: {longitude}")

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

# Function to send an SMS
def send_sms(recipients, name, address, latitude, longitude):
    account_sid = 'AC09178cf8d4ecc00af7180907f7e5c9aa'  # Twilio account SID
    auth_token = 'd791f723171021c36c969aaa0b527390'  # Twilio auth token
    client = Client(account_sid, auth_token)

    for recipient in recipients:
        message = client.messages.create(
            body=f"Match found!\nName: {name}\nAddress: {address}\nLive Location - Latitude: {latitude}, Longitude: {longitude}",
            from_='+12134234529',  # Twilio phone number
            to=recipient  # recipient phone number
        )

        print(f"Message sent to {recipient}: {message.sid}")

def process_frame(frame, dataset, mtcnn, resnet_model):
    faces, live_features = extract_features(frame, mtcnn, resnet_model)
    count1=0
    if faces is not None:
        for idx, row in dataset.iterrows():
            dataset_image_path = row['Image_Path']
            dataset_image = cv2.imread(dataset_image_path)
            _, dataset_features = extract_features(dataset_image, mtcnn, resnet_model)

            for face, live_feature in zip(faces, live_features):
                count =0
                for dataset_feature in dataset_features:
                    distance = torch.sqrt(torch.sum((live_feature - dataset_feature) ** 2)).item()

                    # Use a fraction of the minimum distance as the threshold (you can adjust this fraction)
                    threshold = 0.5 * distance
                    count1+=1
                    if threshold <0.4:
                        count =count+1

            if(count!=0):
                # Send notifications (emails and SMS)
                #print(count)
                print(row['Name' ])
                criminal_id = row['Criminal_ID']
                name = row['Name']
                age = row['Age']
                gender = row['Gender']
                criminal_level = row['Criminal_level']
                crime_type = row['Crime_type']
                identity_mark = row['Identity_mark']
                address = row['Address']
                g = geocoder.ip('me')
                latitude, longitude = g.latlng  # actual live location

                # Send emails
                email_recipients = ['21eucs508@skcet.ac.in']
                send_email(email_recipients, criminal_id, name, age, gender, criminal_level, crime_type, identity_mark, address, latitude, longitude)

                # Send SMS
                sms_recipients = ['+918270955965']
                send_sms(sms_recipients, name, address, latitude, longitude)
                break

# Load dataset from CSV file
csv_file_path = 'image_data.csv'
dataset = pd.read_csv(csv_file_path)

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Initialize InceptionResnetV1 for face recognition
resnet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    cv2.imshow('Video', frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    process_frame(rgb_frame, dataset, mtcnn, resnet_model)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
