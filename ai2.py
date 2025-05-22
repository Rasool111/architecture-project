import cv2
import os
import numpy as np

MODEL_DIR = "models"
KNOWN_FACES_DIR = "known_faces"

gender_model = os.path.join(MODEL_DIR, "gender_net.caffemodel")
gender_proto = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
age_model = os.path.join(MODEL_DIR, "age_net.caffemodel")
age_proto = os.path.join(MODEL_DIR, "age_deploy.prototxt")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

recognizer = cv2.face.LBPHFaceRecognizer_create()
known_faces = []
labels = []


def load_known_faces():
    label_id = 0
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    known_faces.append(img)
                    labels.append(label_id)
            label_id += 1


load_known_faces()
if known_faces:
    recognizer.train(known_faces, np.array(labels))


def analyze_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    report = []

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(frame[y:y + h, x:x + w], 1.0, (227, 227), (78.426, 87.769, 114.895), swapRB=False)

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        label, confidence = recognizer.predict(face_img) if known_faces else (-1, 100)
        name = "Unknown" if label == -1 else os.listdir(KNOWN_FACES_DIR)[label]

        label_text = f"{name}: {gender}, {age} (ثقة: {confidence:.2f}%)"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        report.append(label_text)

    return frame, report


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame, report = analyze_face(frame)
    cv2.imshow('Face Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFace Analysis Report:")
for line in report:
    print(line)
