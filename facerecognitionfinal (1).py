import cv2
import numpy
import os
import time
from datetime import timedelta

size = 4
haar_cascade = "haarcascade_frontalface_default.xml"
dataset = "dataset"
print("Training...\nInitialising process\n")
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(dataset):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(dataset, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (400, 400)
(images, labels) = [numpy.array(lis) for lis in [images, labels]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_cascade)
cam = cv2.VideoCapture(0)
cnt = 0
person_in_frame = {}
start_time = {}

while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        faces = gray[y:y + h, x:x + w]
        resize = cv2.resize(faces, (width, height))

        prediction = model.predict(resize)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        if prediction[1] < 800:
            person_name = names[prediction[0]]
            cv2.putText(img, '%s-%d' % (person_name, prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            print(person_name)

            if person_name not in person_in_frame:
                person_in_frame[person_name] = True
                start_time[person_name] = time.time()
            else:
                elapsed_time = time.time() - start_time[person_name]
                print(f"{person_name} has been in the frame for {str(timedelta(seconds=elapsed_time))}")

            cnt = 0
        else:
            cnt += 1
            cv2.putText(img, 'Unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("input.jpg", img)
                cnt = 0

            # Remove person from person_in_frame dictionary if they are no longer detected
            for person in list(person_in_frame.keys()):
                if person not in names[prediction[0]]:
                    person_in_frame.pop(person)
                    start_time.pop(person)

    cv2.imshow('OpenCV', img)
    key = cv2.waitKey(10)
    if key == ord("q"):
        # Print the final time a person was in the frame
        for person, start in start_time.items():
            elapsed_time = time.time() - start
            print(f"{person} was in the frame for a total of {str(timedelta(seconds=elapsed_time))}")
        break
cam.release()
cv2.destroyAllWindows()