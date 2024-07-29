import cv2
import os

dataset = "dataset"
name = "Shreya"

path = os.path.join(dataset, name)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = 400, 400
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)
cam = cv2.VideoCapture(0)
count = 1
while count < 100:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(gray, 2, 4)
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        faceOnly = gray[y:y+h, x:x+w]
        resize = cv2.resize(faceOnly, (width, height))

        # Save images under the name folder
        cv2.imwrite(os.path.join(path, f"{name}_{count}.jpg"), resize)
        count += 1
    cv2.imshow("Face Detection", img)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
    print(count)
print("Image Captured Succesfully")
cam.release()
cv2.destroyAllWindows()