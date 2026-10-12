import cv2
import torchvision
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

names = {0: "Noel"}
max_distance = 70

face_model = YOLO("yolov8m-face.pt")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognizer.yml")

while cv2.waitKey(1) != ord("x"):
    _, frame = cap.read()

    face_result = face_model(frame, verbose=False)
    processed_feed = face_result[0].plot()

    for box in face_result[0].boxes.xyxy:
        left, top, right, bottom = box.int()
        face = frame[top:bottom, left:right]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        cv2.resize(face, (200, 200))

        pred_label, distance = face_recognizer.predict(face)

        if distance > max_distance:
            name = "Unknown"
        else:
            name = names[pred_label]

        cv2.putText(
            processed_feed,
            name + "|" + str(int(distance)),
            (int(left), int(bottom) + 20),
            0,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("my window", processed_feed)
        cv2.moveWindow("my window", 600, 200)

cv2.waitKey(1000)
cap.release()
cv2.destroyAllWindows()




