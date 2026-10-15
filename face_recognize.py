import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from ultralytics import YOLO


class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition & Rating Control Panel")
        self.root.geometry("380x320")
        
        self.names = {0: "Noel"}
        self.max_distance = 70

        self.tiers = [
            "sub5", "sub3", "ltn", "mtn",
            "htn", "chadlite", "chad", "adamlite", "true adam"
        ]

        self.face_tiers = {}

        print("Loading models...")
        self.face_model = YOLO("yolov8n-face.pt")
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_recognizer.read("face_recognizer.yml")
        print("Models loaded!")

        tk.Label(root, text="Face Detection Control Panel", font=("Arial", 14, "bold")).pack(pady=15)

        self.btn_webcam = tk.Button(root, text="Start Live Camera (Standard)", width=28, height=2, bg="green",
                                    fg="white", command=self.start_webcam)
        self.btn_webcam.pack(pady=5)

        self.btn_rate = tk.Button(root, text="Rate Looks (2+ Faces Required)", width=28, height=2, bg="purple",
                                  fg="white", command=self.start_look_rating)
        self.btn_rate.pack(pady=5)

        self.btn_image = tk.Button(root, text="Detect from Image File", width=28, height=2, bg="blue", fg="white",
                                   command=self.process_image)
        self.btn_image.pack(pady=5)

        self.btn_quit = tk.Button(root, text="Quit", width=28, height=2, bg="red", fg="white", command=root.quit)
        self.btn_quit.pack(pady=5)

    def process_frame(self, frame):
        face_result = self.face_model(frame, imgsz=320, verbose=False)
        processed_feed = face_result[0].plot()

        for box in face_result[0].boxes.xyxy:
            left, top, right, bottom = box.int()

            h, w, _ = frame.shape
            top, bottom = max(0, top), min(h, bottom)
            left, right = max(0, left), min(w, right)

            if top >= bottom or left >= right:
                continue

            face = frame[top:bottom, left:right]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (200, 200))

            pred_label, distance = self.face_recognizer.predict(face_resized)

            if distance > self.max_distance:
                name = "Unknown"
            else:
                name = self.names.get(pred_label, "Unknown")

            cv2.putText(
                processed_feed,
                f"{name} | {int(distance)}",
                (int(left), int(bottom) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        return processed_feed

    def process_look_rating_frame(self, frame):
        face_result = self.face_model.track(frame, imgsz=320, persist=True, verbose=False)
        processed_feed = face_result[0].plot()

        boxes = face_result[0].boxes.xyxy
        track_ids = face_result[0].boxes.id

        num_faces = len(boxes)

        if num_faces >= 2:
            for i, box in enumerate(boxes):
                left, top, right, bottom = box.int()

                h, w, _ = frame.shape
                top, bottom = max(0, top), min(h, bottom)
                left, right = max(0, left), min(w, right)

                if top >= bottom or left >= right:
                    continue

                track_id = int(track_ids[i]) if track_ids is not None else i

                if track_id not in self.face_tiers:
                    face = frame[top:bottom, left:right]
                    face_bytes = face.tobytes()
                    tier_index = sum(face_bytes[::150]) % len(self.tiers)
                    self.face_tiers[track_id] = self.tiers[tier_index]

                tier_rating = self.face_tiers[track_id]

                cv2.putText(
                    processed_feed,
                    f"Tier: {tier_rating}",
                    (int(left), int(bottom) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
        else:
            cv2.putText(
                processed_feed,
                f"Waiting for 2+ faces... (Current: {num_faces})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        return processed_feed

    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Press 'x' in the video window to stop.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_feed = self.process_frame(frame)
            cv2.imshow("Live Camera Feed", processed_feed)

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def start_look_rating(self):
        self.face_tiers.clear()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Press 'x' in the video window to stop look rating.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_feed = self.process_look_rating_frame(frame)
            cv2.imshow("Look Rating Feed (2+ Faces)", processed_feed)

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            return

        frame = cv2.imread(file_path)
        if frame is None:
            print("Error: Could not load image.")
            return

        processed_feed = self.process_frame(frame)
        cv2.imshow("Image Detection Result", processed_feed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
