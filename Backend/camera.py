# camera.py
import cv2 as cv

class Camera:
    def __init__(self, camera_index=0):
        self.capture = cv.VideoCapture(camera_index)

    def get_frame(self):
        ret, frame = self.capture.read()

        if not ret:
            return None

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        return frame

    def release(self):
        self.capture.release()