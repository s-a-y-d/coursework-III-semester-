import numpy as np
import cv2

class SyntheticTests:
    def __init__(self, p: int):
        self.p = p
    def create_empty(self):
        return np.zeros((self.p, self.p), dtype=np.float32)
    def generate_stripe(self, angle_type="diagonal"):
        img = self.create_empty()
        if angle_type == "diagonal":
            cv2.line(img, (0, 0), (self.p-1, self.p-1), 1.0, 1)
        elif angle_type == "horizontal":
            cv2.line(img, (0, self.p//2), (self.p-1, self.p-1//2), 1.0, 1)
        return img
    def generate_cross(self):
        img = self.create_empty()
        mid = self.p // 2
        cv2.line(img, (0, mid), (self.p-1, mid), 1.0, 1) # Горизонталь
        cv2.line(img, (mid, 0), (mid, self.p-1), 1.0, 1) # Вертикаль
        return img
    def generate_window(self):
        img = self.create_empty()
        margin = self.p // 4
        cv2.rectangle(img, (margin, margin), (self.p - margin, self.p - margin), 1.0, 1)
        return img
    def generate_v_shape(self):
        img = self.create_empty()
        pts = np.array([[self.p//4, self.p//4], [self.p//2, self.p*3//4], [self.p*3//4, self.p//4]])
        cv2.polylines(img, [pts], False, 1.0, 1)
        return img
