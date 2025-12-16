import cv2
import numpy as np
import math
import random

def detect_and_effects(frame, face_cascade, eye_cascade, params, ribbon_img):
    frame = cv2.flip(frame, 1)
    frame = cv2.convertScaleAbs(
        frame,
        alpha=params["contrast"] / 100.0,
        beta=params["brightness"]
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # 両頬リボン
        left = (x + int(w*0.25), y + int(h*0.55))
        right = (x + int(w*0.75), y + int(h*0.55))
        overlay_ribbon(frame, left, w, ribbon_img, params["ribbon_scale"])
        overlay_ribbon(frame, right, w, ribbon_img, params["ribbon_scale"])

    return frame


def overlay_ribbon(frame, pos, face_w, ribbon_img, scale):
    cx, cy = pos
    h, w = ribbon_img.shape[:2]

    tw = int(face_w * 0.2 * scale)
    th = int(h * tw / w)
    img = cv2.resize(ribbon_img, (tw, th))

    x1 = max(cx - tw//2, 0)
    y1 = max(cy - th//2, 0)
    x2 = min(x1 + tw, frame.shape[1])
    y2 = min(y1 + th, frame.shape[0])

    overlay = img[0:y2-y1, 0:x2-x1]
    alpha = overlay[:, :, 3:4] / 255.0

    frame[y1:y2, x1:x2] = (
        alpha * overlay[:, :, :3] +
        (1 - alpha) * frame[y1:y2, x1:x2]
    )
