import streamlit as st
import cv2
import numpy as np
import time
import math
import random
import os

# =====================
# Streamlit 設定
# =====================
st.set_page_config(layout="centered")
st.title("配信フィルタのまね")

# =====================
# Haar Cascade
# =====================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# =====================
# UI
# =====================
st.sidebar.header("設定")
brightness = st.sidebar.slider("明るさ", -50, 50, 10)
contrast = st.sidebar.slider("コントラスト", 50, 150, 110)
spawn_n = st.sidebar.slider("生成粒子数 / frame", 0, 5, 2)
particle_size = st.sidebar.slider("粒子サイズ", 1, 8, 3)
motion_mode = st.sidebar.selectbox(
    "運動モード", ["放射", "上昇", "左右拡散", "上下拡散"]
)
diffusion = st.sidebar.slider("拡散強度", 1, 10, 3)
base_speed = st.sidebar.slider("基準速度", 1, 10, 2)
color_mode = st.sidebar.selectbox("粒子色", ["黄色", "白", "ランダム"])
ribbon_scale = st.sidebar.slider("リボンサイズ", 0.5, 2.0, 1.0)

start = st.sidebar.button("▶ 開始")
stop = st.sidebar.button("■ 終了")

# =====================
# session_state
# =====================
if "run" not in st.session_state:
    st.session_state.run = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "particles" not in st.session_state:
    st.session_state.particles = []

if start and not st.session_state.run:
    st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    st.session_state.run = True

if stop and st.session_state.run:
    st.session_state.run = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.session_state.particles = []

frame_area = st.image([])

# =====================
# リボン画像読み込み
# =====================
ribbon_path = "ribbon.png"
if not os.path.exists(ribbon_path):
    st.error("ribbon.png がフォルダにありません")
else:
    ribbon_img = cv2.imread(ribbon_path, cv2.IMREAD_UNCHANGED)  # BGRA

def overlay_ribbon(frame, cx, cy, face_w, face_h, scale=1.0):
    """
    frame: OpenCV画像 (BGR)
    cx, cy: リボン中心位置
    face_w, face_h: 顔矩形の幅・高さ
    scale: 顔に対するリボン倍率（1.0 = 顔幅の25%くらい）
    """
    h, w = ribbon_img.shape[:2]

    # 顔サイズに合わせて縮小
    target_w = int(face_w * 0.2 * scale)   # 顔幅の20%を基準
    target_h = int(h * target_w / w)       # アスペクト比維持
    resized = cv2.resize(ribbon_img, (target_w, target_h))

    # 位置をフレーム内に収める
    x1 = max(int(cx - target_w // 2), 0)
    y1 = max(int(cy - target_h // 2), 0)
    x2 = min(x1 + target_w, frame.shape[1])
    y2 = min(y1 + target_h, frame.shape[0])

    overlay = resized[0:(y2-y1), 0:(x2-x1)]
    alpha = overlay[:, :, 3] / 255.0
    alpha = alpha[..., np.newaxis]

    frame[y1:y2, x1:x2] = (alpha * overlay[:, :, :3] +
                            (1 - alpha) * frame[y1:y2, x1:x2])
# =====================
# 粒子モデル
# =====================
def random_color(mode):
    if mode == "ランダム":
        return tuple(random.randint(150, 255) for _ in range(3))
    elif mode == "白":
        return (255, 255, 255)
    else:
        return (255, 255, 0)

def create_particle(cx, cy):
    if motion_mode == "放射":
        angle = random.uniform(0, 2*math.pi)
        vx = math.cos(angle)*base_speed
        vy = math.sin(angle)*base_speed
    elif motion_mode == "上昇":
        vx = random.uniform(-0.2,0.2)*base_speed
        vy = -random.uniform(0.8,1.2)*base_speed
    elif motion_mode == "左右拡散":
        vx = random.uniform(-1,1)*base_speed
        vy = 0
    elif motion_mode == "上下拡散":
        vx = 0
        vy = random.uniform(-1,1)*base_speed
    return {"x": float(cx), "y": float(cy), "vx": vx, "vy": vy, "color": random_color(color_mode)}

def update_particle(p):
    p["vx"] += random.uniform(-1,1)*diffusion*0.05
    p["vy"] += random.uniform(-1,1)*diffusion*0.05
    p["x"] += p["vx"]
    p["y"] += p["vy"]

# =====================
# メインループ
# =====================
if st.session_state.run and st.session_state.cap and os.path.exists(ribbon_path):
    while st.session_state.run:
        ret, frame = st.session_state.cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        frame = cv2.convertScaleAbs(frame, alpha=contrast/100.0, beta=brightness)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # 粒子生成とリボン描画
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 8)
            for (ex,ey,ew,eh) in eyes[:2]:
                cx = x+ex+ew//2
                cy = y+ey+eh//2
                for _ in range(spawn_n):
                    st.session_state.particles.append(create_particle(cx, cy))
                # 顔矩形に対して目の下あたりに調整
                left_cheek = (x + int(w*0.25), y + int(h*0.55))
                right_cheek = (x + int(w*0.75), y + int(h*0.55))

                overlay_ribbon(frame, *left_cheek, face_w=w, face_h=h, scale=ribbon_scale)
                overlay_ribbon(frame, *right_cheek, face_w=w, face_h=h, scale=ribbon_scale)

        # 粒子描画
        for p in st.session_state.particles:
            update_particle(p)
            cv2.circle(frame, (int(p["x"]), int(p["y"])), particle_size, p["color"], -1)

        frame_area.image(frame, channels="BGR")
        time.sleep(0.03)
