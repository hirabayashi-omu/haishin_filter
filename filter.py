import streamlit as st
import cv2
import os
from effects import detect_and_effects

# 判定
IS_CLOUD = os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true"

st.title("配信フィルタ（分離構成）")

# パラメータ
params = {
    "brightness": st.sidebar.slider("明るさ", -50, 50, 10),
    "contrast": st.sidebar.slider("コントラスト", 50, 150, 110),
    "ribbon_scale": st.sidebar.slider("リボンサイズ", 0.5, 2.0, 1.0)
}

# Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

ribbon_img = cv2.imread("ribbon.png", cv2.IMREAD_UNCHANGED)

frame_area = st.empty()

if IS_CLOUD:
    st.info("クラウド版：画像アップロードモード")
    img = st.file_uploader("画像を選択", type=["jpg", "png"])

    if img:
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame = detect_and_effects(frame, face_cascade, eye_cascade, params, ribbon_img)
        frame_area.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    st.success("ローカル版：Webカメラ使用")
    from camera import get_camera
    cap = get_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_and_effects(frame, face_cascade, eye_cascade, params, ribbon_img)
        frame_area.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
