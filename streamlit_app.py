import streamlit as st
import time
from Backend.camera import Camera

st.title("Camera Feed")

run = st.checkbox("Start Camera")

frame_placeholder = st.empty()

camera = Camera(0)

while run:
    frame = camera.get_frame()

    if frame is None:
        st.error("Failed to access camera.")
        break

    frame_placeholder.image(frame, channels="RGB")

    time.sleep(0.03)

camera.release()