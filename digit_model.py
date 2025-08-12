import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pickle
import cv2

# -------------------------
# Load trained model
# -------------------------
with open("digit_model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------
# Page Config & Custom CSS
# -------------------------
st.set_page_config(page_title="Digit Recognition", layout="centered")

st.markdown(
    """
    <style>
        /* Main page background - dark gradient */
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            font-family: 'Segoe UI', sans-serif;
            color: white;
        }
        /* Title style */
        h1 {
            text-align: center;
            color: #00e5ff;
            text-shadow: 0px 0px 8px rgba(0,229,255,0.8);
        }
        /* Center the content */
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
        }
        /* Canvas style - remove black extra box */
        canvas {
            background-color: white !important;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,229,255,0.6);
        }
        /* Prediction box */
        .result-box {
            background-color: rgba(0, 0, 0, 0.6);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(0, 229, 255, 0.6);
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #00e5ff;
            margin-top: 15px;
        }
        /* Buttons */
        div.stButton > button {
            background-color: #00e5ff;
            color: black;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 0px 0px 10px rgba(0,229,255,0.6);
        }
        div.stButton > button:hover {
            background-color: #00bcd4;
            color: white;
            transform: scale(1.05);
            transition: 0.2s;
            box-shadow: 0px 0px 15px rgba(0,229,255,1);
        }
        /* Footer text */
        .footer {
            text-align: center;
            font-size: 14px;
            color: #aaa;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# App Title & Info
# -------------------------
st.title("✏️ Handwritten Digit Recognition")
st.write("<p style='text-align:center;'>Draw a digit (0–9) below and I'll guess it! 🎯</p>", unsafe_allow_html=True)

# -------------------------
# Centered container for canvas + button
# -------------------------
with st.container():
    st.markdown('<div class="center-container">', unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  # White background for drawing
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    # -------------------------
    # Preprocessing function
    # -------------------------
    def preprocess(img):
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)
        img = 255 - img
        coords = cv2.findNonZero(img)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        digit = img[y:y+h, x:x+w]
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        square[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = digit
        square = cv2.resize(square, (8, 8), interpolation=cv2.INTER_AREA)
        square = cv2.GaussianBlur(square, (3, 3), 0)
        square = (square / 255.0) * 16
        square = square.reshape(1, -1)
        return square

    # -------------------------
    # Prediction
    # -------------------------
    if st.button("🔍 Predict"):
        if canvas_result.image_data is not None:
            processed_img = preprocess(canvas_result.image_data)

            if processed_img is None:
                st.warning("✏️ Please draw a digit first!")
            else:
                prediction = model.predict(processed_img)
                st.markdown(f"<div class='result-box'>Predicted Digit: {prediction[0]} ✅</div>", unsafe_allow_html=True)
                st.image((processed_img.reshape(8, 8) / 16.0), caption="Processed 8x8 Image", width=150)
        else:
            st.warning("✏️ Please draw a digit first!")

    st.markdown('</div>', unsafe_allow_html=True)  # close center container

# -------------------------
# Footer
# -------------------------
st.markdown("<div class='footer'>📌 Built with <b>Streamlit</b>, <b>OpenCV</b>, and <b>scikit-learn</b></div>", unsafe_allow_html=True)
