import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pickle
import cv2
import time

# -------------------------
# Load trained model
# -------------------------
@st.cache_resource
def load_model():
    try:
        with open("digit_model.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("⚠️ Model file 'digit_model.pkl' not found. Please ensure the model is in the same directory.")
        st.stop()

model = load_model()

# -------------------------
# Page Config & Custom CSS
# -------------------------
st.set_page_config(
    page_title="AI Digit Recognition", 
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        .stApp {
            background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #0f4c75);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            font-family: 'Poppins', sans-serif;
            color: white;
        }
        @keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
        .main-title {
            text-align: center;
            background: linear-gradient(45deg, #00e5ff, #64b5f6, #42a5f5, #2196f3);
            background-size: 300% 300%;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            animation: titleGlow 3s ease-in-out infinite alternate;
            font-size: 3rem; font-weight: 700; margin-bottom: 1rem; text-shadow: 0 0 30px rgba(0,229,255,0.5);
        }
        @keyframes titleGlow { 0% { filter: brightness(1) drop-shadow(0 0 5px rgba(0,229,255,0.3)); } 100% { filter: brightness(1.2) drop-shadow(0 0 20px rgba(0,229,255,0.8)); } }
        .subtitle { text-align: center; font-size: 1.2rem; color: rgba(255,255,255,0.8); margin-bottom: 2rem; font-weight: 300; }
        .canvas-container {
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);
            border-radius: 20px; padding: 30px; margin: 20px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.3); transition: all 0.3s ease;
        }
        .canvas-container:hover { transform: translateY(-5px); box-shadow: 0 12px 40px rgba(0,229,255,0.2); }
        canvas {
            background-color: white !important; border-radius: 15px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
            border: 2px solid rgba(0,229,255,0.3) !important; transition: all 0.3s ease !important;
        }
        canvas:hover { border: 2px solid rgba(0,229,255,0.8) !important; box-shadow: 0 6px 25px rgba(0,229,255,0.4) !important; }
        .prediction-result {
            background: linear-gradient(135deg, rgba(0,229,255,0.2), rgba(33,150,243,0.2));
            border: 2px solid rgba(0,229,255,0.5); border-radius: 15px; padding: 25px; text-align: center; margin: 20px 0;
            backdrop-filter: blur(10px); animation: resultPulse 2s ease-in-out;
        }
        @keyframes resultPulse { 0% { transform: scale(0.95); opacity: 0; } 50% { transform: scale(1.02);} 100% { transform: scale(1); opacity: 1; } }
        .prediction-digit { font-size: 4rem; font-weight: 700; color: #00e5ff; text-shadow: 0 0 20px rgba(0,229,255,0.8); margin: 10px 0; }
        .prediction-text { font-size: 1.3rem; color: rgba(255,255,255,0.9); margin-bottom: 10px; }
        .confidence-text { font-size: 1rem; color: rgba(255,255,255,0.7); }
        .stButton > button {
            background: linear-gradient(45deg, #00e5ff, #2196f3) !important; color: white !important; border: none !important; border-radius: 12px !important;
            padding: 12px 30px !important; font-size: 1.1rem !important; font-weight: 600 !important; box-shadow: 0 4px 15px rgba(0,229,255,0.4) !important;
            transition: all 0.3s ease !important; text-transform: uppercase !important; letter-spacing: 1px !important;
        }
        .stButton > button:hover { background: linear-gradient(45deg, #2196f3, #00e5ff) !important; transform: translateY(-2px) scale(1.05) !important; box-shadow: 0 6px 25px rgba(0,229,255,0.6) !important; }
        .stButton > button:active { transform: translateY(0px) scale(0.98) !important; }
        .instructions { background: rgba(255,255,255,0.05); border-left: 4px solid #00e5ff; border-radius: 10px; padding: 20px; margin: 20px 0; backdrop-filter: blur(5px); }
        .instructions h4 { color: #00e5ff; margin-bottom: 10px; font-weight: 600; }
        .instructions ul { color: rgba(255,255,255,0.8); line-height: 1.6; }
        .stat-card {
            background: rgba(255,255,255,0.1); border-radius: 15px; padding: 20px; text-align: center; backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2); transition: all 0.3s ease;
        }
        .stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,229,255,0.2); }
        .stat-number { font-size: 2rem; font-weight: 700; color: #00e5ff; }
        .stat-label { font-size: 0.9rem; color: rgba(255,255,255,0.7); margin-top: 5px; }
        .footer { text-align: center; margin-top: 50px; padding: 20px; background: rgba(0,0,0,0.3); border-radius: 15px; backdrop-filter: blur(10px); }
        .footer-text { color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-bottom: 10px; }
        .creator-name {
            position: fixed; bottom: 20px; right: 20px; background: rgba(0,229,255,0.2); backdrop-filter: blur(10px);
            border: 1px solid rgba(0,229,255,0.3); border-radius: 25px; padding: 10px 20px; color: #00e5ff; font-weight: 600; font-size: 0.9rem; z-index: 1000;
            animation: creatorPulse 3s ease-in-out infinite;
        }
        @keyframes creatorPulse { 0%,100% { box-shadow: 0 0 5px rgba(0,229,255,0.3);} 50% { box-shadow: 0 0 20px rgba(0,229,255,0.6);} }
        .stAlert { background: rgba(255,193,7,0.1) !important; border: 1px solid rgba(255,193,7,0.3) !important; border-radius: 10px !important; backdrop-filter: blur(10px) !important; }
        .loading { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(0,229,255,0.3); border-radius: 50%; border-top-color: #00e5ff; animation: spin 1s ease-in-out infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Session State
# -------------------------
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0
if 'correct_predictions' not in st.session_state:
    st.session_state.correct_predictions = 0
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

# -------------------------
# App Header
# -------------------------
st.markdown("""
    <div class="main-title">🧠 AI Digit Recognition</div>
    <div class="subtitle">✨ Advanced Machine Learning Powered Handwriting Recognition ✨</div>
""", unsafe_allow_html=True)

# --- Reserve placeholders for the 3 stats cards (we'll fill them after button logic)
stats_col1, stats_col2, stats_col3 = st.columns(3)

# -------------------------
# Instructions Panel
# -------------------------
with st.expander("📋 How to Use", expanded=False):
    st.markdown("""
        <div class="instructions">
            <h4>🎯 Quick Start Guide</h4>
            <ul>
                <li>🖊️ Draw a single digit (0-9) in the white canvas below</li>
                <li>✏️ Use your mouse or touchscreen to draw clearly</li>
                <li>🔍 Click "Predict Digit" to see the AI's guess</li>
                <li>🧹 Use "Clear Canvas" to start over</li>
                <li>📊 Use ✅/❌ to mark whether the prediction was correct</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# -------------------------
# Main Canvas Section
# -------------------------
st.markdown('<div class="canvas-container">', unsafe_allow_html=True)

CANVAS_SIZE = 300
STROKE_WIDTH = 15

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.0)",
    stroke_width=STROKE_WIDTH,
    stroke_color="#000000",
    background_color="#FFFFFF",
    background_image=None,
    update_streamlit=True,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    drawing_mode="freedraw",
    point_display_radius=0,
    key=f"drawing_canvas_{st.session_state.canvas_key}",
    display_toolbar=False,
)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Preprocessing
# -------------------------
def preprocess_image(img_data):
    try:
        if img_data.shape[-1] == 4:
            img = cv2.cvtColor(img_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)
        else:
            img = img_data.astype("uint8")
        img = 255 - img
        coords = cv2.findNonZero(img)
        if coords is None:
            return None, "No drawing detected"
        x, y, w, h = cv2.boundingRect(coords)
        padding = 10
        x = max(0, x - padding); y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        digit = img[y:y+h, x:x+w]
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2; x_offset = (size - w) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = digit
        resized = cv2.resize(square, (8, 8), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        normalized = (blurred.astype(np.float32) / 255.0) * 16.0
        feature_vector = normalized.reshape(1, -1)
        return feature_vector, None
    except Exception as e:
        return None, f"Processing error: {str(e)}"

# -------------------------
# Action Buttons
# -------------------------
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    predict_button = st.button("🔍 Predict Digit", use_container_width=True)
with btn_col2:
    clear_button = st.button("🧹 Clear Canvas", use_container_width=True, help="Clear the drawing canvas")

if clear_button:
    st.session_state.canvas_key += 1
    st.rerun()

# -------------------------
# Prediction Logic
# -------------------------
if predict_button:
    if canvas_result.image_data is not None:
        img_array = np.array(canvas_result.image_data)
        if img_array[:, :, 3].sum() == 0:
            st.warning("✏️ Please draw a digit first!")
            st.info("💡 **Tip**: Click and drag on the white canvas above to draw your digit.")
        else:
            with st.spinner('🤖 AI is analyzing your drawing...'):
                time.sleep(0.5)
                processed_img, error = preprocess_image(canvas_result.image_data)
                if error:
                    st.warning(f"⚠️ {error}")
                    st.info("💡 **Tip**: Try drawing a clearer digit with bolder strokes!")
                else:
                    try:
                        prediction = model.predict(processed_img)[0]
                        st.session_state.prediction_count += 1
                        st.session_state.last_prediction = prediction

                        st.markdown(f"""
                            <div class="prediction-result">
                                <div class="prediction-text">🎯 AI Prediction:</div>
                                <div class="prediction-digit">{prediction}</div>
                                <div class="confidence-text">✨ Prediction #{st.session_state.prediction_count}</div>
                            </div>
                        """, unsafe_allow_html=True)

                        _, mid, _ = st.columns([1, 2, 1])
                        with mid:
                            st.image(
                                (processed_img.reshape(8, 8) / 16.0),
                                caption="🔬 Processed 8×8 Image (What AI Sees)",
                                width=200,
                                clamp=True
                            )
                    except Exception as e:
                        st.error(f"⚠️ Prediction failed: {str(e)}")
                        st.info("🔧 Please check if your model is compatible with the input format.")
    else:
        st.warning("✏️ Please draw a digit first!")
        st.info("💡 **Tip**: Click and drag on the white canvas above to draw your digit.")

# -------------------------
# Feedback Section (ALWAYS RENDERED; gated by last_prediction)
# -------------------------
st.markdown("---")
feedback_col1, feedback_col2 = st.columns(2)

no_prediction_yet = st.session_state.last_prediction is None

with feedback_col1:
    if st.button("✅ Correct!", use_container_width=True, key="correct_btn", disabled=no_prediction_yet):
        # increment and immediately rerun so stats update at top placeholders
        st.session_state.correct_predictions += 1
        st.success("🎉 Thanks! Marked as correct.")
        st.balloons()
        st.rerun()

with feedback_col2:
    if st.button("❌ Wrong", use_container_width=True, key="wrong_btn", disabled=no_prediction_yet):
        st.info("🤔 No worries! Try drawing more clearly or redraw.")
        st.rerun()

# -------------------------
# NOW render the Stats (after any button state changes)
# -------------------------
with stats_col1:
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{st.session_state.prediction_count}</div>
            <div class="stat-label">Predictions Made</div>
        </div>
    """, unsafe_allow_html=True)

with stats_col2:
    accuracy = (st.session_state.correct_predictions / max(1, st.session_state.prediction_count)) * 100
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{accuracy:.1f}%</div>
            <div class="stat-label">Your Accuracy</div>
        </div>
    """, unsafe_allow_html=True)

with stats_col3:
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{st.session_state.last_prediction or "–"}</div>
            <div class="stat-label">Last Prediction</div>
        </div>
    """, unsafe_allow_html=True)

# -------------------------
# Tips and Footer
# -------------------------
with st.expander("💡 Tips for Better Recognition", expanded=False):
    st.markdown("""
        <div class="instructions">
            <h4>🎯 Drawing Tips</h4>
            <ul>
                <li>🖊️ <strong>Draw boldly</strong>: Use thick, clear strokes</li>
                <li>📏 <strong>Size matters</strong>: Fill most of the canvas</li>
                <li>🎨 <strong>Stay centered</strong>: Keep your digit in the middle</li>
                <li>✨ <strong>Be clear</strong>: Avoid overlapping lines</li>
                <li>🔄 <strong>Try again</strong>: If wrong, clear and redraw!</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
    <div class="footer">
        <div class="footer-text">🚀 Powered by <strong>Streamlit</strong> • 🖼️ <strong>OpenCV</strong> • 🤖 <strong>scikit-learn</strong></div>
        <div class="footer-text">Built with ❤️ for the AI community</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="creator-name">💫 Rhythm forever ❤️</div>
""", unsafe_allow_html=True)
