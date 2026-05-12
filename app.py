from pathlib import Path
import tempfile
import time

import cv2
import streamlit as st

from dashboard_utils import DEFAULT_MODEL_PATH, get_youtube_stream, load_model


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Helmet Detection Studio",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    *{
        font-family:'Poppins',sans-serif !important;
    }

    .stApp{
        background:
        radial-gradient(circle at top left, rgba(124,58,237,0.25), transparent 30%),
        radial-gradient(circle at bottom right, rgba(6,182,212,0.18), transparent 30%),
        linear-gradient(135deg,#0f172a 0%,#111827 100%);
        color:white;
    }

    section[data-testid="stSidebar"]{
        background:rgba(15,23,42,0.85);
        border-right:1px solid rgba(255,255,255,0.08);
        backdrop-filter:blur(14px);
    }

    .hero{
        position:relative;
        overflow:hidden;
        padding:3.5rem;
        border-radius:32px;
        background:linear-gradient(
            135deg,
            rgba(17,24,39,0.88),
            rgba(30,41,59,0.72)
        );
        border:1px solid rgba(255,255,255,0.08);
        margin-bottom:2rem;
        box-shadow:0 20px 60px rgba(0,0,0,0.45);
    }

    .hero::before{
        content:'';
        position:absolute;
        width:400px;
        height:400px;
        background:rgba(124,58,237,0.22);
        border-radius:50%;
        top:-180px;
        right:-100px;
        filter:blur(90px);
    }

    .hero::after{
        content:'';
        position:absolute;
        width:350px;
        height:350px;
        background:rgba(6,182,212,0.18);
        border-radius:50%;
        bottom:-160px;
        left:-100px;
        filter:blur(90px);
    }

    .hero-badge{
        display:inline-block;
        padding:0.55rem 1rem;
        border-radius:999px;
        background:linear-gradient(135deg,#7c3aed,#06b6d4);
        font-size:0.78rem;
        font-weight:700;
        letter-spacing:1px;
        color:white;
        margin-bottom:1.2rem;
        position:relative;
        z-index:2;
    }

    .hero-title{
        font-size:4rem;
        font-weight:800;
        line-height:1.05;
        margin-bottom:1rem;
        color:white;
        position:relative;
        z-index:2;
    }

    .hero-sub{
        color:#cbd5e1;
        font-size:1.1rem;
        max-width:760px;
        line-height:1.8;
        position:relative;
        z-index:2;
    }

    .glass-card{
        background:rgba(30,41,59,0.65);
        border:1px solid rgba(255,255,255,0.08);
        border-radius:24px;
        padding:1.5rem;
        backdrop-filter:blur(12px);
        box-shadow:0 10px 35px rgba(0,0,0,0.28);
    }

    .section-title{
        font-size:1.6rem;
        font-weight:700;
        color:white;
        margin-bottom:1rem;
    }

    .stTabs [data-baseweb="tab-list"]{
        gap:0.7rem;
    }

    .stTabs [data-baseweb="tab"]{
        background:rgba(30,41,59,0.65);
        border-radius:14px;
        padding:0.9rem 1.2rem;
        color:white;
        font-weight:600;
        border:1px solid rgba(255,255,255,0.06);
    }

    .stTabs [aria-selected="true"]{
        background:linear-gradient(135deg,#7c3aed,#06b6d4) !important;
        color:white !important;
    }

    .upload-box{
        border:2px dashed rgba(255,255,255,0.15);
        border-radius:22px;
        padding:2rem;
        text-align:center;
        background:rgba(15,23,42,0.55);
    }

    .stButton > button{
        width:100%;
        border:none;
        border-radius:14px;
        padding:0.9rem 1rem;
        font-weight:700;
        color:white;
        background:linear-gradient(135deg,#7c3aed,#06b6d4);
        transition:0.25s ease;
    }

    .stButton > button:hover{
        transform:translateY(-2px);
        box-shadow:0 10px 30px rgba(124,58,237,0.35);
    }

    .stMetric{
        background:rgba(30,41,59,0.65);
        border:1px solid rgba(255,255,255,0.08);
        padding:1rem;
        border-radius:18px;
    }

    .stTextInput input{
        background:rgba(15,23,42,0.8) !important;
        border:1px solid rgba(255,255,255,0.08) !important;
        border-radius:14px !important;
        color:white !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def _load_model_cached():
    return load_model(DEFAULT_MODEL_PATH)

# =========================================================
# HERO
# =========================================================

def hero():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-badge">
                ⚡ AI SAFETY MONITORING
            </div>

            <div class="hero-title">
                🪖 Helmet Detection Studio
            </div>

            <div class="hero-sub">
                Advanced real-time helmet detection dashboard powered by AI computer vision.
                Upload images, videos, webcam captures or monitor YouTube live streams instantly.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# IMAGE RESULT POPUP
# =========================================================

@st.dialog("🔍 Detection Result", width="large")
def show_detection_popup(original, detected, detections):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📷 Original")

        st.image(
            original
        )

    with col2:

        st.markdown("### ✅ Detection Output")

        st.image(
            detected
        )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Detections",
        detections
    )

    c2.metric(
        "Status",
        "Processed"
    )

    c3.metric(
        "AI Model",
        "YOLO"
    )

# =========================================================
# IMAGE DETECTION
# =========================================================

def _run_image_detection(model, image_source, conf):

    with st.spinner("Running AI Detection..."):

        result = model.predict(
            source=image_source,
            conf=conf,
            verbose=False,
            imgsz=960
        )[0]

    annotated = result.plot()[:, :, ::-1]

    show_detection_popup(
        image_source,
        annotated,
        len(result.boxes)
    )

# =========================================================
# VIDEO DETECTION
# =========================================================

def _run_video_detection(model, source, conf):

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():

        st.error("Could not open video source")

        return

    frame_slot = st.empty()

    stats_slot = st.empty()

    frame_count = 0

    total_detections = 0

    start = time.time()

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_count += 1

        result = model.predict(
            source=frame,
            conf=conf,
            verbose=False,
            imgsz=640
        )[0]

        total_detections += len(result.boxes)

        annotated = result.plot()[:, :, ::-1]

        frame_slot.image(
            annotated,
            channels="RGB",
            use_container_width=True
        )

        elapsed = max(time.time() - start, 1e-6)

        fps = frame_count / elapsed

        c1, c2, c3 = stats_slot.columns(3)

        c1.metric(
            "FPS",
            f"{fps:.1f}"
        )

        c2.metric(
            "Frames",
            frame_count
        )

        c3.metric(
            "Detections",
            total_detections
        )

    cap.release()

# =========================================================
# MAIN APP
# =========================================================

def main():

    hero()

    with st.sidebar:

        st.markdown("## ⚙️ Detection Controls")

        conf = st.slider(
            "Confidence Threshold",
            0.01,
            0.95,
            0.20,
            0.01
        )

        st.markdown("---")

        st.success("✅ AI Model Loaded")

        st.caption("Model: bestone.pt")

    try:

        model = _load_model_cached()

    except Exception as e:

        st.error(f"Model failed to load: {e}")

        st.info(
            "Make sure bestone.pt exists correctly in your project."
        )

        return

    tabs = st.tabs([
        "🖼 Upload Image",
        "📷 Webcam",
        "📹 Upload Video",
        "🔴 YouTube Live"
    ])

    # =====================================================
    # IMAGE TAB
    # =====================================================

    with tabs[0]:

        st.markdown(
            '<div class="section-title">Upload Image Detection</div>',
            unsafe_allow_html=True
        )

        uploaded_image = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed"
        )

        if uploaded_image:

            st.markdown(
                '<div class="glass-card">',
                unsafe_allow_html=True
            )

            st.image(
                uploaded_image
            )

            if st.button("🔍 Run Detection"):

                _run_image_detection(
                    model,
                    uploaded_image,
                    conf
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

    # =====================================================
    # WEBCAM TAB
    # =====================================================

    with tabs[1]:

        st.markdown(
            '<div class="section-title">Webcam Detection</div>',
            unsafe_allow_html=True
        )

        captured = st.camera_input(
            "Take a Picture"
        )

        if captured is not None:

            _run_image_detection(
                model,
                captured.getvalue(),
                conf
            )

    # =====================================================
    # VIDEO TAB
    # =====================================================

    with tabs[2]:

        st.markdown(
            '<div class="section-title">Upload Video Detection</div>',
            unsafe_allow_html=True
        )

        uploaded_video = st.file_uploader(
            "Upload Video",
            type=["mp4", "mov", "avi", "mkv"],
            label_visibility="collapsed"
        )

        if uploaded_video:

            temp_file = tempfile.NamedTemporaryFile(delete=False)

            temp_file.write(uploaded_video.read())

            st.video(temp_file.name)

            if st.button("▶️ Run Video Detection"):

                _run_video_detection(
                    model,
                    temp_file.name,
                    conf
                )

    # =====================================================
    # YOUTUBE TAB
    # =====================================================

    with tabs[3]:

        st.markdown(
            '<div class="section-title">YouTube Live Detection</div>',
            unsafe_allow_html=True
        )

        youtube_url = st.text_input(
            "Paste YouTube Live URL"
        )

        if st.button("🚀 Start Live Detection"):

            stream = get_youtube_stream(
                youtube_url
            )

            if stream:

                _run_video_detection(
                    model,
                    stream,
                    conf
                )

            else:

                st.error(
                    "Could not extract YouTube stream."
                )

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()
