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

    * {
        font-family: 'Poppins', sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(124,58,237,0.25), transparent 30%),
            radial-gradient(circle at bottom right, rgba(6,182,212,0.2), transparent 30%),
            linear-gradient(135deg, #0f172a 0%, #111827 100%);
        color: white;
    }

    .hero {
        padding: 3rem;
        border-radius: 28px;
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.45);
    }

    .hero-badge {
        display: inline-block;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        background: linear-gradient(135deg,#7c3aed,#06b6d4);
        color: white;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: 1px;
    }

    .hero-title {
        font-size: 3.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 1rem;
        background: linear-gradient(to right, white, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-sub {
        color: #cbd5e1;
        font-size: 1.1rem;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: white;
    }

    .media-card {
        background: rgba(30,41,59,0.75);
        border-radius: 22px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        transition: 0.3s ease;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    .media-card:hover {
        transform: translateY(-6px) scale(1.02);
        border: 1px solid rgba(124,58,237,0.7);
        box-shadow: 0 20px 40px rgba(124,58,237,0.25);
    }

    .stButton > button {
        background: linear-gradient(135deg,#7c3aed,#06b6d4);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 700;
        padding: 0.8rem 1rem;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(124,58,237,0.35);
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(30,41,59,0.6);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin-right: 0.5rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg,#7c3aed,#06b6d4) !important;
        color: white !important;
    }

    .stMetric {
        background: rgba(30,41,59,0.8);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1rem;
        border-radius: 16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# CONSTANTS
# =========================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

# =========================================================
# HELPERS
# =========================================================

def _list_files(folder: str, extensions: set[str]):

    root = Path(folder)

    if not root.exists():
        return []

    return sorted([
        str(p)
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])


def _read_youtube_links():

    return [
        "https://www.youtube.com/watch?v=UemFRPrl1hk",
        "https://www.youtube.com/watch?v=OBJ5Q0lWbqk",
    ]


@st.cache_resource
def _load_model_cached():

    return load_model(DEFAULT_MODEL_PATH)

# =========================================================
# HERO SECTION
# =========================================================

def hero():

    st.markdown(
        """
        <div style="text-align: center; padding: 2.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 2rem;">
            <h1 style="font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: center; gap: 15px;">
                <span style="font-size: 3rem;">👷‍♂️</span> Helmet Check AI
            </h1>
            <p style="font-size: 1.2rem; color: #a1a1aa; max-width: 600px; margin: 0 auto;">
                Advanced real-time computer vision dashboard for automated safety compliance monitoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DETECTION DISPLAY
# =========================================================

def show_detection_popup(original, detected, detections):

    st.markdown("## 🔍 Detection Result")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📷 Original Image")

        st.image(
            original,
            use_container_width=True
        )

    with col2:

        st.markdown("### ✅ Detection Output")

        st.image(
            detected,
            use_container_width=True
        )

    st.markdown("---")

    st.metric(
        "Total Detections",
        detections
    )

# =========================================================
# IMAGE DETECTION
# =========================================================

# =========================================================
# IMAGE DETECTION
# =========================================================

def _run_image_detection(model, image_source, conf):

    with st.spinner("Running Detection..."):

        # Convert uploaded bytes to OpenCV image
        if isinstance(image_source, bytes):

            import numpy as np

            file_bytes = np.asarray(
                bytearray(image_source),
                dtype=np.uint8
            )

            image = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )

        else:
            image = image_source

        result = model.predict(
            source=image,
            conf=conf,
            verbose=False,
            imgsz=960
        )[0]

    annotated = result.plot()[:, :, ::-1]

    show_detection_popup(
        image,
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

        annotated = result.plot()[:, :, ::-1]

        frame_slot.image(
            annotated,
            channels="RGB",
            use_container_width=True
        )

        elapsed = max(time.time() - start, 1e-6)

        fps = frame_count / elapsed

        helmet_count = 0
        no_helmet_count = 0

        for box in result.boxes:

            cls = int(box.cls[0])

            if cls == 0:
                helmet_count += 1
            else:
                no_helmet_count += 1

        c1, c2, c3 = stats_slot.columns(3)

        c1.metric("FPS", f"{fps:.1f}")
        c2.metric("Helmet", helmet_count)
        c3.metric("No Helmet", no_helmet_count)

    cap.release()

# =========================================================
# MAIN APP
# =========================================================

def main():

    hero()

    with st.sidebar:

        st.markdown("## ⚙️ Controls")

        conf = st.slider(
            "Confidence Threshold",
            0.01,
            0.95,
            0.20,
            0.01
        )

        st.caption("Model: bestone.pt")

    model = _load_model_cached()

    image_files = _list_files("images", IMAGE_EXTS)

    video_files = _list_files("videos", VIDEO_EXTS)

    tabs = st.tabs([
        "📷 Webcam",
        "🖼️ Upload Image",
        "📹 Upload Video",
        "🔴 YouTube Live"
    ])


    # =====================================================
    # WEBCAM TAB
    # =====================================================

    with tabs[0]:

        st.markdown(
            '<div class="section-title">Capture From Webcam</div>',
            unsafe_allow_html=True
        )

        captured = st.camera_input("")

        if captured is not None:

            _run_image_detection(
                model,
                captured.getvalue(),
                conf
            )

    # =====================================================
    # UPLOAD IMAGE TAB
    # =====================================================

    with tabs[1]:

        st.markdown(
            '<div class="section-title">Upload Image</div>',
            unsafe_allow_html=True
        )

        uploaded_image = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False
        )

        if uploaded_image:

            st.image(uploaded_image, use_container_width=True)

            _run_image_detection(
                model,
                uploaded_image.getvalue(),
                conf
            )

    # =====================================================
    # UPLOAD VIDEO TAB
    # =====================================================

    with tabs[2]:

        st.markdown(
            '<div class="section-title">Upload Video</div>',
            unsafe_allow_html=True
        )

        uploaded_video = st.file_uploader(
            "Upload Video",
            type=["mp4", "mov", "avi", "mkv"]
        )

        if uploaded_video:

            temp_file = tempfile.NamedTemporaryFile(delete=False)

            temp_file.write(uploaded_video.read())

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

        presets = _read_youtube_links()

        choice = st.radio(
            "Choose Stream",
            [
                "Preset 1",
                "Preset 2",
                "Custom"
            ],
            horizontal=True
        )

        custom_url = ""

        if choice == "Custom":

            custom_url = st.text_input(
                "Paste YouTube URL"
            )

        if choice == "Preset 1":

            url = presets[0]

        elif choice == "Preset 2":

            url = presets[1]

        else:

            url = custom_url

        if st.button("🚀 Start Detection"):

            stream = get_youtube_stream(url)

            if stream:

                _run_video_detection(
                    model,
                    stream,
                    conf
                )

            else:

                st.error("Could not extract YouTube stream")

# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    main()
