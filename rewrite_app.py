import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

new_header = '''import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import fftpack
import time
from chapter2_experiments import (
    image_statistics, 
    pixel_relationships, 
    connected_components, 
    distance_measures, 
    sampling_quantization_extended
)

# --- Configuration ---
st.set_page_config(
    page_title="PronOS DIP Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #F7F7F7;
        --bg-card: #FFFFFF;
        --text-main: #000000;
        --text-muted: #9D9D9D;
        --accent-green: #367D5D;
        --accent-red: #C9442E;
        --border-color: #ECECEC;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-main) !important;
        background-color: var(--bg-main) !important;
    }

    .stApp {
        background-color: var(--bg-main) !important;
    }

    /* Hero Card */
    .hero-card {
        background: var(--bg-card);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        border: 1px solid var(--border-color);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }

    .hero-card h1 {
        color: var(--text-main) !important;
        font-weight: 700 !important;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .hero-card p {
        color: var(--text-muted);
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }

    .status-chip {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        background-color: var(--accent-green);
        color: #FFFFFF;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-card) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }

    .sidebar-card {
        background: var(--bg-main);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
    }
    .sidebar-card h4 {
        margin: 0 0 0.5rem 0;
        color: var(--accent-green);
    }
    .sidebar-card p {
        margin: 0;
        font-size: 0.85rem;
        color: var(--text-muted);
    }

    /* Buttons */
    .stButton > button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: 0.2s ease-in-out;
        padding: 0.6rem 2rem !important;
    }
    .stButton > button:hover {
        background-color: var(--accent-green) !important;
        transform: translateY(-2px);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px !important;
        padding: 8px 16px !important;
        border: 1px solid var(--border-color) !important;
        background-color: var(--bg-main) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-green) !important;
        color: #FFFFFF !important;
        border-color: var(--accent-green) !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--text-main) !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--accent-green) !important;
        font-weight: 600 !important;
    }

    img {
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    /* Inputs */
    .stTextInput > div > div > input, .stSelectbox > div > div > div, .stSlider > div > div > div > div {
        border-radius: 12px !important;
        border: 1px solid var(--border-color) !important;
        background-color: var(--bg-main) !important;
    }

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-card">
        <h1>PronOS Learning Arena</h1>
        <p>A clean, focused educational environment for mastering digital image processing.</p>
        <span class="status-chip">EDU MODE • INTERACTIVE • VISUAL-FIRST</span>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Learning Tracks", "2", "Fundamentals + Advanced")
metric_col2.metric("Interactive Modules", "12", "Step-by-step labs")
metric_col3.metric("Experience Goal", "Practice", "Concept → Experiment")

'''

# 1. Replace from top of file to `# --- Utils ---`
content = re.sub(r'(?s)^.*?# --- Utils ---', new_header + '# --- Utils ---', content)


# 2. Re-insert the sidebar navigation (replacing the top radio header)
new_sidebar = '''
# --- Sidebar Navigation ---
st.sidebar.title("📚 Course Catalog")
st.sidebar.caption("Structured learning tracks • Mission-based experiments")
st.sidebar.markdown(
    """
    <div class="sidebar-card">
        <h4>🎓 Learning Flow</h4>
        <p>1) Pick a track<br>2) Choose a module<br>3) Run and compare experiments</p>
    </div>
    """,
    unsafe_allow_html=True,
)

category = st.sidebar.radio("🎯 Choose Track", ["1. Fundamentals", "2. Advanced Processing"])

if category == "1. Fundamentals":
    st.sidebar.markdown("<span style='font-size:0.78rem;opacity:0.9;color:var(--text-muted);'>Core concepts and intuition builders</span>", unsafe_allow_html=True)
    module = st.sidebar.selectbox("🧩 Select Module", [
        "1.1 Visual Perception", 
        "1.2 EM Spectrum", 
        "1.3 Acquisition", 
        "1.4 Sampling & Quantization", 
        "1.5 Pixel Connectivity", 
        "1.6 Math Tools",
        "1.7 Distance Measures",
        "1.8 Connected Components",
        "1.9 Image Statistics"
    ])
else:
    st.sidebar.markdown("<span style='font-size:0.78rem;opacity:0.9;color:var(--text-muted);'>Hands-on processing and filtering labs</span>", unsafe_allow_html=True)
    module = st.sidebar.selectbox("⚙️ Select Module", [
        "2.1 Frequency Domain", 
        "2.2 Spatial Filtering", 
        "2.3 Morphology"
    ])

st.sidebar.divider()
st.sidebar.caption("Tip: Start from Fundamentals before moving to Advanced modules.")
'''

pattern_nav = r'(?s)# --- GAME ENGINE HEADER NAVIGATION ---.*?# ==========================================\n# PART 1'
content = re.sub(pattern_nav, new_sidebar + '\n# ==========================================\n# PART 1', content)

# 3. Handle Module 1.1 revert to 2-columns (as per diff)
pattern_11 = r'(?s)if module == "1.1 Visual Perception":.*?elif module == "1.2 EM Spectrum":'
module_11_str = '''if module == "1.1 Visual Perception":
    st.header("1. Elements of Visual Perception")
    st.write("Explore how the eye adapts to brightness (Scotopic vs Photopic) and visual illusions.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Controls")
        illum_log = st.slider("Illumination (Log Scale)", -2.0, 1.0, 0.0, 0.1)
        mach_bands = st.checkbox("Show Mach Bands")
        
        illum_lin = (illum_log + 2.0) / 3.0 # Normalize -2..1 to 0..1
        illum_lin = max(0.0, min(1.0, illum_lin))
        
        st.metric("Linear Intensity", f"{illum_lin:.2f}")
        st.metric("Vision Mode", "Scotopic (Rods)" if illum_lin < 0.2 else "Photopic (Cones)")

    with col2:
        st.subheader("Simulation")
        img = generate_flower_scene(illum_lin, mach_bands)
        st.image(img, channels="BGR", caption="Simulated Scene", use_container_width=True)

elif module == "1.2 EM Spectrum":'''
content = re.sub(pattern_11, module_11_str, content)

# 4. Handle Module 1.6 Math Tools completely replacing our existing 1.6 logic (3 pane) with the Diff's massive tab logic
pattern_16 = r'(?s)elif module == "1.6 Math Tools":.*?elif module == "1.7 Distance Measures":'
module_16_str = '''elif module == "1.6 Math Tools":
    st.header("6. Mathematical Functions in Digital Image Processing")
    st.caption("Analyze arithmetic, statistical, trigonometric, transform, morphology, and filtering functions with visual demos.")

    base = np.linspace(0, 255, 256).reshape(1, 256).repeat(256, 0).astype(np.uint8)
    roi = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(roi, (56, 56), (200, 200), 255, -1)

    tab_arith, tab_stats, tab_trig, tab_transform, tab_morph, tab_filter = st.tabs([
        "Arithmetic", "Statistical", "Trigonometric", "Transforms", "Morphology", "Filtering"
    ])

    with tab_arith:
        st.subheader("Arithmetic Functions")
        arith = st.selectbox(
            "Choose Arithmetic Function",
            ["Addition", "Subtraction", "Multiplication", "Division", "Logarithmic", "Exponential", "Power-law (Gamma)"],
        )

        result = base.astype(np.float32)
        if arith == "Addition":
            delta = st.slider("Addition Constant", 0, 120, 40)
            result = base.astype(np.float32) + delta
            st.info("Used for brightness increase.")
        elif arith == "Subtraction":
            delta = st.slider("Subtraction Constant", 0, 120, 40)
            result = base.astype(np.float32) - delta
            st.info("Used for brightness reduction/background removal.")
        elif arith == "Multiplication":
            scale = st.slider("Multiplication Scale", 0.2, 3.0, 1.5, 0.1)
            result = base.astype(np.float32) * scale
            st.info("Used for contrast scaling.")
        elif arith == "Division":
            denom = st.slider("Division Constant", 1.0, 8.0, 2.0, 0.1)
            result = base.astype(np.float32) / denom
            st.info("Used for controlled intensity compression.")
        elif arith == "Logarithmic":
            c = st.slider("Log Constant (c)", 5.0, 80.0, 30.0, 1.0)
            result = c * np.log1p(base.astype(np.float32))
            st.info("Enhances darker pixels and compresses high intensities.")
        elif arith == "Exponential":
            alpha = st.slider("Exponential Strength", 1.0, 6.0, 3.0, 0.1)
            norm = base.astype(np.float32) / 255.0
            result = ((np.exp(alpha * norm) - 1) / (np.exp(alpha) - 1)) * 255
            st.info("Non-linear enhancement emphasizing bright regions.")
        else:
            gamma = st.slider("Gamma (γ)", 0.2, 3.0, 1.6, 0.1)
            result = 255 * ((base.astype(np.float32) / 255.0) ** gamma)
            st.info("Power-law transform (gamma correction) for display tuning.")

        result = np.clip(result, 0, 255).astype(np.uint8)
        display_images(base, result, ("Input Gradient", f"{arith} Output"))

    with tab_stats:
        st.subheader("Statistical Functions")
        noise_sigma = st.slider("Add Gaussian Noise (σ)", 0, 60, 20)
        noisy = np.clip(base.astype(np.float32) + np.random.normal(0, noise_sigma, base.shape), 0, 255).astype(np.uint8)

        mean_val = float(np.mean(noisy))
        median_val = float(np.median(noisy))
        mode_val = int(np.argmax(np.bincount(noisy.flatten(), minlength=256)))
        var_val = float(np.var(noisy))
        std_val = float(np.std(noisy))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean", f"{mean_val:.2f}")
        c2.metric("Median", f"{median_val:.2f}")
        c3.metric("Mode", f"{mode_val}")
        c4.metric("Variance", f"{var_val:.2f}")
        c5.metric("Std Dev", f"{std_val:.2f}")

        hist = np.bincount(noisy.flatten(), minlength=256)
        pdf = hist / hist.sum()
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
        ax[0].plot(hist, color='cyan')
        ax[0].set_title('Histogram')
        ax[0].set_xlabel('Intensity')
        ax[1].plot(pdf, color='orange')
        ax[1].set_title('Probability Density Function (PDF)')
        ax[1].set_xlabel('Intensity')
        st.pyplot(fig)
        display_images(base, noisy, ("Reference", "Noisy Input for Analysis"))

    with tab_trig:
        st.subheader("Trigonometric Functions (sin, cos, tan)")
        freq = st.slider("Signal Frequency", 1, 20, 5)
        x = np.linspace(0, 2 * np.pi, 500)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(x, np.sin(freq * x), label='sin(x)', color='cyan')
        ax.plot(x, np.cos(freq * x), label='cos(x)', color='magenta')
        ax.plot(x, np.tan(freq * x), label='tan(x)', color='yellow', alpha=0.5)
        ax.set_ylim(-4, 4)
        ax.legend()
        ax.set_title('Trigonometric Basis used in Fourier Analysis')
        st.pyplot(fig)
        st.info("sin/cos components form the basis of DFT/FFT for frequency decomposition.")

    with tab_transform:
        st.subheader("Transform Functions")
        dft = np.fft.fftshift(np.fft.fft2(base))
        dft_mag = 20 * np.log1p(np.abs(dft))

        fft = fftpack.fftshift(fftpack.fft2(base))
        fft_mag = 20 * np.log1p(np.abs(fft))

        dct = cv2.dct(base.astype(np.float32))
        dct_mag = np.log1p(np.abs(dct))

        small = cv2.resize(base, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
        low = (small[0::2, 0::2] + small[0::2, 1::2] + small[1::2, 0::2] + small[1::2, 1::2]) / 2
        high_h = (small[0::2, 0::2] - small[0::2, 1::2] + small[1::2, 0::2] - small[1::2, 1::2]) / 2
        high_v = (small[0::2, 0::2] + small[0::2, 1::2] - small[1::2, 0::2] - small[1::2, 1::2]) / 2
        high_d = (small[0::2, 0::2] - small[0::2, 1::2] - small[1::2, 0::2] + small[1::2, 1::2]) / 2
        wavelet_panel = np.block([
            [low, high_h],
            [high_v, high_d],
        ])

        c1, c2 = st.columns(2)
        c1.image(dft_mag, caption="DFT Magnitude", use_container_width=True, clamp=True)
        c2.image(fft_mag, caption="FFT Magnitude", use_container_width=True, clamp=True)
        c3, c4 = st.columns(2)
        c3.image(dct_mag, caption="DCT Coefficients (JPEG concept)", use_container_width=True, clamp=True)
        c4.image(np.abs(wavelet_panel), caption="1-level Haar Wavelet Decomposition", use_container_width=True, clamp=True)

    with tab_morph:
        st.subheader("Morphological Functions")
        morph_op = st.selectbox("Morphology Operation", ["Dilation", "Erosion", "Opening", "Closing"])
        k = st.slider("Kernel Size", 1, 25, 5, step=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = (roi > 0).astype(np.uint8) * 255

        if morph_op == "Dilation":
            morph_res = cv2.dilate(binary, kernel, iterations=1)
        elif morph_op == "Erosion":
            morph_res = cv2.erode(binary, kernel, iterations=1)
        elif morph_op == "Opening":
            morph_res = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        else:
            morph_res = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        display_images(binary, morph_res, ("Input Binary Shape", morph_op))

    with tab_filter:
        st.subheader("Filtering Functions")
        filter_choice = st.selectbox("Select Filter Function", ["Convolution", "Correlation", "Gaussian", "Laplacian", "Sobel"])

        if filter_choice == "Convolution":
            kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9
            filtered = cv2.filter2D(base, -1, cv2.flip(kernel, -1))
        elif filter_choice == "Correlation":
            kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
            filtered = cv2.filter2D(base, -1, kernel)
        elif filter_choice == "Gaussian":
            filtered = cv2.GaussianBlur(base, (7, 7), 1.2)
        elif filter_choice == "Laplacian":
            filtered = cv2.convertScaleAbs(cv2.Laplacian(base, cv2.CV_64F))
        else:
            sx = cv2.Sobel(base, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(base, cv2.CV_64F, 0, 1, ksize=3)
            filtered = cv2.convertScaleAbs(np.hypot(sx, sy))

        display_images(base, filtered, ("Input Image", f"{filter_choice} Output"))

elif module == "1.7 Distance Measures":'''
content = re.sub(pattern_16, module_16_str, content)


with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

