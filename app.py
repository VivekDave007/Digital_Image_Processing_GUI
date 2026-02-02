import streamlit as st
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
    page_title="DIP Workbench Ultimate",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
# --- CSS Styling ---
# --- CSS Styling ---
# --- CSS Styling ---
# --- CSS Styling ---
st.markdown("""
<style>
    /* Import Google Font 'Inter' */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* --- KEYFRAME ANIMATIONS --- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translate3d(0, 20px, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* --- ADAPTIVE GLOBAL STYLES (Uses Theme Variables) --- */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
        scroll-behavior: smooth;
    }
    
    .stApp {
        background-color: var(--background-color);
        animation: fadeInUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
    }
    
    /* --- HEADERS --- */
    h1 {
        /* Simplified for Reliability: Solid Color with Glow */
        color: var(--primary-color) !important;
        font-weight: 800 !important;
        font-size: 3.5rem !important;
        padding-bottom: 0.5rem;
        /* Dynamic Shadow: Uses primary color for a subtle glow */
        text-shadow: 0 0 20px rgba(128, 128, 128, 0.2);
    }
    
    h2, h3 {
        color: var(--text-color) !important;
        border-bottom: 2px solid var(--primary-color);
    }
    
    /* --- SIDEBAR --- */
    section[data-testid="stSidebar"] {
        /* Default Desktop behavior using variables */
        background-color: var(--secondary-background-color);
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Mobile-specific fix: ULTRA-AGGRESSIVE OPAQUE OVERRIDE */
    @media (max-width: 768px) {
        
        /* Target the main container and ALL internal divs recursively */
        section[data-testid="stSidebar"], 
        section[data-testid="stSidebar"] div,
        div[data-testid="stSidebarUserContent"],
        div[data-testid="stSidebarNav"] {
            /* Force the background to be the solid main theme color */
            background-color: var(--background-color) !important;
            
            /* Remove any transparency layers or gradients */
            background-image: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }

        /* Re-apply z-index and border only to the parent container */
        section[data-testid="stSidebar"] {
            border-right: 2px solid rgba(128, 128, 128, 0.2) !important;
            box-shadow: 2px 0 10px rgba(0,0,0,0.2) !important;
            opacity: 1 !important;
            z-index: 999999 !important;
        }
        
        /* Exceptions: Allow buttons/inputs to keep their specific styles, 
           otherwise they become invisible rectangular blocks of color. */
        section[data-testid="stSidebar"] button, 
        section[data-testid="stSidebar"] input, 
        section[data-testid="stSidebar"] [role="checkbox"],
        section[data-testid="stSidebar"] .stSlider,
        section[data-testid="stSidebar"] .stMarkdown, /* Text needs to be visible on top */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
             background-color: transparent !important; 
        }
    }
    
    /* Sidebar Title */
    section[data-testid="stSidebar"] h1 {
        font-size: 2.0rem !important;
        text-align: left !important;
        background: none !important;
        -webkit-text-fill-color: var(--primary-color) !important;
        color: var(--primary-color) !important;
        margin-bottom: 0;
        padding-left: 0.2rem;
    }
    
    /* --- WIDGETS & INPUTS --- */
    /* Let Streamlit handle input backgrounds natively (best for light/dark switching) */
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        border-radius: 8px; 
    }
    
    /* --- BUTTONS --- */
    .stButton > button {
        background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
        color: var(--primary-color);
        border: 1px solid var(--primary-color);
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: var(--primary-color);
        color: var(--background-color); /* Invert on hover */
        box-shadow: 0 0 15px rgba(0, 173, 181, 0.4);
        transform: translateY(-2px);
    }
    
    /* --- IMAGES --- */
    img {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    img:hover {
        transform: scale(1.02);
    }
    
    /* --- ALERTS & EXPANDERS --- */
    .stAlert {
        background-color: var(--secondary-background-color);
        border-left: 4px solid var(--primary-color);
    }
    
    .streamlit-expanderHeader {
        background-color: var(--secondary-background-color);
        color: var(--text-color) !important;
        border-radius: 6px;
    }

</style>
""", unsafe_allow_html=True)

st.title("Digital Image Processing Workbench")

# --- Utils ---
@st.cache_data
def load_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return img

def display_images(original, processed, titles=("Original Image", "Processed Image")):
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption=titles[0], use_container_width=True, clamp=True, channels='GRAY')
    with col2:
        st.image(processed, caption=titles[1], use_container_width=True, clamp=True, channels='GRAY')

# --- Helper Functions (Fundamentals) ---
def generate_flower_scene(illumination, mach_bands):
    """Generates the Module 1 Scene using OpenCV"""
    w, h = 400, 300
    if mach_bands:
        # Draw Mach Bands
        img = np.zeros((h, w, 3), dtype=np.uint8)
        steps = 10
        sw = w // steps
        for i in range(steps):
            val = int(255 * (i/steps) * illumination)
            cv2.rectangle(img, (i*sw, 0), ((i+1)*sw, h), (val, val, val), -1)
        return img
    else:
        # Draw Flower
        img = np.zeros((h, w, 3), dtype=np.uint8)
        bg = int(255 * illumination)
        img[:] = (bg, bg, bg)
        
        is_scotopic = illumination < 0.2
        
        center = (w//2, h//2)
        radius = 80
        
        if is_scotopic:
            color_flower = (100, 100, 100) # Gray
            color_center = (50, 50, 50)
        else:
            color_flower = (50, 50, 255) # Red (BGR)
            color_center = (0, 255, 255) # Yellow (BGR)
            
        cv2.circle(img, center, radius, color_flower, -1)
        cv2.circle(img, center, 30, color_center, -1)
        
        return img

# --- Frequency Domain Helpers ---
def get_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # Log transformation for visualization: s = c * log(1 + r)
    magnitude_spectrum = 20 * np.log(1 + np.abs(fshift))
    return fshift, magnitude_spectrum

def create_filter(shape, filter_type, cutoff, order=1):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u, v, indexing='ij')
    d_uv = np.sqrt((u - crow)**2 + (v - ccol)**2)
    
    mask = np.zeros((rows, cols), dtype=np.float32)

    if filter_type == "Ideal Lowpass":
        mask[d_uv <= cutoff] = 1
    elif filter_type == "Ideal Highpass":
        mask[d_uv > cutoff] = 1
    elif filter_type == "Gaussian Lowpass":
        mask = np.exp(-(d_uv**2) / (2 * (cutoff**2)))
    elif filter_type == "Gaussian Highpass":
        mask = 1 - np.exp(-(d_uv**2) / (2 * (cutoff**2)))
    elif filter_type == "Butterworth Lowpass":
        mask = 1 / (1 + (d_uv / cutoff)**(2 * order))
    elif filter_type == "Butterworth Highpass":
        mask = 1 - (1 / (1 + (d_uv / cutoff)**(2 * order)))
        
    return mask

@st.cache_data
def apply_frequency_filter(img, filter_type, cutoff, order, pad_image=True):
    rows, cols = img.shape
    
    # Padding to avoid wraparound errors (Pad to double size)
    if pad_image:
        padded_rows, padded_cols = 2 * rows, 2 * cols
        padded_img = np.zeros((padded_rows, padded_cols), dtype=img.dtype)
        padded_img[:rows, :cols] = img
    else:
        padded_rows, padded_cols = rows, cols
        padded_img = img

    # FFT
    f = np.fft.fft2(padded_img)
    fshift = np.fft.fftshift(f)
    
    # Create Filter
    mask = create_filter((padded_rows, padded_cols), filter_type, cutoff, order)
    
    # Apply Filter
    fshift_filtered = fshift * mask
    
    # Inverse FFT
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Crop back if padded
    if pad_image:
        img_back = img_back[:rows, :cols]
        
    return img_back, mask, fshift_filtered

# --- Spatial Filtering Helpers ---
def add_noise(img, noise_type, param1=0, param2=0):
    noisy_img = img.copy()
    if noise_type == "Gaussian":
        mean = param1
        sigma = param2
        gauss = np.random.normal(mean, sigma, img.shape).reshape(img.shape)
        noisy_img = img + gauss
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
    elif noise_type == "Salt & Pepper":
        prob = param1
        thres = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = np.random.random()
                if rdn < prob:
                    noisy_img[i][j] = 0
                elif rdn > thres:
                    noisy_img[i][j] = 255

    elif noise_type == "Periodic":
        # Add sinusoidal noise
        freq = param1
        rows, cols = img.shape
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x, y)
        sine_noise = param2 * np.sin(2 * np.pi * freq * X / cols) + param2 * np.sin(2 * np.pi * freq * Y / rows)
        noisy_img = img + sine_noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return noisy_img

@st.cache_data
def apply_spatial_filter(img, filter_name, kernel_size, sigma_x=0):
    if filter_name == "Gaussian Blur":
        # In strict spatial filtering, Gaussian kernel is symmetric so flip doesn't change it,
        # but conceptually we apply convolution.
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma_x)
    elif filter_name == "Median Filter":
        return cv2.medianBlur(img, kernel_size)
    elif filter_name == "Custom Convolution":
        # Example of explicit convolution with flipping
        # Let's create a simple averaging kernel for demo
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        
        # MATH CORRECTION: Flip the kernel 180 degrees for Convolution
        kernel_flipped = cv2.flip(kernel, -1) 
        
        # cv2.filter2D performs Correlation. With flipped kernel, it becomes Convolution.
        return cv2.filter2D(img, -1, kernel_flipped)
    return img

# --- Morphology Helpers ---
@st.cache_data
def apply_morphology(img, op_type, struct_elem_shape, kernel_size):
    shape_dict = {
        "Rect": cv2.MORPH_RECT,
        "Cross": cv2.MORPH_CROSS,
        "Ellipse": cv2.MORPH_ELLIPSE
    }
    shape = shape_dict[struct_elem_shape]
    kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
    
    if op_type == "Erosion":
        return cv2.erode(img, kernel, iterations=1)
    elif op_type == "Dilation":
        return cv2.dilate(img, kernel, iterations=1)
    elif op_type == "Opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif op_type == "Closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


# --- Sidebar Navigation ---
st.sidebar.title("DIP Workbench")
st.sidebar.caption("Comprehensive Image Processing Suite")

category = st.sidebar.radio("Select Category", ["1. Fundamentals", "2. Advanced Processing"])

if category == "1. Fundamentals":
    module = st.sidebar.selectbox("Select Module", [
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
    module = st.sidebar.selectbox("Select Module", [
        "2.1 Frequency Domain", 
        "2.2 Spatial Filtering", 
        "2.3 Morphology"
    ])

st.sidebar.divider()


# ==========================================
# PART 1: FUNDAMENTALS (Logic from app (2).py)
# ==========================================

if module == "1.1 Visual Perception":
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
        
        if mach_bands:
            st.warning("Note the 'overshoot' brightness at the edges (Mach Band Effect).")


elif module == "1.2 EM Spectrum":
    st.header("2. Light & The Electromagnetic Spectrum")
    
    # Log Freq: 6 (Radio) to 22 (Gamma)
    log_freq = st.slider("Frequency (Log Hz)", 6.0, 22.0, 14.5, 0.1)
    
    freq = 10**log_freq
    wavelength = 3e8 / freq
    energy = 4.135e-15 * freq
    
    col_info, col_viz = st.columns(2)
    
    with col_info:
        st.subheader("Physics")
        st.latex(r"\nu = " + f"{freq:.2e} Hz")
        st.latex(r"\lambda = c / \nu = " + f"{wavelength:.2e} m")
        st.latex(r"E = h\nu = " + f"{energy:.2e} eV")
        
    with col_viz:
        st.subheader("Band & Visualization")
        
        if log_freq < 9:
            st.success("Band: **Radio Waves**")
            st.info("📡 Used for TV/Radio broadcasting.")
        elif log_freq < 12:
            st.success("Band: **Microwaves**")
            st.info("📶 Used for Radar & WiFi.")
        elif log_freq < 14.5:
            st.warning("Band: **Infrared**")
            st.markdown("🔥 **Heat Energy / Thermal Imaging**")
        elif log_freq < 15.0:
            st.error("Band: **Visible Light**")
            st.markdown("🌈 **The Human Visual Window**")
            # Show a rainbow gradient or color
        elif log_freq < 17:
            st.info("Band: **Ultraviolet**")
            st.markdown("☀️ Causes sunburn & fluorescence.")
        elif log_freq < 20:
            st.info("Band: **X-Rays**")
            st.markdown("🦴 **Medical Imaging / Security Scanning**")
        else:
            st.error("Band: **Gamma Rays**")
            st.markdown("☢️ Nuclear radiation.")

elif module == "1.3 Acquisition":
    st.header("3. Image Sensing & Acquisition")
    
    tab1, tab2, tab3 = st.tabs(["Single Sensor", "Strip Sensor", "Array Sensor"])
    
    with tab1:
        st.write("**Single Sensor (e.g. Laser Drum Scanner)**: Moves pixel by pixel.")
        if st.button("Simulate Single Capture"):
            # Create a placeholder for the image
            image_placeholder = st.empty()
            status_text = st.empty()
            
            # Create a dummy image (grayscale gradient)
            GRID_SIZE = 20
            img = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            
            # Animate pixel by pixel 
            # (We skip some frames for speed, otherwise 400 updates is too slow for Streamlit)
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    # Fill pixel with a pattern
                    img[y, x] = int((x+y) * 255 / (GRID_SIZE*2))
                    
                    # Update UI every row (to be faster)
                if y % 2 == 0:
                     # Scale up for visibility
                     big_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
                     image_placeholder.image(big_img, caption=f"Scanning Row {y+1}/{GRID_SIZE}...", use_container_width=False, clamp=True)
                     time.sleep(0.05)
            
            # Final Show
            big_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
            image_placeholder.image(big_img, caption="Capture Complete", use_container_width=False, clamp=True)
            status_text.success("Image Captured Row-by-Row, Pixel-by-Pixel!")
            
    with tab2:
        st.write("**Sensor Strip (e.g. Flatbed Scanner)**: Moves row by row.")
        if st.button("Simulate Strip Capture"):
            image_placeholder = st.empty()
            GRID_SIZE = 20
            img = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            
            for y in range(GRID_SIZE):
                # Fill entire row
                for x in range(GRID_SIZE):
                    img[y, x] = int((x+y) * 255 / (GRID_SIZE*2))
                
                big_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
                image_placeholder.image(big_img, caption=f"Scanning Row {y+1}/{GRID_SIZE}...", use_container_width=False, clamp=True)
                time.sleep(0.1)
                
            status_text = st.success("Image Captured Row-by-Row!")

    with tab3:
        st.write("**Sensor Array (e.g. Digital Camera)**: Captures instantly.")
        if st.button("Simulate Array Capture"):
            with st.spinner("Flash..."):
                time.sleep(0.2)
            st.success("📸 SNAP! Instant Capture.")


elif module == "1.4 Sampling & Quantization":
    sampling_quantization_extended.run()


elif module == "1.5 Pixel Connectivity":
    pixel_relationships.run()
    # Old 1.5 logic removed in favor of pixel_relationships.run() which is better.
    # We kept the block structure to replacement work correctly.
    pass


elif module == "1.6 Math Tools":
    st.header("6. Mathematical Tools")
    
    op = st.selectbox("Operation", ["Addition (Noise Reduction)", "Subtraction (Motion Detection)", "Multiplication (ROI Masking)"])
    
    col_a, col_b, col_res = st.columns(3)
    
    # Generate Bases
    img_a = np.linspace(0, 255, 256).reshape(1, 256).repeat(256, 0).astype(np.uint8) # Gradient
    
    img_b = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img_b, (50, 50), (200, 200), 255, -1) # Square
    
    with col_a:
        st.image(img_a, caption="Image A (Gradient)", use_container_width=True)
        
    with col_b:
        st.image(img_b, caption="Image B (Square)", use_container_width=True)
        
    with col_res:
        if "Addition" in op:
            # Noise Demo
            noise = np.random.normal(0, 50, img_a.shape).astype(np.float32)
            noisy = np.clip(img_a + noise, 0, 255).astype(np.uint8)
            
            # Simulated Average
            st.image(noisy, caption="Single Noisy Frame", use_container_width=True)
            st.caption("Averaging 100 frames restores Image A.")
            
        elif "Subtraction" in op:
            res = cv2.absdiff(img_a, img_b) # Just difference for visual
            # Logic: B - A
            # Here let's do (A + Object) - A
            scene_with_object = img_a.copy()
            scene_with_object[50:200, 50:200] = 200
            
            res = cv2.absdiff(scene_with_object, img_a)
            st.image(res, caption="Result (Difference)", use_container_width=True)
            
        elif "Multiplication" in op:
            # Masking
            mask = (img_b > 0).astype(np.float32)
            res = (img_a.astype(np.float32) * mask).astype(np.uint8)
            st.image(res, caption="Result (Masked)", use_container_width=True)


elif module == "1.7 Distance Measures":
    distance_measures.run()

elif module == "1.8 Connected Components":
    connected_components.run()

elif module == "1.9 Image Statistics":
    image_statistics.run()


# ==========================================
# PART 2: ADVANCED PROCESSING (Logic from app.py - Workbench)
# ==========================================
elif category == "2. Advanced Processing":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg", "bmp", "tif"])

    if uploaded_file:
        original_img = load_image(uploaded_file)
        
        if module == "2.1 Frequency Domain":
            st.header("Frequency Domain Filtering")
            with st.expander("📘 Theory: Frequency Domain & FFT"):
                st.write(r"""
                **Concept**: Converts the image from spatial domain $(x,y)$ to frequency domain $(u,v)$.
                - **Low Frequencies**: Provide image structure (smooth regions).
                - **High Frequencies**: Provide edges and details.
                
                **Math**:
                $F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) e^{-j 2\pi (ux/M + vy/N)}$
                
                **Filters**:
                - **Ideal**: Sharp cutoff (Causes ringing/Gibbs phenomenon).
                - **Butterworth**: Smooth transition (Order $n$ controls sharpness).
                - **Gaussian**: No ringing (Fourier transform of Gaussian is Gaussian).
                """)
            
            filter_type = st.sidebar.selectbox("Filter Type", 
                                               ["Ideal Lowpass", "Ideal Highpass", 
                                                "Gaussian Lowpass", "Gaussian Highpass", 
                                                "Butterworth Lowpass", "Butterworth Highpass"])
            
            cutoff = st.sidebar.slider("Cutoff Frequency (D0)", 10, 200, 50)
            
            order = 1
            if "Butterworth" in filter_type:
                order = st.sidebar.slider("Butterworth Order (n)", 1, 10, 2)
                
            pad_choice = st.sidebar.checkbox("Use Padding (Avoid Wraparound)", value=True)
            
            # Visualization of Spectrum
            st.subheader("Frequency Spectrum Analysis")
            col_spec1, col_spec2, col_spec3 = st.columns(3)
            
            # 1. Original Spectrum
            _, mag_spec = get_spectrum(original_img)
            col_spec1.image(mag_spec / np.max(mag_spec), caption="Original Spectrum", clamp=True, use_container_width=True)
            
            # Processing
            processed_img, mask, fshift_filtered = apply_frequency_filter(original_img, filter_type, cutoff, order, pad_choice)
            
            # 2. Filter Mask
            col_spec2.image(mask, caption="Filter Mask", clamp=True, use_container_width=True)
            
            # 3. Filtered Spectrum
            mag_spec_filtered = 20 * np.log(1 + np.abs(fshift_filtered))
            col_spec3.image(mag_spec_filtered / np.max(mag_spec), caption="Filtered Spectrum", clamp=True, use_container_width=True)
            
            st.subheader("Spatial Result")
            display_images(original_img, processed_img)
            
        elif module == "2.2 Spatial Filtering":
            st.header("Spatial Filtering")
            with st.expander("📘 Theory: Spatial Convolution & Noise"):
                st.write(r"""
                **Convolution**:
                $g(x,y) = w(x,y) * f(x,y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} w(s,t) f(x-s, y-t)$
                *Note: The kernel $w$ requires flipping by 180 degrees.*
                
                **Filters**:
                - **Gaussian Blur**: Weighted average (Standard deviation $\sigma$).
                - **Median Filter**: Replaces pixel with median of neighbors. (Best for Salt & Pepper noise).
                """)
            
            action = st.sidebar.radio("Action", ["Add Noise", "Apply Filter"])
            
            if action == "Add Noise":
                noise_type = st.sidebar.selectbox("Noise Type", ["Gaussian", "Salt & Pepper", "Periodic"])
                param1, param2 = 0.0, 0.0
                
                if noise_type == "Gaussian":
                    param1 = st.sidebar.slider("Mean", -50.0, 50.0, 0.0)
                    param2 = st.sidebar.slider("Sigma", 0.0, 100.0, 25.0)
                elif noise_type == "Salt & Pepper":
                    param1 = st.sidebar.slider("Probability", 0.0, 1.0, 0.05)
                elif noise_type == "Periodic":
                    param1 = st.sidebar.slider("Frequency", 1.0, 100.0, 20.0)
                    param2 = st.sidebar.slider("Amplitude", 0.0, 100.0, 30.0)
                    
                processed_img = add_noise(original_img, noise_type, param1, param2)
                display_images(original_img, processed_img, ("Original", f"Noisy ({noise_type})"))
                
            else:
                filter_name = st.sidebar.selectbox("Filter", ["Gaussian Blur", "Median Filter", "Custom Convolution"])
                k_size = st.sidebar.slider("Kernel Size", 1, 31, 5, step=2)
                
                sigma = 0
                if filter_name == "Gaussian Blur":
                    sigma = st.sidebar.slider("Sigma X", 0.1, 10.0, 1.0)
                    
                processed_img = apply_spatial_filter(original_img, filter_name, k_size, sigma)
                
                if filter_name == "Custom Convolution":
                     st.info("Note: Kernel was FLIPPED 180 degrees to perform true Convolution.")
                     
                display_images(original_img, processed_img)

        elif module == "2.3 Morphology":
            st.header("Morphological Operations")
            with st.expander("📘 Theory: Morphology"):
                st.write("""
                **Operations based on shapes (Structuring Elements)**.
                - **Erosion**: Shrinks bright regions (removes small anomalies).
                - **Dilation**: Expands bright regions (fills gaps).
                - **Opening**: Erosion followed by Dilation (Removes noise).
                - **Closing**: Dilation followed by Erosion (Fills holes).
                """)
            
            op_type = st.sidebar.selectbox("Operation", ["Erosion", "Dilation", "Opening", "Closing"])
            shape_txt = st.sidebar.selectbox("Structuring Element Shape", ["Rect", "Cross", "Ellipse"])
            k_size = st.sidebar.slider("Structuring Element Size", 1, 31, 5, step=2) # Must be odd for stability usually, but cv2 ok
            
            processed_img = apply_morphology(original_img, op_type, shape_txt, k_size)
            display_images(original_img, processed_img)
            
        # --- Download Section ---
        st.divider()
        if 'processed_img' in locals():
            is_success, buffer = cv2.imencode(".png", processed_img)
            if is_success:
                 st.download_button(
                    label="⬇️ Download Processed Image",
                    data=buffer.tobytes(),
                    file_name="processed_image.png",
                    mime="image/png"
                 )
            
    else:
        st.info("Please upload an image to use the Advanced Processing modules.")

st.sidebar.info("Developed by Vivek Dave.")



