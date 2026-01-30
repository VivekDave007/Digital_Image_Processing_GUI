import streamlit as st
import time

try:
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from scipy.spatial import distance
except ImportError as e:
    st.error(f"Critical Dependency Error: {e}")
    st.info("Attempting to run with limited functionality...")
    # Minimal fallback or stop
    st.stop()

# Set Page Config
st.set_page_config(
    page_title="Digital Image Processing - Ch2",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    h1 { color: #00ADB5; }
    h2 { color: #EEEEEE; }
    h3 { color: #00ADB5; }
    .stSlider > div > div > div > div { background-color: #00ADB5; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
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

# --- Sidebar Navigation ---
st.sidebar.title("DIP Fundamentals")
st.sidebar.caption("Gonzalez & Woods, Chapter 2")
module = st.sidebar.radio("Select Module", [
    "1. Visual Perception", 
    "2. EM Spectrum", 
    "3. Acquisition", 
    "4. Sampling", 
    "5. Relationships", 
    "6. Math Tools"
])

st.sidebar.divider()
st.sidebar.info("Developed by Vivek Dave. \nDeployment Ready.")

# --- Main Content ---
st.title("Interactive Digital Image Processing")

# ==========================================
# MODULE 1: VISUAL PERCEPTION
# ==========================================
if module == "1. Visual Perception":
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


# ==========================================
# MODULE 2: EM SPECTRUM
# ==========================================
elif module == "2. EM Spectrum":
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

# ==========================================
# MODULE 3: ACQUISITION
# ==========================================
elif module == "3. Acquisition":
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


# ==========================================
# MODULE 4: SAMPLING
# ==========================================
elif module == "4. Sampling":
    st.header("4. Sampling & Quantization")
    
    col_ctrl, col_img = st.columns([1, 2])
    
    with col_ctrl:
        N = st.selectbox("Spatial Sampling (N)", [512, 256, 128, 64, 32, 16], index=0)
        k = st.slider("Quantization Levels (k bits)", 1, 8, 8)
        
    with col_img:
        # Generate Source
        x = np.linspace(0, 1, 512)
        xv, yv = np.meshgrid(x, x)
        src = (xv * 255).astype(np.uint8) # Gradient
        cv2.circle(src, (256, 256), 100, 255 if k>1 else 1, -1) # Inverted circle logic would need masking, keeping simple
        if k > 1:
             mask = ((xv-0.5)**2 + (yv-0.5)**2 < 0.3**2)
             src[mask] = 255 - src[mask]
        
        # 1. Sampling (Resize Down -> Up)
        small = cv2.resize(src, (N, N), interpolation=cv2.INTER_NEAREST)
        resampled = cv2.resize(small, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # 2. Quantization
        levels = 2 ** k
        factor = 256 / levels
        quantized = (np.floor(resampled / factor) * factor).astype(np.uint8)
        
        st.image(quantized, caption=f"Result: {N}x{N}, {k}-bit", use_container_width=True)
        if N < 64: st.warning("Notice the 'Checkerboard' pixelation effect.")
        if k < 4: st.warning("Notice the 'False Contouring' bands in the gradient.")


# ==========================================
# MODULE 5: RELATIONSHIPS
# ==========================================
elif module == "5. Relationships":
    st.header("5. Pixel Relationships")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Inputs")
        p_x = st.number_input("Pixel P (x)", 0, 9, 4)
        p_y = st.number_input("Pixel P (y)", 0, 9, 4)
        
        metric = st.selectbox("Distance Metric", ["Euclidean", "City-Block (D4)", "Chessboard (D8)"])
        
    with col2:
        # Compute Distances for 10x10 grid
        grid = np.zeros((10, 10))
        for r in range(10):
            for c in range(10):
                if metric == "Euclidean":
                    d = np.sqrt((r-p_y)**2 + (c-p_x)**2)
                elif metric == "City-Block (D4)":
                    d = abs(r-p_y) + abs(c-p_x)
                else: # Chessboard
                    d = max(abs(r-p_y), abs(c-p_x))
                grid[r, c] = d
                
        fig, ax = plt.subplots()
        im = ax.imshow(grid, cmap="viridis", origin='upper')
        
        # Annotate
        for r in range(10):
            for c in range(10):
                ax.text(c, r, f"{grid[r,c]:.1f}", ha="center", va="center", color="w", fontsize=8)
                
        # Highlight Center
        ax.add_patch(plt.Rectangle((p_x-0.5, p_y-0.5), 1, 1, fill=False, edgecolor='red', lw=3))
        
        st.pyplot(fig)
        st.caption("Distance Map from P (Red Box). Values show distance $D(p, q)$.")


# ==========================================
# MODULE 6: MATH TOOLS
# ==========================================
elif module == "6. Math Tools":
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
