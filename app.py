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
    page_title="PronOS DIP Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    /* Hero Card - Use Theme Aware standard transparent overlays */
    .hero-card {
        background: rgba(128, 128, 128, 0.05);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }

    .hero-card h1 {
        font-weight: 700 !important;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .hero-card p {
        font-size: 1.1rem;
        margin-bottom: 1rem;
        opacity: 0.8;
    }

    .status-chip {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        background-color: #367D5D;
        color: #FFFFFF !important;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Sidebar Custom Containers */
    .sidebar-card {
        background: rgba(128, 128, 128, 0.05);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .sidebar-card h4 {
        margin: 0 0 0.5rem 0;
        color: #367D5D !important;
    }
    .sidebar-card p {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.8;
    }

    /* Pill Buttons (Theme Aware) */
    .stButton > button {
        border-radius: 999px !important;
        font-weight: 600 !important;
        transition: 0.2s ease-in-out;
        padding: 0.6rem 2rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
    }

    /* Pill Tabs (Theme Aware) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px !important;
        padding: 8px 16px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
    }

    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
    }

    img {
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    /* Inputs rounded */
    .stTextInput > div > div > input, .stSelectbox > div > div > div, .stSlider > div > div > div > div {
        border-radius: 12px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
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


def generate_processed_vision_image(distance_m, light_intensity, zoom_level, global_img=None):
    """Simulate perceived/processed image based on distance and illumination."""
    h, w = 320, 320
    
    if global_img is not None:
        # Use global image, scale and zoom it based on distance
        input_img = cv2.resize(global_img, (w, h))
        scale = np.clip(zoom_level * (1.5 / max(distance_m, 0.3)), 0.2, 5.0)
        M = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
        img = cv2.warpAffine(input_img, M, (w, h))
    else:
        input_img = np.zeros((h, w), dtype=np.uint8)
        size = int(np.clip(70 * 1.0 * (1.5 / 1.5), 18, 120)) # Fixed size for input
        center = (w // 2, h // 2)
        cv2.circle(input_img, center, size, 190, -1)
        cv2.line(input_img, (center[0] - size, center[1]), (center[0] + size, center[1]), 240, 3)
        cv2.line(input_img, (center[0], center[1] - size), (center[0], center[1] + size), 240, 3)
        
        # apply zoom for the eye view
        img = np.zeros((h, w), dtype=np.uint8)
        size_perceived = int(np.clip(70 * zoom_level * (1.5 / max(distance_m, 0.3)), 18, 120))
        cv2.circle(img, center, size_perceived, 190, -1)
        cv2.line(img, (center[0] - size_perceived, center[1]), (center[0] + size_perceived, center[1]), 240, 3)
        cv2.line(img, (center[0], center[1] - size_perceived), (center[0], center[1] + size_perceived), 240, 3)

    # Illumination controls brightness + noise level
    brightness_scale = 0.55 + 0.9 * light_intensity
    img = np.clip(img.astype(np.float32) * brightness_scale, 0, 255)

    noise_sigma = 22 * (1.0 - light_intensity)
    noise = np.random.normal(0, noise_sigma, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Distance-driven blur (farther object => more blur)
    blur_k = int(np.clip(1 + 2 * distance_m, 1, 21))
    if blur_k % 2 == 0:
        blur_k += 1
    img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)

    # Basic enhancement stage (simulate digital post-processing)
    denoised = cv2.GaussianBlur(img, (3, 3), 0)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    enhanced = cv2.filter2D(denoised, -1, sharpen_kernel)

    return input_img, img, np.clip(enhanced, 0, 255).astype(np.uint8)




def simulate_physiological_vision(global_img, pupil, lens, cornea):
    if global_img is not None:
        img = cv2.resize(global_img, (320, 320))
    else:
        img = np.zeros((320, 320), dtype=np.uint8)
        cv2.circle(img, (160, 160), 80, 200, -1)
        cv2.line(img, (160, 60), (160, 260), 255, 5)
        cv2.line(img, (60, 160), (260, 160), 255, 5)
        
    result = img.astype(np.float32)

    # 1. Pupil Dilation -> Brightness (0.7 is baseline)
    brightness_factor = pupil / 0.7
    result = result * brightness_factor
    
    # 2. Lens Thickness -> Defocus blur (Myopia/Hyperopia)
    lens_dev = abs(lens - 0.7)
    if lens_dev > 0.05:
        k = int(lens_dev * 35)
        if k % 2 == 0: 
            k += 1
        if k >= 3:
            result = cv2.GaussianBlur(result, (k, k), 0)
            
    # 3. Cornea Bulge -> Astigmatism (directional blur)
    cornea_dev = cornea - 1.2
    if abs(cornea_dev) > 0.05:
        k = int(abs(cornea_dev) * 45)
        if k % 2 == 0: 
            k += 1
        if k >= 3:
            kernel = np.zeros((k, k), dtype=np.float32)
            if cornea_dev > 0:
                kernel[:, k//2] = 1.0 / k # Vertical blur for steep cornea
            else:
                kernel[k//2, :] = 1.0 / k # Horizontal blur for flat cornea
            result = cv2.filter2D(result, -1, kernel)
            
    return img, np.clip(result, 0, 255).astype(np.uint8)

def render_anatomical_eye_scene(highlight_part="None", show_labels=True, pupil_dilation=0.7, lens_thickness=0.7, cornea_bulge=1.2):
    """Render a 2D cross-section of the human eye anatomy with interactive highlighting."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#0a1228') # Cyberpunk dark background
    
    # Coordinates for center of the eye
    cx, cy = 5.0, 5.0
    radius = 3.0

    # Colors
    colors = {
        "Sclera": "#e0e0e0",
        "Cornea": "#a8d8ea",
        "Choroid": "#ff6b6b",
        "Retina": "#ffd166",
        "Lens": "#cdb4db",
        "Iris": "#457b9d",
        "Optic Nerve": "#f4a261",
        "Macula": "#ff9f1c"
    }

    # Apply highlight glow if selected
    def get_color(part_name, base_color):
        if highlight_part == part_name:
            return "#00f3ff" # Neon cyan glow for highlighted part
        return base_color

    def get_alpha(part_name, base_alpha):
        if highlight_part == part_name:
            return 1.0
        elif highlight_part != "None":
            return max(0.2, base_alpha - 0.4) # Dim others
        return base_alpha

    # Sclera (Outer white shell)
    sclera = plt.Circle((cx, cy), radius, color=get_color("Sclera", colors["Sclera"]), 
                        fill=False, lw=4, alpha=get_alpha("Sclera", 0.9))
    ax.add_patch(sclera)

    # Choroid (Vascular layer)
    choroid = plt.Circle((cx, cy), radius - 0.1, color=get_color("Choroid", colors["Choroid"]), 
                         fill=False, lw=3, alpha=get_alpha("Choroid", 0.8))
    ax.add_patch(choroid)

    # Retina (Inner neural layer - stops at anterior)
    theta1, theta2 = 45, 315 # degrees for the back curve
    retina = cv2.ellipse2Poly((int(cx*100), int(cy*100)), (int((radius-0.2)*100), int((radius-0.2)*100)), 
                              0, theta1, theta2, 5)
    retina_x = retina[:, 0] / 100.0
    retina_y = retina[:, 1] / 100.0
    ax.plot(retina_x, retina_y, color=get_color("Retina", colors["Retina"]), 
            lw=3, alpha=get_alpha("Retina", 0.9))

    # Macula (Fovea centralis)
    macula_y = cy
    macula_x = cx + radius - 0.2
    ax.plot([macula_x, macula_x], [macula_y - 0.3, macula_y + 0.3], 
            color=get_color("Macula", colors["Macula"]), lw=5, alpha=get_alpha("Macula", 1.0))

    # Optic Nerve (Exit pathway)
    nerve_y1, nerve_y2 = cy - 0.4, cy - 0.8
    nerve_x = cx + radius - 0.15
    ax.plot([nerve_x, nerve_x + 1.5], [nerve_y1, nerve_y1 - 0.2], 
            color=get_color("Optic Nerve", colors["Optic Nerve"]), lw=3, alpha=get_alpha("Optic Nerve", 0.9))
    ax.plot([nerve_x, nerve_x + 1.5], [nerve_y2, nerve_y2 - 0.2], 
            color=get_color("Optic Nerve", colors["Optic Nerve"]), lw=3, alpha=get_alpha("Optic Nerve", 0.9))
    # 'Blind spot' gap in retina
    ax.plot([nerve_x-0.05, nerve_x-0.05], [nerve_y2, nerve_y1], color='#0a1228', lw=4, zorder=5)

    # Cornea (Bulging front clear part)
    # cornea_bulge is 0.8 (flat) to 1.6 (steep) - standard is 1.2
    cornea_cx = cx - radius + 1.6 - cornea_bulge
    cornea_radius = cornea_bulge
    cornea = plt.Circle((cornea_cx, cy), cornea_radius, color=get_color("Cornea", colors["Cornea"]), 
                        fill=False, lw=3, alpha=get_alpha("Cornea", 0.7))
    ax.add_patch(cornea)
    # Mask inner part of cornea circle to make it a bulge
    ax.add_patch(plt.Rectangle((cx - radius, 0), radius, 10, color='#0a1228', zorder=2))

    # Re-draw the anterior sclera edge to meet cornea
    ax.plot([cx - radius + 0.1, cx - radius + 0.3], [cy + 0.9, cy + 1.1], color=get_color("Sclera", colors["Sclera"]), lw=4, zorder=3)
    ax.plot([cx - radius + 0.1, cx - radius + 0.3], [cy - 0.9, cy - 1.1], color=get_color("Sclera", colors["Sclera"]), lw=4, zorder=3)

    # Lens (Crystalline structure)
    # Use Ellipse to stretch width independently of height (0.4 thin -> 1.2 thick)
    from matplotlib.patches import Ellipse
    lens = Ellipse((cx - radius + 0.8, cy), lens_thickness * 2, 0.7 * 2, color=get_color("Lens", colors["Lens"]), 
                      fill=True, alpha=get_alpha("Lens", 0.5), zorder=4)
    ax.add_patch(lens)
    ax.add_patch(Ellipse((cx - radius + 0.8, cy), lens_thickness * 2, 0.7 * 2, color=get_color("Lens", colors["Lens"]), 
                      fill=False, lw=2, alpha=get_alpha("Lens", 0.9), zorder=4))

    # Iris (Colored part controlling pupil)
    # Dilation controls the gap. Total height is 0.6. Pupil gap is `dilation` value (0.2 narrow -> 1.5 wide)
    # We move the rectangles up/down based on dilation
    gap = pupil_dilation / 2.0
    iris_top = plt.Rectangle((cx - radius + 0.3, cy + gap), 0.15, 1.0 - gap, 
                             color=get_color("Iris", colors["Iris"]), zorder=4, alpha=get_alpha("Iris", 0.9))
    iris_bottom = plt.Rectangle((cx - radius + 0.3, cy - 1.0), 0.15, 1.0 - gap, 
                                color=get_color("Iris", colors["Iris"]), zorder=4, alpha=get_alpha("Iris", 0.9))
    ax.add_patch(iris_top)
    ax.add_patch(iris_bottom)

    # Draw incoming light beam
    # Match the incoming beam exactly to the new pupil gap
    beam_alpha = 0.3 if highlight_part != "None" else 0.6
    ax.fill_between([0, cx - radius + 0.45], [cy + gap, cy + gap], [cy - gap, cy - gap], 
                    color='#42f5ff', alpha=beam_alpha, zorder=1)
    # Focused beam converges from lens edges to macula
    ax.fill_between([cx - radius + 0.8, macula_x], [cy + 0.65, cy], [cy - 0.65, cy], 
                    color='#42f5ff', alpha=beam_alpha - 0.1, zorder=1)


    # Labels
    if show_labels:
        label_color = "#ffffff"
        font_size = 9
        
        def add_label(x, y, text, target_x, target_y):
            alpha = 1.0 if highlight_part == text or highlight_part == "None" else 0.3
            color = "#00f3ff" if highlight_part == text else label_color
            ax.text(x, y, text, color=color, fontsize=font_size, ha='center', va='center', alpha=alpha)
            ax.plot([x + (0.2 if x < cx else -0.2), target_x], [y, target_y], 
                    color='#ffffff', lw=0.5, alpha=alpha * 0.5, ls='--')

        add_label(2.0, 7.5, "Cornea", cx - radius + 0.1, cy + 0.5)
        add_label(2.5, 8.2, "Iris", cx - radius + 0.4, cy + 0.6)
        add_label(3.5, 8.8, "Lens", cx - radius + 0.8, cy + 0.5)
        add_label(6.5, 8.8, "Sclera", cx + 0.5, cy + radius)
        add_label(7.5, 8.2, "Choroid", cx + 1.5, cy + radius - 0.2)
        add_label(8.5, 7.5, "Retina", cx + 2.4, cy + radius - 1.2)
        add_label(9.0, 5.0, "Macula", macula_x, macula_y)
        add_label(9.0, 3.5, "Optic Nerve", nerve_x + 0.5, nerve_y1 - 0.1)

    ax.set_aspect('equal')
    ax.set_xlim(0, 10)
    ax.set_ylim(1, 9)
    ax.axis('off')
    
    return fig


def render_human_eye_scene(person_x, object_x, light_intensity, wavelength_nm, zoom_level):
    """2D game-like human + eye ray model with zoom region."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_facecolor('#0a1228')

    distance_m = abs(object_x - person_x)

    # Object (thing)
    obj_size = np.clip(0.18 * zoom_level, 0.08, 0.35)
    ax.add_patch(plt.Rectangle((object_x - obj_size / 2, 0.20), obj_size, obj_size, color='#42f5ff', alpha=0.9))
    ax.text(object_x, 0.47, 'Thing', color='white', ha='center', fontsize=10)

    # Human-like stick figure
    head_center = (person_x, 0.52)
    ax.add_patch(plt.Circle(head_center, 0.09, fill=False, lw=2.2, color='#ffd166'))
    ax.plot([person_x, person_x], [0.16, 0.43], color='#ffd166', lw=2.4)  # body
    ax.plot([person_x - 0.10, person_x + 0.10], [0.32, 0.28], color='#ffd166', lw=2.2)  # arms
    ax.plot([person_x, person_x - 0.09], [0.16, 0.02], color='#ffd166', lw=2.2)  # leg
    ax.plot([person_x, person_x + 0.10], [0.16, 0.02], color='#ffd166', lw=2.2)  # leg

    eye = (person_x + 0.03, 0.55)
    ax.scatter([eye[0]], [eye[1]], c='yellow', s=32, zorder=5)
    ax.text(person_x, 0.66, 'Human', color='white', ha='center', fontsize=10)

    # Ray color by wavelength
    if wavelength_nm < 500:
        ray_color = '#5cc8ff'
    elif wavelength_nm < 590:
        ray_color = '#74ffa1'
    else:
        ray_color = '#ffbf69'

    # Multiple rays from object edge points to eye
    ray_alpha = 0.25 + 0.7 * light_intensity
    for off in np.linspace(-obj_size / 2, obj_size / 2, 7):
        start = (object_x, 0.20 + obj_size / 2 + off * 0.7)
        ax.plot([start[0], eye[0]], [start[1], eye[1]], color=ray_color, alpha=ray_alpha, lw=1.4)

    # Eye zoom inset with retinal interpretation
    inset_x0 = person_x + 0.40
    ax.add_patch(plt.Circle((inset_x0, 0.42), 0.23, fill=False, lw=2.0, color='#cdb4db'))
    ax.plot([inset_x0 - 0.23, inset_x0 - 0.17], [0.42, 0.42], color='#cdb4db', lw=2)
    ax.plot([inset_x0 + 0.17, inset_x0 + 0.23], [0.42, 0.42], color='#cdb4db', lw=2)
    ax.plot([inset_x0 + 0.13, inset_x0 + 0.13], [0.25, 0.59], color='#ff4d6d', lw=2, alpha=0.8)  # retina

    retinal_scale = np.clip(0.12 * zoom_level / max(distance_m, 0.25), 0.03, 0.16)
    ax.add_patch(plt.Rectangle((inset_x0 + 0.11 - retinal_scale/2, 0.42 - retinal_scale/2), retinal_scale, retinal_scale,
                               color='#42f5ff', alpha=0.75))
    ax.text(inset_x0, 0.69, 'Eye Zoom View', color='white', ha='center', fontsize=9)

    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.05, 0.9)
    ax.set_xlabel('Scene Axis (drag person/object by sliders)')
    ax.set_yticks([])
    ax.set_title('Human Vision Interaction: distance + light intensity + eye zoom')

    return fig, distance_m


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


# Navigation State
if "active_module" not in st.session_state:
    st.session_state.active_module = "1.1 Visual Perception"

def nav_button(label, module_id):
    is_active = st.session_state.active_module == module_id
    if st.button(label, type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.active_module = module_id
        st.rerun()

# --- GLOBAL IMAGE OVERRIDE (Moved Up) ---
st.sidebar.markdown("### 🖼️ Global Signal Override")
st.sidebar.caption("Upload an image here to override procedurally generated signals across all compatible modules.")
global_upload = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "bmp", "tif"], label_visibility="collapsed")

global_img = None
if global_upload:
    file_bytes = np.asarray(bytearray(global_upload.read()), dtype=np.uint8)
    global_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if global_img is not None:
        st.sidebar.success("Global Signal Locked")
        
st.sidebar.divider()

# --- TRACK SELECTION ---
category = st.sidebar.radio("🎯 Choose Track", ["1. Fundamentals", "2. Advanced Processing"])
st.sidebar.caption("Tip: Start from Fundamentals before moving to Advanced modules.")

if category == "1. Fundamentals":
    with st.sidebar.expander("📘 Chapter 1: Fundamentals of Image Formation", expanded=False):
        nav_button("1.1 Visual Perception", "1.1 Visual Perception")
        st.caption("• Human eye structure  \n• Image formation in retina  \n• Brightness adaptation & contrast  \n• Rods and cones")
        st.write("")
        nav_button("1.2 Electromagnetic (EM) Spectrum", "1.2 Electromagnetic (EM) Spectrum")
        st.caption("• Visible light range (400nm-700nm)  \n• Infrared, X-rays, UV  \n• Image sensing across spectrum")
        st.write("")
        nav_button("1.3 Image Acquisition", "1.3 Image Acquisition")
        st.caption("• Sensors (CCD, CMOS)  \n• Image capture process  \n• Analog to digital conversion  \n• Basic imaging system model")
        
    with st.sidebar.expander("📘 Chapter 2: Image Digitization", expanded=False):
        nav_button("2.1 Sampling & Quantization", "2.1 Sampling & Quantization")
        st.caption("• Continuous to digital conversion  \n• Spatial sampling  \n• Gray level quantization  \n• Resolution (spatial & intensity)")
        st.write("")
        nav_button("2.2 Pixel Connectivity", "2.2 Pixel Connectivity")
        st.caption("• 4-connectivity  \n• 8-connectivity  \n• m-connectivity")
        st.write("")
        nav_button("2.3 Distance Measures", "2.3 Distance Measures")
        st.caption("• Euclidean distance  \n• City block distance  \n• Chessboard distance")
        
    with st.sidebar.expander("📘 Chapter 3: Image Representation & Mathematical Tools", expanded=False):
        nav_button("3.1 Mathematical Tools", "3.1 Mathematical Tools")
        st.caption("• Set theory basics  \n• Matrix representation  \n• Logical operations  \n• Basic algebra used in DIP")
        st.write("")
        nav_button("3.2 Connected Components", "3.2 Connected Components")
        st.caption("• Region labeling  \n• Object detection  \n• Component extraction")
        st.write("")
        nav_button("3.3 Image Statistics", "3.3 Image Statistics")
        st.caption("• Mean  \n• Variance  \n• Histogram  \n• Probability density functions")
        
    with st.sidebar.expander("📘 Chapter 4: Advanced Vision Concepts", expanded=False):
        nav_button("4.1 3D Eye Vision Game Model", "4.1 3D Eye Vision Game Model")
        st.caption("• Stereo vision  \n• Depth perception  \n• Parallax  \n• 3D reconstruction basics  \n• Applications in AR/VR & gaming")

else:
    with st.sidebar.expander("📘 Chapter 5: Spatial Domain Processing", expanded=False):
        nav_button("5.1 Spatial Filtering", "5.1 Spatial Filtering")
        st.caption("**🔹 Concept:** Operations performed directly on pixels.  \n**🔹 Topics Included:** Image smoothing (Mean filter, Gaussian filter), Image sharpening (Laplacian, High-pass filter), Convolution & correlation, Mask/kernel operations  \n**🔹 Formula:** $g(x,y)=T[f(x,y)]$  \n**🔹 Used For:** Noise removal, Edge enhancement, Image improvement")
        
    with st.sidebar.expander("📘 Chapter 6: Frequency Domain Processing", expanded=False):
        nav_button("6.1 Frequency Domain", "6.1 Frequency Domain")
        st.caption("**🔹 Concept:** Processing image after converting it using Fourier Transform.  \n**🔹 Steps:** Apply DFT (Discrete Fourier Transform), Modify frequency components, Apply Inverse DFT  \n**🔹 Filters:** Low Pass Filter (Blur), High Pass Filter (Edge detection), Band Pass Filter  \n**🔹 Advantage:** Better for: Periodic noise removal, Global enhancement")
        
    with st.sidebar.expander("📘 Chapter 7: Morphological Image Processing", expanded=False):
        nav_button("7.1 Morphology", "7.1 Morphology")
        st.caption("**🔹 Concept:** Shape-based image processing (mainly binary images)  \n**🔹 Basic Operations:** Erosion, Dilation, Opening, Closing  \n**🔹 Uses:** Object boundary extraction, Removing small noise, Filling holes, Shape detection")

# Set module alias for backend logic compatibility
module = st.session_state.active_module

# ==========================================
# PART 1: FUNDAMENTALS (Logic from app (2).py)
# ==========================================

if module == "1.1 Visual Perception":
    st.header("1.1 Elements of Visual Perception")
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
        if global_img is not None:
            # Scale global image by illumination purely for demonstration
            img_scaled = np.clip(global_img.astype(np.float32) * illum_lin * 1.5, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2BGR)
        else:
            img = generate_flower_scene(illum_lin, mach_bands)
        st.image(img, channels="BGR", caption="Simulated Scene", use_container_width=True)

elif module == "1.2 Electromagnetic (EM) Spectrum":
    st.header("1.2 Light & The Electromagnetic Spectrum")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.subheader("Command Center")
        # Log Freq: 6 (Radio) to 22 (Gamma)
        log_freq = st.slider("Frequency (Log Hz)", 6.0, 22.0, 14.5, 0.1)
        freq = 10**log_freq
        wavelength = 3e8 / freq
        energy = 4.135e-15 * freq
        
    with col2:
        st.subheader("Physics Engine")
        st.latex(r"\nu = " + f"{freq:.2e} Hz")
        st.latex(r"\lambda = c / \nu = " + f"{wavelength:.2e} m")
        st.latex(r"E = h\nu = " + f"{energy:.2e} eV")
        
    with col3:
        st.subheader("Band Analysis")
        if log_freq < 9:
            st.success("Target Band: **Radio Waves**")
            st.info("📡 Application: Broadcasting")
        elif log_freq < 12:
            st.success("Target Band: **Microwaves**")
            st.info("📶 Application: Radar & WiFi")
        elif log_freq < 14.5:
            st.warning("Target Band: **Infrared**")
            st.markdown("🔥 **Thermal Signature Detected**")
        elif log_freq < 15.0:
            st.error("Target Band: **Visible Light**")
            st.markdown("🌈 **Human Visual Window Active**")
        elif log_freq < 17:
            st.info("Target Band: **Ultraviolet**")
            st.markdown("☀️ Radiation: Fluorescence")
        elif log_freq < 20:
            st.info("Target Band: **X-Rays**")
            st.markdown("🦴 **Penetrating Scan Active**")
        else:
            st.error("Target Band: **Gamma Rays**")
            st.markdown("☢️ **Critical Nuclear Radiation**")


    
elif module == "1.3 Image Acquisition":
    st.header("1.3 Image Sensing & Acquisition")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Store state so we can render across columns cleanly
    if "capture_mode" not in st.session_state:
        st.session_state.capture_mode = "Single Sensor (Point)"
    
    with col1:
        st.subheader("Command Center")
        st.session_state.capture_mode = st.radio("Sensor Topology", [
            "Single Sensor (Point)", 
            "Sensor Strip (Line)", 
            "Sensor Array (Flat/Instant)"
        ])
        
        trigger = st.button("INITIATE CAPTURE SEQUENCE", use_container_width=True)
    
    with col2:
        st.subheader("Viewport Alpha")
        image_placeholder = st.empty()
        
        if trigger:
            GRID_SIZE = 20
            if global_img is not None:
                img = cv2.resize(global_img, (GRID_SIZE, GRID_SIZE))
            else:
                img = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            
            if st.session_state.capture_mode == "Single Sensor (Point)":
                for y in range(GRID_SIZE):
                    for x in range(GRID_SIZE):
                        img[y, x] = int((x+y) * 255 / (GRID_SIZE*2))
                    if y % 2 == 0:
                        big_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
                        image_placeholder.image(big_img, caption=f"Scanning Row {y+1}/{GRID_SIZE}...", use_container_width=False, clamp=True)
                        time.sleep(0.05)
            
            elif st.session_state.capture_mode == "Sensor Strip (Line)":
                for y in range(GRID_SIZE):
                    for x in range(GRID_SIZE):
                        img[y, x] = int((x+y) * 255 / (GRID_SIZE*2))
                    big_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
                    image_placeholder.image(big_img, caption=f"Scanning Row {y+1}/{GRID_SIZE}...", use_container_width=False, clamp=True)
                    time.sleep(0.1)
            
            elif st.session_state.capture_mode == "Sensor Array (Flat/Instant)":
                with st.spinner("Flash..."):
                    time.sleep(0.2)
                for y in range(GRID_SIZE):
                    for x in range(GRID_SIZE):
                        img[y, x] = int((x+y) * 255 / (GRID_SIZE*2))
            
            big_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
            image_placeholder.image(big_img, caption="Capture Complete", use_container_width=False, clamp=True)
            
        else:
            st.info("Awaiting Capture Trigger...")

    with col3:
        st.subheader("System Telemetry")
        if st.session_state.capture_mode == "Single Sensor (Point)":
            st.info("Scanner Type: **Microdensitometer**")
            st.metric("Acquisition Speed", "Slow")
            st.metric("Mechanics", "2-Axis Motion (X-Y)")
        elif st.session_state.capture_mode == "Sensor Strip (Line)":
            st.info("Scanner Type: **Flatbed / Drum**")
            st.metric("Acquisition Speed", "Medium")
            st.metric("Mechanics", "1-Axis Motion")
        else:
            st.info("Scanner Type: **Digital Camera / CCD App**")
            st.metric("Acquisition Speed", "Instant")
            st.metric("Mechanics", "Solid State (No Moving Parts)")

elif module == "2.1 Sampling & Quantization":
    sampling_quantization_extended.run(global_img)


elif module == "2.2 Pixel Connectivity":
    pixel_relationships.run(global_img)


elif module == "3.1 Mathematical Tools":
    st.header("3.1 Mathematical Functions in Digital Image Processing")
    st.caption("Analyze arithmetic, statistical, trigonometric, transform, morphology, and filtering functions with visual demos.")

    with st.expander("📘 Quick Analysis: Where each function is used in DIP", expanded=False):
        st.markdown("""
        - **Arithmetic** (`+`, `-`, `×`, `÷`, `log`, `exp`, `gamma`) → brightness/contrast and dynamic range control.
        - **Statistical** (mean, median, mode, variance, std, histogram, PDF) → denoising decisions and segmentation thresholds.
        - **Trigonometric** (`sin`, `cos`, `tan`) → signal modeling and Fourier basis understanding.
        - **Transforms** (DFT, FFT, DCT, Wavelet) → frequency analysis, compression and multi-resolution interpretation.
        - **Morphology** (dilation, erosion, opening, closing) → binary shape cleanup and object structure analysis.
        - **Filtering** (convolution, correlation, Gaussian, Laplacian, Sobel) → smoothing, sharpening and edge extraction.
        """)

    if global_img is not None:
        base = cv2.resize(global_img, (256, 256))
    else:
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

elif module == "2.3 Distance Measures":
    distance_measures.run(global_img)

elif module == "3.2 Connected Components":
    connected_components.run(global_img)

elif module == "3.3 Image Statistics":
    image_statistics.run(global_img)



elif module == "4.1 3D Eye Vision Game Model":
    st.header("4.1 Human Vision Game Model (Interactive)")
    st.write("Explore the anatomical structure of the eye and simulate visual perception based on physics.")

    tab_anatomy, tab_physics = st.tabs(["👁️ Anatomical Cross-Section", "🌌 Vision Physics Engine"])

    with tab_anatomy:
        st.subheader("Interactive Human Eye Anatomy")
        ac1, ac2 = st.columns([1, 2])
        
        with ac1:
            st.info("Biological Components")
            show_labels = st.checkbox("Show Anatomical Labels", value=True)
            highlight_part = st.selectbox(
                "Isolate Structure", 
                ["None", "Cornea", "Iris", "Lens", "Retina", "Macula", "Optic Nerve", "Sclera", "Choroid"]
            )
            
            st.markdown("### Physiological Controls")
            pupil_dilation = st.slider("Pupil Dilation (Iris Gap)", 0.2, 1.5, 0.7, 0.1, 
                                      help="Simulates bright light (constricted) vs dark/sympathetic nervous system (dilated)")
            lens_thickness = st.slider("Lens Thickness (Focus)", 0.4, 1.2, 0.7, 0.1,
                                      help="Simulates accommodation: thin for far objects, thick for near objects")
            cornea_bulge = st.slider("Cornea Bulge (Astigmatism)", 0.8, 1.6, 1.2, 0.1,
                                    help="Alters the primary refractive surface of the eye")
            
        with ac2:
            fig_anatomy = render_anatomical_eye_scene(highlight_part, show_labels, pupil_dilation, lens_thickness, cornea_bulge)
            st.pyplot(fig_anatomy)
            
        st.markdown("---")
        st.subheader("Simulated Retinal Vision")
        st.markdown("Observe how the physical shape of the eye directly impacts the image formed on the retina. Distorting the lens or cornea creates **defocus** and **astigmatism**.")
        
        in_img, out_img = simulate_physiological_vision(global_img, pupil_dilation, lens_thickness, cornea_bulge)
        
        vc1, vc2 = st.columns(2)
        with vc1:
            st.image(in_img, caption="External Object (Input)", use_container_width=True, clamp=True)
        with vc2:
            st.image(out_img, caption="Retinal Projection (Output)", use_container_width=True, clamp=True)

    with tab_physics:
        st.subheader("Ray Tracing & Perception Simulator")
        c1, c2, c3 = st.columns(3)
        with c1:
            person_x = st.slider("Drag Human Position", 0.8, 8.5, 5.0, 0.1)
        with c2:
            object_x = st.slider("Drag Thing Position", 0.5, 8.8, 2.0, 0.1)
        with c3:
            zoom_level = st.slider("Eye Zoom", 0.6, 2.5, 1.4, 0.1)

        c4, c5 = st.columns(2)
        with c4:
            light_intensity = st.slider("Light Ray Intensity", 0.1, 1.0, 0.75, 0.05)
        with c5:
            wavelength_nm = st.slider("Wavelength (nm)", 420, 700, 550, 5)

        scene_fig, distance_m = render_human_eye_scene(person_x, object_x, light_intensity, wavelength_nm, zoom_level)
        st.pyplot(scene_fig)

        input_img, perceived, processed = generate_processed_vision_image(distance_m, light_intensity, zoom_level, global_img)
        
        st.markdown("### Vision Pipeline")
        i1, i2, i3 = st.columns(3)
        with i1:
            st.image(input_img, caption="1. Input Image (Object)", use_container_width=True, clamp=True)
        with i2:
            st.image(perceived, caption="2. Perceived (Retina)", use_container_width=True, clamp=True)
        with i3:
            st.image(processed, caption="3. Processed (Brain/Digital)", use_container_width=True, clamp=True)

        st.metric("Distance Between Human and Thing", f"{distance_m:.2f} m")
        st.markdown('''
        **Input controls:** distance change (human vs thing position) and light ray intensity.  
        **Output:** processed image of the thing after simulated vision + digital enhancement.
        ''')

# ==========================================
# PART 2: ADVANCED PROCESSING (Logic from app.py - Workbench)
# ==========================================
elif category == "2. Advanced Processing":
    if global_img is not None:
        original_img = global_img
    else:
        # Provide a default image if none uploaded for advanced processing
        original_img = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(original_img, (150, 150), 100, 255, -1)
        st.warning("No Global Image Uploaded. Using default test shape.")

    if True:
        
        if module == "6.1 Frequency Domain":
            st.header("6.1 Frequency Domain Filtering")
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
            
        elif module == "5.1 Spatial Filtering":
            st.header("5.1 Spatial Filtering")
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

        elif module == "7.1 Morphology":
            st.header("7.1 Morphological Operations")
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



