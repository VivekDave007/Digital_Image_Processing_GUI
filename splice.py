import sys

def main():
    try:
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        # 1. Insert helper functions before get_spectrum
        helper_funcs = '''
def generate_processed_vision_image(distance_m, light_intensity, zoom_level):
    """Simulate perceived/processed image based on distance and illumination."""
    h, w = 320, 320
    img = np.zeros((h, w), dtype=np.uint8)

    # Simple "thing" object: circle + cross detail
    size = int(np.clip(70 * zoom_level * (1.5 / max(distance_m, 0.3)), 18, 120))
    center = (w // 2, h // 2)
    cv2.circle(img, center, size, 190, -1)
    cv2.line(img, (center[0] - size, center[1]), (center[0] + size, center[1]), 240, 3)
    cv2.line(img, (center[0], center[1] - size), (center[0], center[1] + size), 240, 3)

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

    return img, np.clip(enhanced, 0, 255).astype(np.uint8)


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

'''
        if "def render_human_eye_scene" not in content:
            content = content.replace("# --- Frequency Domain Helpers ---", helper_funcs + "\n# --- Frequency Domain Helpers ---")
        else:
            print("Helpers already inserted")
            
        # 2. Add '1.10 3D Eye Vision Game Model' to the module list
        sidebar_old = """
        "1.6 Math Tools",
        "1.7 Distance Measures",
        "1.8 Connected Components",
        "1.9 Image Statistics"
    ])"""
        sidebar_new = """
        "1.6 Math Tools",
        "1.7 Distance Measures",
        "1.8 Connected Components",
        "1.9 Image Statistics",
        "1.10 3D Eye Vision Game Model"
    ])"""
        if "1.10 3D Eye Vision Game Model" not in content:
             if sidebar_old in content:
                 content = content.replace(sidebar_old, sidebar_new)
             else:
                 print("Could not find sidebar block to replace.")
        else:
            print("1.10 already in sidebar.")
             
        # 3. Add 1.10 module block before "elif category == "2. Advanced Processing":"
        
        module_1_10_code = """
elif module == "1.10 3D Eye Vision Game Model":
    st.header("10. Human Vision Game Model (Interactive)")
    st.write("Move (drag via sliders) the human and thing positions, zoom into eye vision, and observe processed output image with distance/light-ray changes.")

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

    perceived, processed = generate_processed_vision_image(distance_m, light_intensity, zoom_level)
    display_images(perceived, processed, ("Perceived Image (retina-like)", "Processed Image (digital enhancement)"))

    st.metric("Distance Between Human and Thing", f"{distance_m:.2f} m")
    st.markdown('''
    **Input controls:** distance change (human vs thing position) and light ray intensity.  
    **Output:** processed image of the thing after simulated vision + digital enhancement.
    ''')

"""
        adv_processing = "# ==========================================\n# PART 2: ADVANCED PROCESSING (Logic from app.py - Workbench)\n# =========================================="
        if 'elif module == "1.10 3D Eye Vision Game Model":' not in content:
            if adv_processing in content:
                content = content.replace(adv_processing, module_1_10_code + adv_processing)
            else:
                 print("Could not find Advanced Processing separator block.")
        else:
            print("1.10 module already present.")
            
        # 4. Update the 1.6 Math tools block to add the st.expander
        math_tools_old = '''elif module == "1.6 Math Tools":
    st.header("6. Mathematical Functions in Digital Image Processing")
    st.caption("Analyze arithmetic, statistical, trigonometric, transform, morphology, and filtering functions with visual demos.")

    base = np.linspace(0'''
        
        math_tools_new = '''elif module == "1.6 Math Tools":
    st.header("6. Mathematical Functions in Digital Image Processing")
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

    base = np.linspace(0'''
        if "📘 Quick Analysis: Where each function is used in DIP" not in content:
            if math_tools_old in content:
                content = content.replace(math_tools_old, math_tools_new)
            else:
                 print("Could not find Math Tools block to update.")
        else:
             print("Math Tools already updated.")
            
        with open("app.py", "w", encoding="utf-8") as f:
            f.write(content)
        print("Success! Splice Operation Completed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
