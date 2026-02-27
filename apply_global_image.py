import os
import sys

def process_app_py():
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add global uploader to sidebar
    sidebar_insert = '''
st.sidebar.divider()
st.sidebar.caption("Tip: Start from Fundamentals before moving to Advanced modules.")

# --- GLOBAL IMAGE OVERRIDE ---
st.sidebar.markdown("### 🖼️ Global Signal Override")
st.sidebar.caption("Upload an image here to override procedurally generated signals across all compatible modules.")
global_upload = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "bmp", "tif"], label_visibility="collapsed")

global_img = None
if global_upload:
    file_bytes = np.asarray(bytearray(global_upload.read()), dtype=np.uint8)
    global_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if global_img is not None:
        st.sidebar.success("Global Signal Locked")
'''
    if "# --- GLOBAL IMAGE OVERRIDE ---" not in content:
        content = content.replace(
            'st.sidebar.caption("Tip: Start from Fundamentals before moving to Advanced modules.")\n',
            sidebar_insert
        )

    # 2. Update 1.1 module to use global_img
    mod1_1_old = '''    with col2:
        st.subheader("Simulation")
        img = generate_flower_scene(illum_lin, mach_bands)'''
    mod1_1_new = '''    with col2:
        st.subheader("Simulation")
        if global_img is not None:
            # Scale global image by illumination purely for demonstration
            img_scaled = np.clip(global_img.astype(np.float32) * illum_lin * 1.5, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2BGR)
        else:
            img = generate_flower_scene(illum_lin, mach_bands)'''
    content = content.replace(mod1_1_old, mod1_1_new)

    # 3. Update 1.3 module to use global_img
    mod1_3_old = '''        if trigger:
            GRID_SIZE = 20
            img = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)'''
    mod1_3_new = '''        if trigger:
            GRID_SIZE = 20
            if global_img is not None:
                img = cv2.resize(global_img, (GRID_SIZE, GRID_SIZE))
            else:
                img = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)'''
    content = content.replace(mod1_3_old, mod1_3_new)

    # 4. Update 1.6 module to use global_img
    mod1_6_old = r'''        """)

    base = np.linspace(0, 255, 256).reshape(1, 256).repeat(256, 0).astype(np.uint8)'''
    mod1_6_new = r'''        """)

    if global_img is not None:
        base = cv2.resize(global_img, (256, 256))
    else:
        base = np.linspace(0, 255, 256).reshape(1, 256).repeat(256, 0).astype(np.uint8)'''
    content = content.replace(mod1_6_old, mod1_6_new)

    # 5. Update 1.10 module to use global_img
    mod1_10_old = '''def generate_processed_vision_image(distance_m, light_intensity, zoom_level):
    """Simulate perceived/processed image based on distance and illumination."""
    h, w = 320, 320
    img = np.zeros((h, w), dtype=np.uint8)

    # Simple "thing" object: circle + cross detail
    size = int(np.clip(70 * zoom_level * (1.5 / max(distance_m, 0.3)), 18, 120))
    center = (w // 2, h // 2)
    cv2.circle(img, center, size, 190, -1)
    cv2.line(img, (center[0] - size, center[1]), (center[0] + size, center[1]), 240, 3)
    cv2.line(img, (center[0], center[1] - size), (center[0], center[1] + size), 240, 3)'''
    
    mod1_10_new = '''def generate_processed_vision_image(distance_m, light_intensity, zoom_level, global_img=None):
    """Simulate perceived/processed image based on distance and illumination."""
    h, w = 320, 320
    
    if global_img is not None:
        # Use global image, scale and zoom it based on distance
        img = cv2.resize(global_img, (w, h))
        scale = np.clip(zoom_level * (1.5 / max(distance_m, 0.3)), 0.2, 5.0)
        M = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
        img = cv2.warpAffine(img, M, (w, h))
    else:
        img = np.zeros((h, w), dtype=np.uint8)
        size = int(np.clip(70 * zoom_level * (1.5 / max(distance_m, 0.3)), 18, 120))
        center = (w // 2, h // 2)
        cv2.circle(img, center, size, 190, -1)
        cv2.line(img, (center[0] - size, center[1]), (center[0] + size, center[1]), 240, 3)
        cv2.line(img, (center[0], center[1] - size), (center[0], center[1] + size), 240, 3)'''
    content = content.replace(mod1_10_old, mod1_10_new)

    # Pass global_img to 1.10 function call
    content = content.replace(
        "perceived, processed = generate_processed_vision_image(distance_m, light_intensity, zoom_level)",
        "perceived, processed = generate_processed_vision_image(distance_m, light_intensity, zoom_level, global_img)"
    )

    # 6. Update external module calls (chapter 2)
    content = content.replace("sampling_quantization_extended.run()", "sampling_quantization_extended.run(global_img)")
    content = content.replace("pixel_relationships.run()", "pixel_relationships.run(global_img)")
    content = content.replace("distance_measures.run()", "distance_measures.run(global_img)")
    content = content.replace("connected_components.run()", "connected_components.run(global_img)")
    content = content.replace("image_statistics.run()", "image_statistics.run(global_img)")

    # 7. advanced section removed file uploader
    adv_old = '''# ==========================================
elif category == "2. Advanced Processing":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg", "bmp", "tif"])

    if uploaded_file:
        original_img = load_image(uploaded_file)'''
    adv_new = '''# ==========================================
elif category == "2. Advanced Processing":
    if global_img is not None:
        original_img = global_img
    else:
        # Provide a default image if none uploaded for advanced processing
        original_img = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(original_img, (150, 150), 100, 255, -1)
        st.warning("No Global Image Uploaded. Using default test shape.")

    if True:'''
    content = content.replace(adv_old, adv_new)

    with open("app.py", "w", encoding="utf-8") as f:
        f.write(content)


def process_module(filepath, has_uploader=True):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Change signature
    content = content.replace("def run():", "def run(global_img=None):")
    
    if has_uploader:
        # Replace the local uploader blocks with global_img usage
        if "sampling_quantization" in filepath:
            old_block = '''        use_uploaded = st.checkbox("Override Source Signal")
        if use_uploaded:
            uploaded_file = st.file_uploader("Upload Target", type=['png', 'jpg'])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (256, 256))
            else:
                img = np.zeros((256, 256), dtype=np.uint8)
        else:
            img = np.zeros((256, 256), dtype=np.uint8)
            for i in range(256):
                img[:, i] = i'''
            new_block = '''        if global_img is not None:
            img = cv2.resize(global_img, (256, 256))
        else:
            img = np.zeros((256, 256), dtype=np.uint8)
            for i in range(256):
                img[:, i] = i'''
            content = content.replace(old_block, new_block)
            
        elif "image_statistics" in filepath:
            old_block = '''        # Source Image Selection
        uploaded_file = st.file_uploader("Override Source Signal", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        else:
            img = np.zeros((200, 200), dtype=np.uint8)
            for i in range(200):
                img[i, :] = i
            cv2.rectangle(img, (50, 50), (150, 150), (100), -1)'''
            new_block = '''        if global_img is not None:
            img = cv2.resize(global_img, (200, 200))
        else:
            img = np.zeros((200, 200), dtype=np.uint8)
            for i in range(200):
                img[i, :] = i
            cv2.rectangle(img, (50, 50), (150, 150), (100), -1)'''
            content = content.replace(old_block, new_block)
            
        elif "connected_components" in filepath:
            old_block = '''        uploaded_file = st.file_uploader("Override Binary Source", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            binary_image = (binary_image / 255).astype(np.uint8)
        else:
            binary_image = np.zeros((20, 20), dtype=np.uint8)
            binary_image[2:6, 2:6] = 1
            binary_image[10:15, 10:15] = 1
            binary_image[10:18, 2:5] = 1
            binary_image[2:5, 12:18] = 1'''
            new_block = '''        if global_img is not None:
            img = cv2.resize(global_img, (100, 100)) # Small for fast CCL
            _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            binary_image = (binary_image / 255).astype(np.uint8)
        else:
            binary_image = np.zeros((20, 20), dtype=np.uint8)
            binary_image[2:6, 2:6] = 1
            binary_image[10:15, 10:15] = 1
            binary_image[10:18, 2:5] = 1
            binary_image[2:5, 12:18] = 1'''
            content = content.replace(old_block, new_block)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    base_dir = "d:\\DIP Another GUI"
    process_app_py()
    process_module(os.path.join(base_dir, "chapter2_experiments", "sampling_quantization_extended.py"), True)
    process_module(os.path.join(base_dir, "chapter2_experiments", "image_statistics.py"), True)
    process_module(os.path.join(base_dir, "chapter2_experiments", "connected_components.py"), True)
    process_module(os.path.join(base_dir, "chapter2_experiments", "pixel_relationships.py"), False)
    process_module(os.path.join(base_dir, "chapter2_experiments", "distance_measures.py"), False)
    print("Injection complete!")
