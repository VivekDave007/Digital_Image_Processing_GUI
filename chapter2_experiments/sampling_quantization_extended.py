import streamlit as st
import numpy as np
import cv2

def compute_mse_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 0, float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return mse, psnr

def run(global_img=None):
    st.markdown("<h1>1.10 Sampling & Quantization</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1], gap="small")
    
    if "sq_mode" not in st.session_state:
        st.session_state.sq_mode = "Spatial Sampling"

    with col1:
        st.subheader("Command Center")
        
        # Source Image
        use_uploaded = st.checkbox("Override Source Signal")
        if use_uploaded:
            uploaded_file = st.file_uploader("Upload Target", type=['png', 'jpg'])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (256, 256))
            else:
                img = np.zeros((256, 256), dtype=np.uint8)
        else:
            H, W = 256, 256
            img = np.zeros((H, W), dtype=np.uint8)
            Y, X = np.ogrid[:H, :W]
            center = (H//2, W//2)
            dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)
            img = 255 * (1 - dist_from_center / (np.sqrt(2)*128))
            img = np.clip(img, 0, 255).astype(np.uint8)

        st.session_state.sq_mode = st.radio("Operation Matrix", ["Spatial Sampling", "Intensity Quantization"])
        
        output_img = img.copy()
        
        if st.session_state.sq_mode == "Spatial Sampling":
            st.info("Reduces spatial resolution (Downsampling).")
            ratio = st.select_slider("Sampling Scale", options=[0.5, 0.25, 0.125, 0.05])
            
            H, W = img.shape
            new_w, new_h = max(1, int(W * ratio)), max(1, int(H * ratio))
            
            sampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            output_img = cv2.resize(sampled, (W, H), interpolation=cv2.INTER_NEAREST)
            
        elif st.session_state.sq_mode == "Intensity Quantization":
            st.info("Reduces the number of gray levels (Bit Depth).")
            levels = st.select_slider("Quantization Levels", options=[128, 64, 32, 16, 8, 4, 2])
            
            div = 256 / levels
            quantized = np.floor(img / div) * (255 / (levels - 1))
            output_img = np.uint8(quantized)

    with col2:
        st.subheader("Viewport Alpha")
        c1, c2 = st.columns(2)
        c1.image(img, caption="Original Signal (256x256, 8-bit)", use_container_width=True, clamp=True)
        
        if st.session_state.sq_mode == "Spatial Sampling":
            c2.image(output_img, caption=f"Lossy Output (Scaled {ratio}x)", use_container_width=True, clamp=True)
        else:
            c2.image(output_img, caption=f"Lossy Output ({levels} Levels)", use_container_width=True, clamp=True)

    with col3:
        st.subheader("System Telemetry")
        
        mse, psnr = compute_mse_psnr(img, output_img)
        
        st.metric("MSE (Error)", f"{mse:.2f}")
        st.metric("PSNR (Quality)", f"{psnr:.2f} dB" if psnr != float('inf') else "Perfect")
        
        st.markdown("---")
        if st.session_state.sq_mode == "Spatial Sampling":
            st.warning("**Aliasing Warning**: Severe downsampling creates blocking artifacts and loses high-frequency details.")
            st.metric("Sensor Array", f"{new_w} x {new_h}")
        else:
            st.warning("**False Contouring**: Low bit-depths create artificial banding in smooth gradients.")
            bits = int(np.log2(levels)) if levels > 0 else 0
            st.metric("Bit Depth", f"{bits}-bit / pixel")
