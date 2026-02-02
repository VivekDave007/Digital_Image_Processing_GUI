import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_mse_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 0, float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return mse, psnr

def sampling_demo(image):
    st.subheader("Spatial Sampling (Downsampling)")
    H, W = image.shape
    ratios = [0.5, 0.25, 0.125]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.image(image, caption=f"Original ({W}x{H})", clamp=True, use_container_width=True)
    
    cols = [col2, col3, col4]
    
    for i, r in enumerate(ratios):
        new_w, new_h = int(W * r), int(H * r)
        sampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        upscaled = cv2.resize(sampled, (W, H), interpolation=cv2.INTER_NEAREST)
        
        mse, psnr = compute_mse_psnr(image, upscaled)
        
        cols[i].image(upscaled, caption=f"Scale {r} ({new_w}x{new_h})\nPSNR: {psnr:.2f}dB", clamp=True, use_container_width=True)

def quantization_demo(image):
    st.subheader("Gray-Level Quantization")
    levels_list = [128, 32, 8, 4]
    
    st.image(image, caption="Original (256 levels)", width=200)
    
    st.write("Results:")
    cols = st.columns(4)
    
    for i, levels in enumerate(levels_list):
        div = 256 / levels
        quantized = np.floor(image / div) * (255 / (levels - 1))
        quantized = np.uint8(quantized)
        
        mse, psnr = compute_mse_psnr(image, quantized)
        
        cols[i].image(quantized, caption=f"Levels: {levels}\nPSNR: {psnr:.2f}dB", clamp=True, use_container_width=True)

def run():
    st.header("Topic C: Sampling & Quantization")
    
    # Generate Synthetic Gradient
    H, W = 256, 256
    img = np.zeros((H, W), dtype=np.uint8)
    Y, X = np.ogrid[:H, :W]
    center = (H//2, W//2)
    dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)
    img = 255 * (1 - dist_from_center / (np.sqrt(2)*128))
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    use_uploaded = st.checkbox("Use Uploaded Image instead of Synthetic?")
    if use_uploaded:
        uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg'])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256)) # Normalize size for demo
    
    sampling_demo(img)
    st.divider()
    quantization_demo(img)
