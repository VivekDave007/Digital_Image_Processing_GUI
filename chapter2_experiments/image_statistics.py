import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_basic_statistics(image):
    st.markdown("#### Basic Statistics")
    mean_val = np.mean(image)
    var_val = np.var(image)
    std_val = np.std(image)
    min_val = np.min(image)
    max_val = np.max(image)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean", f"{mean_val:.2f}")
    col2.metric("Variance", f"{var_val:.2f}")
    col3.metric("Std Dev", f"{std_val:.2f}")
    col4.metric("Min", f"{min_val}")
    col5.metric("Max", f"{max_val}")
    
    # Histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(hist, color='black')
    ax.set_title("Grayscale Histogram")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    ax.set_xlim([0, 256])
    st.pyplot(fig)

def contrast_stretching(image):
    st.markdown("#### Contrast Stretching")
    min_val = np.min(image)
    max_val = np.max(image)
    
    st.write(f"Original Range: [{min_val}, {max_val}]")
    
    if max_val - min_val == 0:
        st.warning("Image has constant value, cannot stretch contrast.")
        return image
        
    stretched = 255.0 * (image - min_val) / (max_val - min_val)
    stretched = np.uint8(stretched)
    
    display_cols = st.columns(2)
    with display_cols[0]:
        st.image(image, caption="Original", clamp=True, use_container_width=True)
    with display_cols[1]:
        st.image(stretched, caption="Contrast Stretched", clamp=True, use_container_width=True)
        
    return stretched

def add_noise(image, noise_type="Gaussian"):
    noisy_image = image.copy()
    
    if noise_type == "Gaussian":
        mean = st.slider("Mean", -50.0, 50.0, 0.0)
        sigma = st.slider("Sigma", 0.0, 100.0, 25.0)
        gauss = np.random.normal(mean, sigma, image.shape)
        noisy_image = image.astype(np.float32) + gauss
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
    elif noise_type == "Salt & Pepper":
        prob = st.slider("Noise Probability", 0.0, 0.5, 0.05)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = np.random.random()
                if rdn < prob:
                    noisy_image[i][j] = 0
                elif rdn > thres:
                    noisy_image[i][j] = 255
                    
    return noisy_image

def run():
    st.header("Topic D: Image Statistics")
    
    # Allow user to upload or use synthetic
    uploaded_file = st.file_uploader("Upload Image (Optional)", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    else:
        st.info("Using synthetic gradient image.")
        img = np.zeros((200, 200), dtype=np.uint8)
        for i in range(200):
            img[i, :] = i
        cv2.rectangle(img, (50, 50), (150, 150), (100), -1)

    st.image(img, caption="Input Image", use_container_width=True)
    
    tab1, tab2, tab3 = st.tabs(["Statistics", "Contrast", "Noise"])
    
    with tab1:
        compute_basic_statistics(img)
        
    with tab2:
        contrast_stretching(img)
        
    with tab3:
        noise_type = st.selectbox("Noise Type", ["Gaussian", "Salt & Pepper"])
        noisy = add_noise(img, noise_type)
        st.image(noisy, caption=f"Noisy Image ({noise_type})", use_container_width=True)
        st.write("Statistics of Noisy Image:")
        compute_basic_statistics(noisy)
