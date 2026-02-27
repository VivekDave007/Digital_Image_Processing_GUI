import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

def run(global_img=None):
    st.markdown("<h1>1.9 Image Statistics</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1], gap="small")
    
    if "stats_mode" not in st.session_state:
        st.session_state.stats_mode = "Basic Analysis"
        
    with col1:
        st.subheader("Command Center")
        
        if global_img is not None:
            img = cv2.resize(global_img, (200, 200))
        else:
            img = np.zeros((200, 200), dtype=np.uint8)
            for i in range(200):
                img[i, :] = i
            cv2.rectangle(img, (50, 50), (150, 150), (100), -1)
            
        st.session_state.stats_mode = st.radio(
            "Operation Matrix", 
            ["Basic Analysis", "Contrast Stretching", "Inject Gaussian Noise", "Inject Salt & Pepper"]
        )
        
        # Dynamic Parameters based on mode
        output_img = img.copy()
        
        if st.session_state.stats_mode == "Contrast Stretching":
            st.info("Linearly scales pixel intensities to utilize the full 0-255 dynamic range.")
            
        elif st.session_state.stats_mode == "Inject Gaussian Noise":
            noise_mean = st.slider("Gaussian μ (Mean)", -50.0, 50.0, 0.0)
            noise_sigma = st.slider("Gaussian σ (Std Dev)", 0.0, 100.0, 25.0)
            gauss = np.random.normal(noise_mean, noise_sigma, img.shape)
            noisy_image = img.astype(np.float32) + gauss
            output_img = np.clip(noisy_image, 0, 255).astype(np.uint8)
            
        elif st.session_state.stats_mode == "Inject Salt & Pepper":
            noise_prob = st.slider("Corruption Probability (p)", 0.0, 0.5, 0.05)
            thres = 1 - noise_prob
            
            # Fast vectorized S&P Noise (much faster than nested loops in old version)
            rdn = np.random.random(img.shape)
            output_img[rdn < noise_prob] = 0
            output_img[rdn > thres] = 255
            
    with col2:
        st.subheader("Viewport Alpha")
        
        if st.session_state.stats_mode == "Basic Analysis":
            st.image(output_img, caption="Analyzed Source Signal", use_container_width=True, clamp=True)
            
        elif st.session_state.stats_mode == "Contrast Stretching":
            min_val, max_val = np.min(img), np.max(img)
            if max_val - min_val > 0:
                stretched = 255.0 * (img - min_val) / (max_val - min_val)
                output_img = np.uint8(stretched)
            c1, c2 = st.columns(2)
            c1.image(img, caption=f"Original Profile: [{min_val}, {max_val}]", use_container_width=True, clamp=True)
            c2.image(output_img, caption="Stretched Output: [0, 255]", use_container_width=True, clamp=True)
            
        else:
            c1, c2 = st.columns(2)
            c1.image(img, caption="Pristine Source", use_container_width=True, clamp=True)
            c2.image(output_img, caption="Corrupted Payload", use_container_width=True, clamp=True)

    with col3:
        st.subheader("System Telemetry")
        
        # Calculate Stats for the CURRENT active image (output_img)
        mean_val = np.mean(output_img)
        var_val = np.var(output_img)
        std_val = np.std(output_img)
        min_val = np.min(output_img)
        max_val = np.max(output_img)
        
        m1, m2 = st.columns(2)
        m1.metric("Mean (μ)", f"{mean_val:.2f}")
        m2.metric("Std Dev (σ)", f"{std_val:.2f}")
        
        m3, m4 = st.columns(2)
        m3.metric("Variance", f"{var_val:.2f}")
        m4.metric("Range", f"[{min_val}, {max_val}]")
        
        st.markdown("### Intensity Profile")
        hist = cv2.calcHist([output_img], [0], None, [256], [0, 256])
        
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#090a0f') # Cyberpunk Match
        ax.set_facecolor('#090a0f')
        ax.plot(hist, color='#00f3ff')
        ax.fill_between(np.arange(256), hist.flatten(), color='#00f3ff', alpha=0.3)
        ax.tick_params(colors="#8a8d98")
        for spine in ax.spines.values():
            spine.set_color('#8a8d98')
        ax.set_xlim([0, 255])
        st.pyplot(fig)
