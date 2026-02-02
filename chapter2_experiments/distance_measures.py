import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_distances(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    de = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    d4 = abs(x1 - x2) + abs(y1 - y2)
    d8 = max(abs(x1 - x2), abs(y1 - y2))
    
    st.write(f"**Distances between {p1} and {p2}:**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Euclidean (De)", f"{de:.2f}")
    col2.metric("City-block (D4)", f"{d4}")
    col3.metric("Chessboard (D8)", f"{d8}")

def distance_transform_demo():
    st.subheader("Distance Transform Visualization")
    size = 100
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Create objects (Foreground = 1, Background = 0 for visualization)
    # distanceTransform calculates distance to nearest zero.
    # So we invert logic: Objects = 0, Background = 1.
    
    # We want distance FROM object boundary.
    # Let's say Object is White (1). We want distance from 1 to 0.
    # Actually OpenCV distanceTransform: "Calculates the distance to the closest zero pixel for each pixel of the source image."
    # So if we want distance INSIDE an object, object should be 1, background 0.
    
    cv2.circle(img, (50, 50), 10, 1, -1)
    cv2.rectangle(img, (20, 20), (30, 80), 1, -1)
    
    dist_l2 = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    dist_l1 = cv2.distanceTransform(img, cv2.DIST_L1, 3)
    dist_c = cv2.distanceTransform(img, cv2.DIST_C, 3)
    
    def normalize(d):
        return cv2.normalize(d, None, 0, 1.0, cv2.NORM_MINMAX)

    col0, col1, col2, col3 = st.columns(4)
    col0.image(img * 255, caption="Binary Image", clamp=True, use_container_width=True)
    
    col1.image(normalize(dist_l2), caption="Euclidean (L2)", clamp=True, use_container_width=True)
    col2.image(normalize(dist_l1), caption="City-block (L1)", clamp=True, use_container_width=True)
    col3.image(normalize(dist_c), caption="Chessboard (D8)", clamp=True, use_container_width=True)
    
    # 3D Plot
    if st.checkbox("Show 3D Surface Plot (Euclidean)"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(size), np.arange(size))
        ax.plot_surface(X, Y, dist_l2, cmap='viridis')
        ax.set_title("3D Distance Transform")
        st.pyplot(fig)

def run():
    st.header("Topic B: Distance Measures")
    
    tab1, tab2 = st.tabs(["Point Metrics", "Distance Transforms"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            x1 = st.number_input("P1 X", 0, 100, 10)
            y1 = st.number_input("P1 Y", 0, 100, 10)
        with c2:
            x2 = st.number_input("P2 X", 0, 100, 20)
            y2 = st.number_input("P2 Y", 0, 100, 20)
            
        calculate_distances((x1, y1), (x2, y2))
        
    with tab2:
        distance_transform_demo()
