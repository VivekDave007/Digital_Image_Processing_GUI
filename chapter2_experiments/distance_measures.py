import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

def run(global_img=None):
    st.markdown("<h1>1.7 Distance Measures</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1], gap="small")
    
    # Store operation mode in session state to persist choices
    if "dist_mode" not in st.session_state:
        st.session_state.dist_mode = "Point Metrics"

    with col1:
        st.subheader("Command Center")
        st.session_state.dist_mode = st.radio("Simulation Mode", ["Point Metrics", "Distance Transforms"])
        
        if st.session_state.dist_mode == "Point Metrics":
            st.markdown("### Point 1 Coordinates")
            x1 = st.slider("P1 X", 0, 100, 10, key="px1")
            y1 = st.slider("P1 Y", 0, 100, 10, key="py1")
            
            st.markdown("### Point 2 Coordinates")
            x2 = st.slider("P2 X", 0, 100, 80, key="px2")
            y2 = st.slider("P2 Y", 0, 100, 80, key="py2")
            
            p1 = (x1, y1)
            p2 = (x2, y2)
            
        elif st.session_state.dist_mode == "Distance Transforms":
            st.markdown("### Transform Parameters")
            show_3d = st.checkbox("Enable 3D Surface Plot Rendering (L2)", value=False)
            
    with col2:
        st.subheader("Viewport Alpha")
        
        if st.session_state.dist_mode == "Point Metrics":
            # Visualize the two points on a 100x100 grid
            grid = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.circle(grid, p1, 2, (0, 255, 255), -1) # Yellow
            cv2.circle(grid, p2, 2, (0, 0, 255), -1)   # Red
            cv2.line(grid, p1, p2, (255, 255, 255), 1)
            
            # Zoom up for better visibility
            big_grid = cv2.resize(grid, (400, 400), interpolation=cv2.INTER_NEAREST)
            st.image(big_grid, caption="Spatial Domain Coordinate Grid (100x100)", use_container_width=False, clamp=True)
            
        elif st.session_state.dist_mode == "Distance Transforms":
            size = 100
            img = np.zeros((size, size), dtype=np.uint8)
            cv2.circle(img, (50, 50), 10, 1, -1)
            cv2.rectangle(img, (20, 20), (30, 80), 1, -1)
            
            dist_l2 = cv2.distanceTransform(img, cv2.DIST_L2, 5)
            dist_l1 = cv2.distanceTransform(img, cv2.DIST_L1, 3)
            dist_c = cv2.distanceTransform(img, cv2.DIST_C, 3)
            
            def normalize(d):
                return cv2.normalize(d, None, 0, 1.0, cv2.NORM_MINMAX)

            if show_3d:
                fig = plt.figure(figsize=(6, 4))
                fig.patch.set_facecolor('#090a0f') # Match Cyberpunk BG
                ax = fig.add_subplot(111, projection='3d')
                ax.set_facecolor('#090a0f')
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                
                X, Y = np.meshgrid(np.arange(size), np.arange(size))
                ax.plot_surface(X, Y, dist_l2, cmap='plasma')
                ax.set_title("3D Euclidean Distance Map", color="#00f3ff")
                ax.tick_params(colors="#8a8d98")
                st.pyplot(fig)
            else:
                c1, c2 = st.columns(2)
                c1.image(img * 255, caption="1. Binary Source Block", clamp=True, use_container_width=True)
                c2.image(normalize(dist_l2), caption="2. Euclidean Map (L2)", clamp=True, use_container_width=True)
                
                c3, c4 = st.columns(2)
                c3.image(normalize(dist_l1), caption="3. City-block Map (L1)", clamp=True, use_container_width=True)
                c4.image(normalize(dist_c), caption="4. Chessboard Map (D8)", clamp=True, use_container_width=True)

    with col3:
        st.subheader("System Telemetry")
        
        if st.session_state.dist_mode == "Point Metrics":
            de = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            d4 = abs(x1 - x2) + abs(y1 - y2)
            d8 = max(abs(x1 - x2), abs(y1 - y2))
            
            st.metric("Euclidean (De)", f"{de:.2f}")
            st.caption("Straight-line radial distance. Computes the hypotenuse.")
            
            st.metric("City-block (D4)", f"{d4}")
            st.caption("Manhattan distance. Constrained to horizontal/vertical grid movement.")
            
            st.metric("Chessboard (D8)", f"{d8}")
            st.caption("Chebyshev distance. Diagonal movement costs the same as orthogonal.")
            
        elif st.session_state.dist_mode == "Distance Transforms":
            st.info("Calculates the distance to the closest zero pixel for each reference pixel.")
            st.metric("Input State", "Binary Map")
            st.metric("Output State", "Float Density Map")
            st.markdown("The resulting intensity of a pixel represents its absolute distance from the object boundary.")
