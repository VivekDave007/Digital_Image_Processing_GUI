import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_neighbors_4(x, y, shape):
    H, W = shape
    neighbors = []
    if x > 0: neighbors.append((x-1, y))
    if x < H-1: neighbors.append((x+1, y))
    if y > 0: neighbors.append((x, y-1))
    if y < W-1: neighbors.append((x, y+1))
    return neighbors

def get_neighbors_diagonal(x, y, shape):
    H, W = shape
    neighbors = []
    if x > 0 and y > 0: neighbors.append((x-1, y-1))
    if x > 0 and y < W-1: neighbors.append((x-1, y+1))
    if x < H-1 and y > 0: neighbors.append((x+1, y-1))
    if x < H-1 and y < W-1: neighbors.append((x+1, y+1))
    return neighbors

def get_neighbors_8(x, y, shape):
    return get_neighbors_4(x, y, shape) + get_neighbors_diagonal(x, y, shape)

def get_neighbors_m(x, y, img):
    p_val = img[x, y]
    shape = img.shape
    H, W = shape
    
    n4 = get_neighbors_4(x, y, shape)
    nd = get_neighbors_diagonal(x, y, shape)
    
    m_neighbors = []
    
    for nx, ny in n4:
        if img[nx, ny] == 1:
            m_neighbors.append((nx, ny))
            
    for qx, qy in nd:
        if img[qx, qy] == 1:
            n4_p = set(n4)
            n4_q = set(get_neighbors_4(qx, qy, shape))
            
            intersection = n4_p.intersection(n4_q)
            
            is_empty_intersection_of_ones = True
            for kx, ky in intersection:
                if img[kx, ky] == 1:
                    is_empty_intersection_of_ones = False
                    break
            
            if is_empty_intersection_of_ones:
                m_neighbors.append((qx, qy))
                
    return m_neighbors

def run(global_img=None):
    st.markdown("<h1>1.11 Pixel Relationships</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1], gap="small")

    if "pixel_mode" not in st.session_state:
        st.session_state.pixel_mode = "Connectivity Analysis"

    with col1:
        st.subheader("Command Center")
        st.session_state.pixel_mode = st.radio("Operation Matrix", ["Connectivity Analysis", "Boundary Extraction"])

        if st.session_state.pixel_mode == "Connectivity Analysis":
            size = st.slider("Sensor Grid Size", 5, 20, 10)
            seed = st.number_input("RNG Seed (Terrain Generation)", 0, 100, 42)
        else:
            add_circle = st.checkbox("Inject Secondary Object (Circle)", value=False)
            kernel_size = st.slider("Structuring Element Matrix", 3, 9, 3, step=2)

    with col2:
        st.subheader("Viewport Alpha")

        if st.session_state.pixel_mode == "Connectivity Analysis":
            np.random.seed(seed)
            img = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])
            p_x, p_y = size//2, size//2
            img[p_x, p_y] = 1 # Force center
            
            conn4 = [(nx, ny) for nx, ny in get_neighbors_4(p_x, p_y, img.shape) if img[nx, ny] == 1]
            conn8 = [(nx, ny) for nx, ny in get_neighbors_8(p_x, p_y, img.shape) if img[nx, ny] == 1]
            connm = get_neighbors_m(p_x, p_y, img)
            
            def render_grid(title, neighbors, ax):
                vis = np.zeros((size, size, 3), dtype=np.float32)
                vis[img == 1] = [0.3, 0.3, 0.3] # Gray objects
                vis[p_x, p_y] = [1, 0, 0.2] # Neon Pink Center
                for nx, ny in neighbors:
                    vis[nx, ny] = [0, 0.95, 1] # Neon Cyan Neighbors
                    
                ax.imshow(vis)
                ax.set_title(title, color="#e0e0e0", fontsize=10)
                ax.axis('off')
                ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
                ax.grid(which='minor', color='#12141d', linestyle='-', linewidth=1)
                for spine in ax.spines.values():
                    spine.set_color('#8a8d98')

            fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
            fig.patch.set_facecolor('#090a0f')
            render_grid(f"4-Connected ({len(conn4)})", conn4, axes[0])
            render_grid(f"8-Connected ({len(conn8)})", conn8, axes[1])
            render_grid(f"m-Connected ({len(connm)})", connm, axes[2])
            plt.tight_layout()
            st.pyplot(fig)

        else:
            H, W = 100, 100
            img = np.zeros((H, W), dtype=np.uint8)
            cv2.rectangle(img, (30, 30), (70, 70), 1, -1)
            if add_circle:
                cv2.circle(img, (60, 60), 20, 1, -1)
                
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            eroded = cv2.erode(img, kernel, iterations=1)
            boundary = img - eroded
            
            c1, c2, c3 = st.columns(3)
            c1.image(img * 255, caption="1. Original Region (A)", use_container_width=True, clamp=True)
            c2.image(eroded * 255, caption="2. Eroded Region (A ⊖ B)", use_container_width=True, clamp=True)
            c3.image(boundary * 255, caption="3. Boundary β(A)", use_container_width=True, clamp=True)

    with col3:
        st.subheader("System Telemetry")
        
        if st.session_state.pixel_mode == "Connectivity Analysis":
            st.info(f"Targeting center node at coordinates: **({p_x}, {p_y})**")
            st.metric("Detected N4 Connections", len(conn4))
            st.metric("Detected N8 Connections", len(conn8))
            st.metric("Detected Nm Connections", len(connm))
            st.caption("m-connectivity eliminates multiple path ambiguities (8-connectivity loops).")
        else:
            st.info("Morphological Boundary Extraction")
            st.latex(r"\beta(A) = A - (A \ominus B)")
            st.metric("Structuring Element (B)", f"{kernel_size}x{kernel_size}")
            st.caption("Eroding the image shrinks objects. Subtracting the eroded image from the original leaves only the boundary perimeter.")
