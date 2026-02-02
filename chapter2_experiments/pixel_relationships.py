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

def visualize_connectivity_demo():
    st.subheader("Pixel Connectivity Demo")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        size = st.number_input("Grid Size", 5, 20, 10)
        seed = st.number_input("Random Seed", 0, 100, 42)
        
    np.random.seed(seed)
    img = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])
    
    p_x, p_y = size//2, size//2
    img[p_x, p_y] = 1 # Force center
    
    st.write(f"Analyzing connectivity for center pixel at ({p_x}, {p_y})")

    conn4 = [(nx, ny) for nx, ny in get_neighbors_4(p_x, p_y, img.shape) if img[nx, ny] == 1]
    conn8 = [(nx, ny) for nx, ny in get_neighbors_8(p_x, p_y, img.shape) if img[nx, ny] == 1]
    connm = get_neighbors_m(p_x, p_y, img)
    
    def show_grid(title, neighbors, ax):
        vis = np.zeros((size, size, 3), dtype=np.float32)
        vis[img == 1] = [0.5, 0.5, 0.5] # Gray objects
        vis[p_x, p_y] = [1, 0, 0] # Red Center
        for nx, ny in neighbors:
            vis[nx, ny] = [0, 1, 0] # Green Neighbors
            
        ax.imshow(vis)
        ax.set_title(title)
        ax.axis('off')
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    show_grid(f"4-Connected ({len(conn4)})", conn4, axes[0])
    show_grid(f"8-Connected ({len(conn8)})", conn8, axes[1])
    show_grid(f"m-Connected ({len(connm)})", connm, axes[2])
    plt.tight_layout()
    st.pyplot(fig)

def boundary_extraction_demo():
    st.subheader("Region Boundary Extraction")
    st.write("Formula: $\\beta(A) = A - (A \ominus B)$")
    
    H, W = 100, 100
    img = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), 1, -1)
    
    if st.checkbox("Add Circle"):
        cv2.circle(img, (60, 60), 20, 1, -1)
        
    kernel_size = st.slider("Structuring Element Size", 3, 9, 3, step=2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    eroded = cv2.erode(img, kernel, iterations=1)
    boundary = img - eroded
    
    col1, col2, col3 = st.columns(3)
    col1.image(img * 255, caption="Original Region", clamp=True)
    col2.image(eroded * 255, caption="Eroded Region", clamp=True)
    col3.image(boundary * 255, caption="Extracted Boundary", clamp=True)

def run():
    st.header("Topic A: Pixel Relationships")
    tab1, tab2 = st.tabs(["Connectivity", "Boundaries"])
    with tab1:
        visualize_connectivity_demo()
    with tab2:
        boundary_extraction_demo()
