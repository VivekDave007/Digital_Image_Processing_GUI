import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

def manual_connected_components(binary_img, connectivity=4):
    H, W = binary_img.shape
    labels = np.zeros((H, W), dtype=np.int32)
    current_label = 0
    
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else: 
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
                
    with st.spinner(f'Running BFS for {connectivity}-connectivity...'):
        for i in range(H):
            for j in range(W):
                if binary_img[i, j] == 1 and labels[i, j] == 0:
                    current_label += 1
                    queue = deque([(i, j)])
                    labels[i, j] = current_label
                    
                    while queue:
                        cx, cy = queue.popleft()
                        
                        for dx, dy in dirs:
                            nx, ny = cx + dx, cy + dy
                            
                            if 0 <= nx < H and 0 <= ny < W:
                                if binary_img[nx, ny] == 1 and labels[nx, ny] == 0:
                                    labels[nx, ny] = current_label
                                    queue.append((nx, ny))
                                
    return labels, current_label

def library_connected_components(binary_img, connectivity=8):
    num_labels, labels = cv2.connectedComponents(binary_img, connectivity=connectivity)
    return labels, num_labels - 1 

def visualize_components(title, labels, num_labels, ax):
    label_hue = np.uint8(179 * labels / np.max(labels)) if np.max(labels) > 0 else np.zeros_like(labels, dtype=np.uint8)
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[labels == 0] = 0
    
    ax.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{title}\nCount: {num_labels}")
    ax.axis('off')

def run():
    st.header("Topic A: Connected Components")
    
    connectivity = st.radio("Connectivity", [4, 8])
    
    # Generate Synthetic Data
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (30, 30), 1, -1)
    cv2.circle(img, (70, 70), 15, 1, -1)
    cv2.line(img, (50, 10), (50, 40), 1, 3)
    cv2.line(img, (50, 40), (80, 40), 1, 3)
    cv2.line(img, (80, 40), (80, 10), 1, 3)
    img[20, 60] = 1 # Noise
    img[22, 62] = 1 
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Manual Implementation (BFS)")
        labels_man, count_man = manual_connected_components(img, connectivity=connectivity)
        fig1, ax1 = plt.subplots()
        visualize_components("BFS Result", labels_man, count_man, ax1)
        st.pyplot(fig1)
        st.info(f"Detected {count_man} components.")
        
    with col2:
        st.subheader("Library Implementation (OpenCV)")
        labels_lib, count_lib = library_connected_components(img, connectivity=connectivity)
        fig2, ax2 = plt.subplots()
        visualize_components("OpenCV Result", labels_lib, count_lib, ax2)
        st.pyplot(fig2)
        st.info(f"Detected {count_lib} components.")
        
    st.image(img * 255, caption="Original Binary Image", width=300)
    
