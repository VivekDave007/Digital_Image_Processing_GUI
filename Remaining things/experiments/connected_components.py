import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

def manual_connected_components(binary_img, connectivity=4):
    """
    Manually implements connected component labeling using BFS.
    
    Args:
        binary_img: Binary image (0 background, 1 foreground).
        connectivity: 4 or 8.
    
    Returns:
        labels: Matrix of same size with component labels.
        num_labels: Total count of components.
        stats: Basic stats (area per label).
    """
    print(f"Running Manual Connected Components (BFS) - {connectivity}-connectivity...")
    H, W = binary_img.shape
    labels = np.zeros((H, W), dtype=np.int32)
    current_label = 0
    
    # Directions
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else: # 8
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
                
    for i in range(H):
        for j in range(W):
            if binary_img[i, j] == 1 and labels[i, j] == 0:
                current_label += 1
                # Start BFS
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
    """
    Wrapper for cv2.connectedComponents.
    """
    print(f"Running Library (OpenCV) Connected Components - {connectivity}-connectivity...")
    # cv2.connectedComponents expects image to be uint8
    num_labels, labels = cv2.connectedComponents(binary_img, connectivity=connectivity)
    return labels, num_labels - 1 # subtracting background

def visualize_components(title, labels, num_labels):
    # Map labels to colors
    # Create a random color map
    # We add 1 to num_labels for background (0)
    label_hue = np.uint8(179 * labels / np.max(labels)) if np.max(labels) > 0 else np.zeros_like(labels, dtype=np.uint8)
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # Set background to black
    labeled_img[labels == 0] = 0
    
    # For clearer matplotlib viz, let's use a distinct colormap
    # But for raw image handling usually RGB. 
    # Let's adhere to a simple distinct coloring usually available in matplotlib 'tab20'
    
    plt.imshow(labels, cmap='nipy_spectral', interpolation='nearest')
    plt.title(f"{title}\nCount: {num_labels}")
    plt.colorbar()
    plt.axis('off')

def run_experiment():
    print("\n--- Connected Components Experiment ---")
    
    # generate synthetic binary image
    img = np.zeros((100, 100), dtype=np.uint8)
    
    # Object 1: Rectangle
    cv2.rectangle(img, (10, 10), (30, 30), 1, -1)
    
    # Object 2: Circle
    cv2.circle(img, (70, 70), 15, 1, -1)
    
    # Object 3: Complex shape (U-shape)
    cv2.line(img, (50, 10), (50, 40), 1, 3)
    cv2.line(img, (50, 40), (80, 40), 1, 3)
    cv2.line(img, (80, 40), (80, 10), 1, 3)
    
    # Object 4: Small noise dots
    img[20, 60] = 1
    img[22, 62] = 1 # diagonally connected to previous if 8-conn
    
    plt.figure(figsize=(15, 5))
    
    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Binary Image")
    plt.axis('off')
    
    # Manual 4-connectivity
    labels_man, count_man = manual_connected_components(img, connectivity=4)
    plt.subplot(1, 3, 2)
    visualize_components("Manual BFS (4-connectivity)", labels_man, count_man)
    
    # Library 8-connectivity
    labels_lib, count_lib = library_connected_components(img, connectivity=8)
    plt.subplot(1, 3, 3)
    visualize_components("Library (8-connectivity)", labels_lib, count_lib)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Manual (4-conn) found {count_man} components.")
    print(f"Library (8-conn) found {count_lib} components.")

if __name__ == "__main__":
    run_experiment()
