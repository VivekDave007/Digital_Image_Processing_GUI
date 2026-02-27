import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_neighbors_4(x, y, shape):
    """Returns coordinates of 4-neighbors."""
    H, W = shape
    neighbors = []
    if x > 0: neighbors.append((x-1, y))
    if x < H-1: neighbors.append((x+1, y))
    if y > 0: neighbors.append((x, y-1))
    if y < W-1: neighbors.append((x, y+1))
    return neighbors

def get_neighbors_diagonal(x, y, shape):
    """Returns coordinates of diagonal neighbors (ND)."""
    H, W = shape
    neighbors = []
    if x > 0 and y > 0: neighbors.append((x-1, y-1))
    if x > 0 and y < W-1: neighbors.append((x-1, y+1))
    if x < H-1 and y > 0: neighbors.append((x+1, y-1))
    if x < H-1 and y < W-1: neighbors.append((x+1, y+1))
    return neighbors

def get_neighbors_8(x, y, shape):
    """Returns coordinates of 8-neighbors (N4 + ND)."""
    return get_neighbors_4(x, y, shape) + get_neighbors_diagonal(x, y, shape)

def get_neighbors_m(x, y, img):
    """
    Returns m-neighbors.
    Rule: 
    1. q is in N4(p) OR
    2. q is in ND(p) AND (N4(p) INTERSECT N4(q)) is empty (set of 1-valued pixels)
    """
    p_val = img[x, y]
    shape = img.shape
    H, W = shape
    
    n4 = get_neighbors_4(x, y, shape)
    nd = get_neighbors_diagonal(x, y, shape)
    
    m_neighbors = []
    
    # Condition 1: 4-neighbors are always m-neighbors if they exist (assuming connectivity typically implies checking values, 
    # but here we follow the formal definition usually applied to binary regions where we check existence in V.
    # We assume V={1} for "object" pixels. The "neighbors" are candidates.
    # Usually we define adjacency between p and q.
    # Here we list candidates that *would* be m-connected if they have the value from set V.
    # Let's assume we are looking for valid m-paths on the FOREGROUND (val=1).
    
    # Actually, the problem asks to "Show which pixels are neighbors...".
    # We will show potential neighbors based on geometry and m-connectivity rule logic 
    # assuming the neighbor pixel itself is also 'active' (value 1).
    # If the image is binary, we only care about neighbors that are 1.
    
    # Add valid 4-neighbors
    for nx, ny in n4:
        if img[nx, ny] == 1:
            m_neighbors.append((nx, ny))
            
    # Add valid Diagonal neighbors if condition 2 holds
    for qx, qy in nd:
        if img[qx, qy] == 1:
            # Check intersection of N4(p) and N4(q)
            n4_p = set(n4)
            n4_q = set(get_neighbors_4(qx, qy, shape))
            
            intersection = n4_p.intersection(n4_q)
            
            # Check if any pixel in intersection is 1 (from set V)
            is_empty_intersection_of_ones = True
            for kx, ky in intersection:
                if img[kx, ky] == 1:
                    is_empty_intersection_of_ones = False
                    break
            
            if is_empty_intersection_of_ones:
                m_neighbors.append((qx, qy))
                
    return m_neighbors

def visualize_connectivity_demo():
    print("\n--- Pixel Connectivity Demo ---")
    size = 10
    img = np.zeros((size, size), dtype=int)
    
    # Create random scattered points
    np.random.seed(42)
    img = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])
    
    # Pick a center pixel that is 1
    p_x, p_y = size//2, size//2
    img[p_x, p_y] = 1 # Force center to be 1
    
    print(f"Analyzing connectivity for pixel at ({p_x}, {p_y})")
    print("Image patch (center 1 shown):")
    print(img[p_x-2:p_x+3, p_y-2:p_y+3]) # Show small area around center
    
    # 4-Connectivity
    n4 = get_neighbors_4(p_x, p_y, img.shape)
    conn4 = [(nx, ny) for nx, ny in n4 if img[nx, ny] == 1]
    
    # 8-Connectivity
    n8 = get_neighbors_8(p_x, p_y, img.shape)
    conn8 = [(nx, ny) for nx, ny in n8 if img[nx, ny] == 1]
    
    # m-Connectivity
    connm = get_neighbors_m(p_x, p_y, img)
    
    # Visualization using RGB
    # Background: Black (0)
    # Objects (1): Gray
    # Center P: Red
    # Neighbors: Green
    
    def show_grid(title, neighbors, ax):
        vis = np.zeros((size, size, 3), dtype=np.float32)
        # Draw objects
        vis[img == 1] = [0.5, 0.5, 0.5]
        # Draw Center
        vis[p_x, p_y] = [1, 0, 0]
        # Draw Neighbors
        for nx, ny in neighbors:
            vis[nx, ny] = [0, 1, 0]
            
        ax.imshow(vis)
        ax.set_title(title)
        ax.axis('off')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    show_grid(f"4-Connected (count: {len(conn4)})", conn4, axes[0])
    show_grid(f"8-Connected (count: {len(conn8)})", conn8, axes[1])
    show_grid(f"m-Connected (count: {len(connm)})", connm, axes[2])
    plt.tight_layout()
    plt.show()

def boundary_extraction_demo():
    print("\n--- Region Boundary Extraction ---")
    # Create a binary image with a shape (square)
    H, W = 100, 100
    binary_img = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(binary_img, (30, 30), (70, 70), 1, -1)
    
    # Add some noise/irregularity to make it interesting
    # cv2.circle(binary_img, (65, 65), 10, 0, -1)
    
    # Structuring element for erosion (3x3 cross or square)
    kernel = np.ones((3, 3), np.uint8) # 8-connectivity boundary
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)) # 4-connectivity boundary
    
    eroded = cv2.erode(binary_img, kernel, iterations=1)
    boundary = binary_img - eroded
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(binary_img, cmap='gray')
    plt.title("Original Region")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(eroded, cmap='gray')
    plt.title("Eroded Region")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(boundary, cmap='gray')
    plt.title("Boundary (Region - Eroded)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def run_experiment():
    visualize_connectivity_demo()
    boundary_extraction_demo()

if __name__ == "__main__":
    run_experiment()
