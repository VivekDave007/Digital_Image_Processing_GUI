import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_distances(p1, p2):
    """
    Calculates Euclidean, City-block (D4), and Chessboard (D8) distances between two points.
    
    Args:
        p1: Tuple (x, y)
        p2: Tuple (x, y)
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Euclidean (De)
    de = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    # City-block (D4) - Manhattan
    d4 = abs(x1 - x2) + abs(y1 - y2)
    
    # Chessboard (D8) - Max diff
    d8 = max(abs(x1 - x2), abs(y1 - y2))
    
    print(f"\nDistances between {p1} and {p2}:")
    print(f"  Euclidean (De): {de:.2f}")
    print(f"  City-block (D4): {d4}")
    print(f"  Chessboard (D8): {d8}")
    return de, d4, d8

def distance_transform_demo():
    print("\n--- Distance Transform Demo ---")
    size = 100
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Create some "foreground" objects (pixels = 1)
    # Note: OpenCV distanceTransform calculates distance to closest ZERO pixel.
    # So we need to invert the logic: Objects should be 0, Background 1 for the transform 
    # OR we treat our objects as the 'sources' and we want distance FROM them?
    # Usually Distance Transform is distance from every pixel to the nearest "boundary" or "background".
    # Standard usage: Input is binary image. Distances calculated for foreground pixels to nearest zero.
    
    # Let's create a binary image where we have an object in the center.
    cv2.circle(img, (50, 50), 10, 1, -1)
    cv2.rectangle(img, (20, 20), (30, 80), 1, -1)
    
    # For visualization, we want distance FROM the object boundary OUTWARDS...
    # OR distance from background INTO the object center (Skeletonization context).
    # "Input: binary image where foreground pixels are 1 and background are 0. Compute a distance transform..."
    # Usually this means for every '1' pixel, how far is it from a '0' pixel.
    
    # L2 (Euclidean)
    dist_l2 = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    
    # L1 (City-block / D4)
    # OpenCV only supports L1, L2, C (Chessboard).
    dist_l1 = cv2.distanceTransform(img, cv2.DIST_L1, 3)
    
    # C (Chessboard / D8)
    dist_c = cv2.distanceTransform(img, cv2.DIST_C, 3)
    
    # Normalize for visualization
    def normalize(d):
        return cv2.normalize(d, None, 0, 1.0, cv2.NORM_MINMAX)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(normalize(dist_l2), cmap='viridis')
    plt.title("Euclidean (L2)")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(normalize(dist_l1), cmap='magma')
    plt.title("City-block (L1)")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(normalize(dist_c), cmap='plasma')
    plt.title("Chessboard (L_inf)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Optional: Surface plot for one of them
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(size), np.arange(size))
        ax.plot_surface(X, Y, dist_l2, cmap='viridis')
        ax.set_title("3D Surface Plot of Euclidean Distance Transform")
        plt.show()
    except Exception as e:
        print("Could not show 3D plot:", e)

def run_experiment():
    # Point-to-point demo
    calculate_distances((10, 10), (13, 14)) # 3, 4 -> 5, 7, 4
    calculate_distances((0, 0), (10, 10))
    
    distance_transform_demo()

if __name__ == "__main__":
    run_experiment()
