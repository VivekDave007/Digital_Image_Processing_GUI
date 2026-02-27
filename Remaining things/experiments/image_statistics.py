import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_basic_statistics(image: np.ndarray):
    """
    Computes and prints basic statistics of a grayscale image.
    
    Args:
        image: Input grayscale image.
    """
    print("\n--- Basic Image Statistics ---")
    mean_val = np.mean(image)
    var_val = np.var(image)
    std_val = np.std(image)
    min_val = np.min(image)
    max_val = np.max(image)
    
    print(f"Mean: {mean_val:.2f}")
    print(f"Variance: {var_val:.2f}")
    print(f"Standard Deviation: {std_val:.2f}")
    print(f"Min Intensity: {min_val}")
    print(f"Max Intensity: {max_val}")
    
    # Histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Analyzed Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.plot(hist, color='black')
    plt.title("Grayscale Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()

def contrast_stretching(image: np.ndarray):
    """
    Performs simple contrast stretching (normalization) to full [0, 255] range.
    Formula: g(x,y) = 255 * (f(x,y) - min) / (max - min)
    
    Args:
        image: Input grayscale image.
    """
    print("\n--- Contrast Stretching ---")
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val - min_val == 0:
        print("Image has constant value, cannot stretch contrast.")
        return image
        
    stretched = 255.0 * (image - min_val) / (max_val - min_val)
    stretched = np.uint8(stretched)
    
    print(f"Original Range: [{min_val}, {max_val}]")
    print(f"Stretched Range: [{np.min(stretched)}, {np.max(stretched)}]")
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title("Original (Low Contrast?)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(stretched, cmap='gray', vmin=0, vmax=255)
    plt.title("Contrast Stretched")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return stretched

def add_noise(image: np.ndarray, noise_type="gaussian"):
    """
    Adds noise to an image to demonstrate statistical changes.
    
    Args:
        image: Input grayscale image.
        noise_type: "gaussian" or "salt_pepper".
    """
    print(f"\n--- Adding {noise_type.capitalize()} Noise ---")
    noisy_image = image.copy()
    
    if noise_type == "gaussian":
        mean = 0
        sigma = 25
        gauss = np.random.normal(mean, sigma, image.shape)
        noisy_image = image.astype(np.float32) + gauss
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
    elif noise_type == "salt_pepper":
        prob = 0.05
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = np.random.random()
                if rdn < prob:
                    noisy_image[i][j] = 0
                elif rdn > thres:
                    noisy_image[i][j] = 255
                    
    compute_basic_statistics(noisy_image)
    return noisy_image

def run_experiment():
    print("Running Image Statistics Experiment...")
    # Create a synthetic image if no file is provided, or user can load one.
    # For demo purposes, we'll create a synthetic gradient image with some structure.
    img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(200):
        img[i, :] = i  # Vertical gradient
    
    # Add a rectangle
    cv2.rectangle(img, (50, 50), (150, 150), (100), -1)
    
    print("Using a synthetic gradient image for demonstration.")
    
    compute_basic_statistics(img)
    
    # Create a low contrast version for stretching
    low_contrast = (img * 0.3 + 50).astype(np.uint8)
    contrast_stretching(low_contrast)
    
    # Noise demo
    add_noise(img, "gaussian")
    add_noise(img, "salt_pepper")

if __name__ == "__main__":
    run_experiment()
