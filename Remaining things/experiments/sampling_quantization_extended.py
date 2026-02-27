import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_mse_psnr(img1, img2):
    """
    Computes MSE and PSNR between two images.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 0, float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return mse, psnr

def sampling_demo(image):
    """
    Generates downsampled versions of the image.
    """
    print("\n--- Spatial Sampling Demo ---")
    H, W = image.shape
    ratios = [0.5, 0.25, 0.125]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Original\n({W}x{H})")
    plt.axis('off')
    
    for i, r in enumerate(ratios):
        # Resize down (simulate sampling)
        new_w, new_h = int(W * r), int(H * r)
        sampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Resize back up for display/comparison using Nearest Neighbor (to see the blocks)
        upscaled = cv2.resize(sampled, (W, H), interpolation=cv2.INTER_NEAREST)
        
        mse, psnr = compute_mse_psnr(image, upscaled)
        
        plt.subplot(1, 4, i+2)
        plt.imshow(upscaled, cmap='gray')
        plt.title(f"Scale: {r}\n({new_w}x{new_h})\nPSNR: {psnr:.2f}dB")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

def quantization_demo(image):
    """
    Generates quantized versions of the image (reducing gray levels).
    Levels: 256, 128, 64, 32, 16, 8, 4, 2
    """
    print("\n--- Gray-Level Quantization Demo ---")
    levels_list = [128, 64, 32, 16, 8, 4, 2]
    
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original (256)")
    plt.axis('off')
    
    print(f"{'Levels':<10} | {'MSE':<10} | {'PSNR (dB)':<10}")
    print("-" * 36)
    
    for i, levels in enumerate(levels_list):
        # Quantize
        # Example: 4 levels -> 0, 85, 170, 255
        # Formula: floor(val / (256/levels)) * (255/(levels-1))
        
        div = 256 / levels
        quantized = np.floor(image / div) * (255 / (levels - 1))
        quantized = np.uint8(quantized)
        
        mse, psnr = compute_mse_psnr(image, quantized)
        print(f"{levels:<10} | {mse:<10.2f} | {psnr:<10.2f}")
        
        plt.subplot(2, 4, i+2)
        plt.imshow(quantized, cmap='gray')
        plt.title(f"Levels: {levels}\nPSNR: {psnr:.2f}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

def run_experiment():
    # Load or create an image
    # Using the standard 'lena' or similar if available, else synthetic
    # Let's create a smooth gradient circle to see quantization bands clearly
    H, W = 256, 256
    img = np.zeros((H, W), dtype=np.uint8)
    
    # Radial gradient
    Y, X = np.ogrid[:H, :W]
    center = (H//2, W//2)
    dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)
    img = 255 * (1 - dist_from_center / (np.sqrt(2)*128))
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    print("Using synthetic radial gradient image.")
    
    sampling_demo(img)
    quantization_demo(img)

if __name__ == "__main__":
    run_experiment()
