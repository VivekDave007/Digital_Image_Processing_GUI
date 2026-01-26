# Digital Image Processing - Chapter 2 Educational System

**An Interactive GUI for "Digital Image Fundamentals" (Gonzalez & Woods)**

This application represents a rigorous, book-faithful translation of **Chapter 2** of the standard Digital Image Processing textbook into an interactive software experience. It is designed to build intuition through real-time simulation of optical and mathematical concepts.

## 📚 Modules Overview

The system is divided into 6 interactive learning modules:

### 1. Elements of Visual Perception
*   **Concept**: How the eye adapts to brightness and perceives contrast.
*   **Simulation**: 
    *   **Scotopic vs Photopic Vision**: Watch the scene fade to grayscale and lose detail as you lower the illumination (simulating rod cell dominance).
    *   **Mach Bands**: Toggle the optical illusion where the eye perceives "overshoot" bands at sharp intensity transitions.

### 2. Light & The Electromagnetic Spectrum
*   **Concept**: The physical nature of proper image formation.
*   **Simulation**: 
    *   **Interactive Spectrum**: Slide from Radio waves to Gamma rays.
    *   **Real-time Physics**: See specific values for Frequency ($\nu$), Wavelength ($\lambda$), and Energy ($E$) update instantly.
    *   **Contextual Views**: See what an "image" looks like in different bands (e.g., Thermal for IR, Bones for X-Ray).

### 3. Image Sensing & Acquisition
*   **Concept**: How continuous energy is converted into digital data.
*   **Simulation**: 
    *   **Single Sensor**: Visualize a laser/drum scanner capturing pixels one by one.
    *   **Sensor Strip**: Visualize a flatbed scanner capturing row by row.
    *   **Sensor Array**: Visualize a digital camera capturing the full frame instantly.

### 4. Image Sampling & Quantization
*   **Concept**: The two steps to create a digital image.
*   **Simulation**: 
    *   **Sampling ($N$)**: Reduce spatial resolution (e.g., 512x512 $\to$ 16x16) to see checkerboard pixelation artifacts.
    *   **Quantization ($k$)**: Reduce intensity levels (e.g., 8-bit $\to$ 1-bit) to see false contouring/banding artifacts.

### 5. Some Basic Relationships Between Pixels
*   **Concept**: Spatial geometry on a grid.
*   **Simulation**: 
    *   **Connectivity**: Explore 4-neighbors ($N_4$) vs 8-neighbors ($N_8$).
    *   **Pathfinding**: Animate Breadth-First Search (BFS) to see how connectivity affects paths between pixels.
    *   **Distance Metrics**: Visualize Euclidean, City-Block, and Chessboard distance maps.

### 6. Mathematical Tools
*   **Concept**: Treating images as matrices.
*   **Simulation**: 
    *   **Arithmetic**: Add, Subtract, Multiply, and Divide images.
    *   **Noise Averaging**: Witness the "Law of Large Numbers" by averaging 100 noisy frames to recover a clean signal ($\sigma_{\bar{g}} = \sigma_{\eta} / \sqrt{K}$).

## 🚀 Installation & Usage

### Prerequisites
*   Python 3.10+
*   Dependencies listed in `requirements.txt`

### Setup
1.  Clone the repository or download the source.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the App
Execute the main entry point:
```bash
python main.py
```

## 🛠️ Technology Stack
*   **GUI Framework**: PySide6 (Qt for Python)
*   **Computation**: NumPy (Matrix operations)
*   **Rendering**: QPainter / Qt Custom Widgets

---
*Created for Educational Purposes based on Gonzalez & Woods, Digital Image Processing, 4th Ed.*
