# PronOS Digital Image Processing (DIP) Platform

![PronOS Hero](https://img.shields.io/badge/UI_Theme-PronOS_Clean-367D5D?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

An advanced, interactive educational platform for Digital Image Processing (DIP). This application provides a hands-on, visually rich environment for understanding complex computer vision algorithms, mathematical transformations, and electromagnetic spectrum fundamentals.

It features a custom-built "PronOS" clean light theme for maximum readability and a professional UX.

## 🚀 Features

### Track 1: Fundamentals
*   **1.1 Visual Perception**: Interactive simulation of human eye adaptation (Scotopic vs. Photopic vision) and Mach Band illusions.
*   **1.2 EM Spectrum**: Calculate and visualize properties of the Electromagnetic Spectrum (Frequency, Wavelength, Energy) across different bands.
*   **1.3 Acquisition**: Simulates different image acquisition topologies (Microdensitometer point scanning, Line scanning, and Solid State array capture) with real-time visualizations.
*   **1.4 Sampling & Quantization**: Explore how spatial resolution and intensity levels affect image quality (includes zooming and histogram analysis).
*   **1.5 Pixel Connectivity**: Interactive grid to understand 4-connectivity, 8-connectivity, and m-connectivity.
*   **1.6 Math Tools**: A comprehensive suite of mathematical functions (Arithmetic, Statistical, Trigonometric, Transforms, Morphology, Filtering) applied to a unified input source.
*   **1.7 Distance Measures**: Learn Euclidean, City-Block (D4), and Chessboard (D8) distance metrics.
*   **1.8 Connected Components**: Simulate Connected Component Labeling algorithms on binary shapes.
*   **1.9 Image Statistics**: Analyze histograms, PDFs, Mean, Variance, Mode, and spatial distributions of noise.
*   **1.10 3D Eye Vision Game Model**: A sophisticated 2D/3D physics model demonstrating optics, light rays, wavelength scattering, and retinal perception using Matplotlib.

### Track 2: Advanced Processing
*   **2.1 Frequency Domain**: Apply Ideal, Butterworth, and Gaussian lowpass/highpass filters in the Fourier domain.
*   **2.2 Spatial Filtering**: Add specialized noise and apply custom convolution, median, and Gaussian blurs.
*   **2.3 Morphology**: Perform complex morphological operations (Erosion, Dilation, Opening, Closing) with adjustable structuring elements.

### Global Features
*   **Unified Global Image Override**: Upload a custom image from the sidebar to inject it universally into over a dozen different physics and mathematical simulations across the app!

## 💻 Tech Stack

*   **Frontend / Framework**: Streamlit
*   **Image Processing**: OpenCV (`opencv-python-headless`)
*   **Math & Matrices**: NumPy, SciPy
*   **Plotting**: Matplotlib

## 🛠️ Local Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
    cd YOUR_REPOSITORY
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## 🌐 Deployment (Streamlit Community Cloud)

This repository is pre-configured for direct deployment to **[Streamlit Community Cloud](https://streamlit.io/cloud)**.

1.  Push your code to a public GitHub repository.
2.  Log in to Streamlit Community Cloud.
3.  Click **New app**.
4.  Select your repository, branch (`main`), and main file path (`app.py`).
5.  Click **Deploy**.

> **Note**: This app uses `opencv-python-headless` in `requirements.txt` to ensure compatibility with Linux cloud environments without requiring system-level X11 installations.

## 📁 Repository Structure

```text
├── app.py                     # Main Streamlit application and routing
├── requirements.txt           # Python dependencies
├── .gitignore                 # Standard Python/Streamlit ignore file
└── chapter2_experiments/      # Modular logic files for various chapters
    ├── connected_components.py
    ├── distance_measures.py
    ├── image_statistics.py
    ├── pixel_relationships.py
    └── sampling_quantization_extended.py
```
