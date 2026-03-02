# PronOS Digital Image Processing (DIP) Platform

![PronOS Hero](https://img.shields.io/badge/UI_Theme-PronOS_Clean-367D5D?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

An advanced, interactive educational platform for Digital Image Processing (DIP). This application provides a hands-on, visually rich environment for understanding complex computer vision algorithms, mathematical transformations, and electromagnetic spectrum fundamentals.

It features a custom-built "PronOS" clean light theme for maximum readability and a professional UX.

## 🚀 Features

### 1. Fundamentals Concept Modules
Organized into interactive Chapters:
*   **Chapter 1: Fundamentals of Image Formation**: 
    *   `1.1` Visual Perception (Scotopic/Photopic adaptation, Mach bands)
    *   `1.2` Electromagnetic Spectrum
    *   `1.3` Image Acquisition topology
*   **Chapter 2: Image Digitization**: 
    *   `2.1` Sampling & Quantization resolution testing
    *   `2.2` Pixel Connectivity (4, 8, m-paths)
    *   `2.3` Distance Measures
*   **Chapter 3: Image Representation & Math Tools**: 
    *   `3.1` Mathematical Tools (Interactive 6-tab suite for Arithmetic, Statistical, Trigonometric, Transforms, Morphology, and Filtering)
    *   `3.2` Connected Components
    *   `3.3` Image Statistics
*   **Chapter 4: Advanced Vision Concepts**: 
    *   `4.1` 3D Eye Vision Game Model: A sophisticated physics engine demonstrating light rays tracking a visual object into the eye.
    *   `Anatomical Eye Exploration`: Interactive, dynamically rendered 2D eye cross-section. Toggle labels, highlight physiological features, and physically mutate sliders (Pupil Dilation, Lens Thickness, Cornea Bulge) to simulate Myopia and Astigmatism directly onto an image feed!

### 2. Advanced Processing Track
*   **Chapter 5: Spatial Domain Processing**:
    *   `5.1` Spatial Filtering (Convolution, median, and Gaussian blur kernels)
*   **Chapter 6: Frequency Domain Processing**:
    *   `6.1` Frequency Domain (Ideal, Butterworth, and Gaussian filtering)
*   **Chapter 7: Morphological Image Processing**:
    *   `7.1` Morphology (Erosion, Dilation, Opening, Closing routines)

### Global Features
*   **Native Streamlit Light & Dark Mode**: The custom CSS seamlessly inherits Streamlit's native theme engine without aggressive overrides. Toggle between light and dark modes in the settings menu for a pixel-perfect, highly readable aesthetic in any environment.
*   **Unified Global Image Override**: Upload a custom image directly from the unified sidebar to inject it simultaneously across a dozen different physics and mathematical simulations across the app's chapters.



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

## 🌐 Deployment (Render Free Tier via GitHub)

This repository is fully configured to deploy as a **Docker Container** on the **[Render](https://render.com/) Free Tier**, driven automatically by GitHub pushes. 

### Why Render + Docker?
Streamlit apps running complex image processing libraries (like OpenCV) often crash on standard cloud hosting due to missing C++ system dependencies. We have included a `Dockerfile` that automatically handles these Linux dependencies (like `libgl1-mesa-glx`) so the app runs flawlessly. The included `render.yaml` acts as a one-click deployment blueprint.

### Deployment Steps:
1.  **Push to GitHub**:
    Ensure this codebase is pushed to a public or private GitHub repository.
    ```bash
    git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
    git push -u origin main
    ```
2.  **Connect Render**:
    *   Sign up / Log in to [Render Dashboard](https://dashboard.render.com/).
    *   Click the **New +** button and select **Blueprint**.
3.  **Deploy**:
    *   Connect your GitHub account.
    *   Select the repository where you pushed this code.
    *   Render will automatically detect the `render.yaml` file.
    *   Click **Apply**.
    
Render will now read the blueprint, build the Docker container using our `Dockerfile`, install OpenCV and Streamlit, and generate a live URL for your application!

> **Note**: Because this uses the Render Free Tier, the application will "spin down" after 15 minutes of inactivity. When you visit it again, it may take 30-50 seconds to wake up.

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
