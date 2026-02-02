# DIP Web: Interactive Image Processing App

**A Streamlit-based educational tool for Digital Image Processing (Gonzalez & Woods, Chapter 2).**

This application allows you to run simulations and experiments directly in your web browser, covering "Digital Image Fundamentals" comprehensively.

## 🚀 Quick Start (Local)

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the App**:
    ```bash
    streamlit run app.py
    ```
    The app will open automatically in your default browser at `http://localhost:8501`.

## ☁️ Deployment (Render/Streamlit Cloud)

This project is ready for cloud deployment.
*   **Build Command**: `pip install -r requirements.txt`
*   **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## 📚 Modules Included

### Part 1: Fundamentals (Chapter 2)
1.  **Visual Perception**: Simulate eye adaptation (Scotopic/Photopic) and Mach bands.
2.  **EM Spectrum**: Interactive frequency/energy calculator and band visualization.
3.  **Acquisition**: Simulate sensor capture modes (Single, Strip, vs Array).
4.  **Sampling & Quantization**: 
    *   Spatial Resolution: Visualize downsampling effects (pixelation).
    *   Gray-Level Resolution: Visualize quantization effects (false contouring).
    *   Metrics: MSE and PSNR calculation.
5.  **Pixel Connectivity**: 
    *   Visualize 4-, 8-, and m-connectivity.
    *   Region boundary extraction ($\beta(A)$).
6.  **Math Tools**: Image arithmetic (Addition for noise reduction, Subtraction, Multiplication).
7.  **Distance Measures**: 
    *   Calculate Euclidean ($D_E$), City-Block ($D_4$), and Chessboard ($D_8$) distances.
    *   Visualize Distance Transforms (3D Surface plots).
8.  **Connected Components**: 
    *   Manual BFS implementation vs OpenCV Library.
    *   Component labeling and counting.
9.  **Image Statistics**: 
    *   Basic Stats: Mean, Variance, Min, Max.
    *   Histogram Visualization.
    *   Contrast Stretching/Normalization.
    *   Noise Simulation (Gaussian, Salt & Pepper).

### Part 2: Advanced Processing
1.  **Frequency Domain**: FFT, Lowpass/Highpass filters (Ideal, Butterworth, Gaussian).
2.  **Spatial Filtering**: Convolution (Gaussian Blur, Median Filter).
3.  **Morphology**: Erosion, Dilation, Opening, Closing.

## 🛠️ Tech Stack
*   **Streamlit**: Web UI Framework
*   **OpenCV**: Image Processing Logic
*   **NumPy**: Matrix Operations
*   **Matplotlib**: Graphing & Plotting
