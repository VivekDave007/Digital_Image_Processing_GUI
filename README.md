# DIP Web: Interactive Image Processing App

**A Streamlit-based educational tool for Digital Image Processing (Gonzalez & Woods, Chapter 2).**

This application allows you to run the simulations directly in your web browser without installing any desktop GUI libraries.

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

### Deploy on Render.com
1.  Push this folder to a GitHub repository (e.g., `DIP_Web`).
2.  Create a **New Web Service** on Render.
3.  Connect your GitHub repo.
4.  Use these settings:
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5.  Click **Deploy**.

## 📚 Modules Included

1.  **Visual Perception**: Simulate eye adaptation and Mach bands.
2.  **EM Spectrum**: Interactive frequency/energy calculator.
3.  **Acquisition**: Simulate sensor capture modes (Single vs Array).
4.  **Sampling**: Adjust spatial resolution ($N$) in real-time.
5.  **Relationships**: Visualize distance metrics (Euclidean, City-Block) on a grid.
6.  **Math Tools**: Image arithmetic (Addition, Subtraction) and Noise Reduction.

## 🛠️ Tech Stack
*   **Streamlit**: Web UI Framework
*   **OpenCV**: Image Processing Logic
*   **NumPy**: Matrix Operations
*   **Matplotlib**: Graphing & Plotting
