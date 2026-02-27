# DIP Chapter 2 Experiments

This project contains practical implementations of concepts from Chapter 2 of "Digital Image Processing" by Gonzalez & Woods.

## Structure

- `main.py`: Main entry point with a menu to select experiments.
- `experiments/`: Folder containing separate modules for each topic.
    - `pixel_relationships.py`: Connectivity (4, 8, m) and boundary extraction.
    - `connected_components.py`: Labeling using valid algorithms (BFS/DFS vs Library).
    - `distance_measures.py`: Euclidean, City-block, Chessboard distances and transforms.
    - `sampling_quantization_extended.py`: Image resolution and gray-level resolution effects.
    - `image_statistics.py`: Basic statistical tools, histograms, and contrast stretching.

## Prerequisites

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

## Installation

```bash
pip install numpy opencv-python matplotlib
```

## How to Run

1. Navigate to the `Remaining things` directory.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Select an experiment from the menu by entering the corresponding number.

## Topics Covered

### A. Pixel Relationships & Connectivity
- **Visualizing Connectivity**: Shows neighbors for 4-, 8-, and m-connectivity.
- **Region Boundaries**: Extracts boundaries using morphological erosion ($Region - Eroded(Region)$).
- **Connected Components**: Implements labeling algorithms manually and compares with `cv2.connectedComponents`.

### B. Distance Measures
- **Metrics**: Computes Euclidean ($D_E$), City-block ($D_4$), and Chessboard ($D_8$) distances.
- **Distance Transform**: Visualizes distance maps for different metrics.

### C. Sampling and Quantization
- **Downsampling**: Reduces spatial resolution.
- **Quantization**: Reduces number of gray levels.
- **Quality Metrics**: Computes MSE and PSNR for degraded images.

### D. Image Statistics
- **Stats**: Mean, variance, standard deviation, min, max.
- **Histogram**: Computes and plots image histograms.
- **Contrast**: Simple linear contrast stretching.
