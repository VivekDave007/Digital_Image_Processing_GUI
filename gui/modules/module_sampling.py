import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox, QSplitter)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor

class ModuleSampling(QWidget):
    def __init__(self):
        super().__init__()
        
        # Parameters
        self.sampling_n = 512 # Spatial Resolution (NxN)
        self.quant_k = 8      # Intensity Levels (2^k)
        
        # Generate Source Image (Gradient + Circle)
        self.source_size = 512
        self.source_img = self.generate_source_image()
        
        layout = QHBoxLayout(self)
        
        # -- Controls --
        controls = QWidget()
        controls.setFixedWidth(250)
        controls.setStyleSheet("background-color: #2b2b2b; padding: 10px;")
        cl = QVBoxLayout(controls)
        
        cl.addWidget(QLabel("<h2>4. Sampling & Quantization</h2>"))
        cl.addWidget(QLabel("Reduce spatial resolution (Sampling) or gray levels (Quantization) to see artifacts."))
        cl.addSpacing(20)
        
        # Sampling Slider
        cl.addWidget(QLabel("Spatial Sampling (N)"))
        self.sl_samp = QSlider(Qt.Horizontal)
        self.sl_samp.setRange(16, 512)
        self.sl_samp.setValue(512)
        self.sl_samp.setTickInterval(32)
        self.sl_samp.valueChanged.connect(self.update_viz)
        cl.addWidget(self.sl_samp)
        self.lbl_samp = QLabel("512 x 512")
        self.lbl_samp.setAlignment(Qt.AlignCenter)
        cl.addWidget(self.lbl_samp)
        
        cl.addSpacing(20)
        
        # Quantization Slider (Bits)
        cl.addWidget(QLabel("Quantization (k bits)"))
        self.sl_quant = QSlider(Qt.Horizontal)
        self.sl_quant.setRange(1, 8)
        self.sl_quant.setValue(8)
        self.sl_quant.setTickPosition(QSlider.TicksBelow)
        self.sl_quant.valueChanged.connect(self.update_viz)
        cl.addWidget(self.sl_quant)
        self.lbl_quant = QLabel("8 bits (256 levels)")
        self.lbl_quant.setAlignment(Qt.AlignCenter)
        cl.addWidget(self.lbl_quant)
        
        cl.addStretch()
        layout.addWidget(controls)
        
        # -- Visualization --
        # Splitter to show Before/After or just Result
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #000; border: 1px solid #444;")
        
        layout.addWidget(self.display_label, 1)
        
        self.update_viz()

    def generate_source_image(self):
        # Create a synthetic 512x512 numpy image
        # Continuous gradient background + A geometric shape
        x = np.linspace(0, 1, self.source_size)
        y = np.linspace(0, 1, self.source_size)
        xv, yv = np.meshgrid(x, y)
        
        # Gradient
        img = xv * 255
        
        # Circle in center (High contrast)
        res = self.source_size
        cx, cy = res//2, res//2
        r = res//3
        mask = (xv-0.5)**2 + (yv-0.5)**2 < (0.33)**2
        
        img[mask] = 255 - img[mask] # Invert gradient in circle
        
        return img.astype(np.uint8)

    def update_viz(self):
        # Get Params
        # Sampling: Snap to powers of 2 for cleaner visuals or just simple stepping
        s_val = self.sl_samp.value()
        # Ensure it's somewhat discrete
        s_val = int(s_val / 4) * 4 
        if s_val < 4: s_val = 4
        self.sampling_n = s_val
        
        k_val = self.sl_quant.value()
        self.quant_k = k_val
        
        self.lbl_samp.setText(f"{self.sampling_n} x {self.sampling_n}")
        levels = 2**k_val
        self.lbl_quant.setText(f"{k_val} bits ({levels} levels)")
        
        # Process Image
        processed = self.process_image()
        
        # Display
        h, w = processed.shape
        qt_img = QImage(processed.data, w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qt_img)
        
        # Scale up to fit view if sample is small (Nearest neighbor to show pixels)
        disp_h = self.display_label.height()
        if disp_h < 100: disp_h = 512
        
        self.display_label.setPixmap(pix.scaled(disp_h, disp_h, Qt.KeepAspectRatio, Qt.FastTransformation))

    def process_image(self):
        # 1. Sampling (Resize Down then Up)
        src = self.source_img
        
        # Simple Nearest Neighbor Logic using slicing
        # Step size
        step = self.source_size / self.sampling_n
        
        # Indices
        idx = (np.arange(self.sampling_n) * step).astype(int)
        
        # Downsample
        sampled = src[idx, :][:, idx]
        
        # 2. Quantization
        # Reduce levels. 
        # e.g. 256 levels -> 4 levels (0, 85, 170, 255)
        levels = 2 ** self.quant_k
        factor = 256 / levels
        
        quantized = (np.floor(sampled / factor) * factor).astype(np.uint8)
        
        # If we really want to simulate false contouring we usually display just the small image blown up,
        # but to match the "source size" for comparison, we could repeat values.
        # But QPixmap.scaled(FastTransformation) does nearest neighbor for us visually.
        
        return quantized
    
    def resizeEvent(self, event):
        self.update_viz()
        super().resizeEvent(event)
