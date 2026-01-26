import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider, QPushButton, QGroupBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor

class ModuleMath(QWidget):
    def __init__(self):
        super().__init__()
        
        self.img_size = 256
        self.img_a = self.generate_image_a()
        self.img_b = self.generate_image_b()
        self.result_img = np.zeros_like(self.img_a)
        
        self.noise_frames = []
        self.avg_count = 0
        self.is_averaging = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.averaging_step)
        
        layout = QHBoxLayout(self)
        
        # -- Controls --
        ctrl = QWidget()
        ctrl.setFixedWidth(280)
        ctrl.setStyleSheet("background-color: #2b2b2b; padding: 10px; border-radius: 8px;")
        cl = QVBoxLayout(ctrl)
        
        cl.addWidget(QLabel("<h2>6. Mathematical Tools</h2>"))
        cl.addWidget(QLabel("Operations performed pixel-by-pixel."))
        
        cl.addSpacing(20)
        cl.addWidget(QLabel("<b>Operation</b>"))
        self.combo_op = QComboBox()
        self.combo_op.addItems([
            "Addition (A + B)", 
            "Subtraction (B - A)", 
            "Multiplication (A * B)", 
            "Division (B / A)",
            "Logical AND (A & B)",
            "Logical OR (A | B)"
        ])
        self.combo_op.currentIndexChanged.connect(self.perform_op)
        cl.addWidget(self.combo_op)
        
        # Specific Demos
        cl.addSpacing(30)
        group_noise = QGroupBox("Noise Reduction Demo (Average)")
        gl = QVBoxLayout(group_noise)
        self.btn_noise = QPushButton("Start Averaging (100 frames)")
        self.btn_noise.clicked.connect(self.start_averaging)
        gl.addWidget(self.btn_noise)
        self.lbl_noise = QLabel("Frames: 0")
        gl.addWidget(self.lbl_noise)
        cl.addWidget(group_noise)

        cl.addStretch()
        layout.addWidget(ctrl)
        
        # -- Visualization --
        viz = QWidget()
        vl = QVBoxLayout(viz)
        
        # Top Row: Inputs
        top_row = QHBoxLayout()
        self.lbl_a = self.create_img_label("Image A (Background)")
        self.lbl_b = self.create_img_label("Image B (Object)")
        top_row.addWidget(self.lbl_a)
        top_row.addWidget(self.lbl_b)
        
        vl.addLayout(top_row)
        
        # Bottom Row: Result
        self.lbl_res = self.create_img_label("Result")
        self.lbl_res.setFixedSize(300, 300) # Make result bigger
        vl.addWidget(self.lbl_res, alignment=Qt.AlignCenter)
        
        layout.addWidget(viz, 1)
        
        # Init Visualization
        self.show_img(self.lbl_a, self.img_a)
        self.show_img(self.lbl_b, self.img_b)
        self.perform_op()

    def create_img_label(self, title):
        w = QWidget()
        l = QVBoxLayout(w)
        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_img = QLabel()
        lbl_img.setAlignment(Qt.AlignCenter)
        lbl_img.setStyleSheet("background-color: #000; border: 1px solid #555;")
        lbl_img.setFixedSize(200, 200)
        l.addWidget(lbl_title)
        l.addWidget(lbl_img)
        w.img_lbl = lbl_img # Store ref
        return w

    def show_img(self, widget_container, numpy_img):
        h, w = numpy_img.shape
        # Normalize for display if needed (float to uint8)
        disp = numpy_img.astype(np.uint8)
        qt_img = QImage(disp.data, w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qt_img)
        widget_container.img_lbl.setPixmap(pix.scaled(widget_container.img_lbl.size(), Qt.KeepAspectRatio))

    def generate_image_a(self):
        # Background: Gradient
        x = np.linspace(0, 1, self.img_size)
        xv, yv = np.meshgrid(x, x)
        return (xv * 100).astype(np.float32) # Dark gradient

    def generate_image_b(self):
        # Background + Object
        img = self.generate_image_a()
        # Add Square Object
        ox, oy = 80, 80
        size = 100
        img[oy:oy+size, ox:ox+size] = 200 # Bright object
        return img

    def perform_op(self):
        op_idx = self.combo_op.currentIndex()
        
        A = self.img_a
        B = self.img_b
        
        res = np.zeros_like(A)
        
        if op_idx == 0: # Add
            res = A + B
            res = np.clip(res, 0, 255)
        elif op_idx == 1: # Sub (B - A) -> Should isolate object
            res = B - A
            res = np.clip(res, 0, 255) # Negatives become 0
        elif op_idx == 2: # Mult
            # Normalize to 0..1 then mult
            res = (A/255.0) * (B/255.0) * 255.0
        elif op_idx == 3: # Div
            # Avoid div zero
            safe_a = A.copy()
            safe_a[safe_a < 1] = 1
            res = (B / safe_a) * 50 # Scale up
            res = np.clip(res, 0, 255)
        elif op_idx == 4: # AND (Binary)
            bin_a = (A > 50).astype(np.uint8) * 255
            bin_b = (B > 150).astype(np.uint8) * 255
            res = np.bitwise_and(bin_a, bin_b)
        elif op_idx == 5: # OR
            bin_a = (A > 50).astype(np.uint8) * 255
            bin_b = (B > 150).astype(np.uint8) * 255
            res = np.bitwise_or(bin_a, bin_b)

        self.show_img(self.lbl_res, res)

    def start_averaging(self):
        self.avg_count = 0
        self.sum_buffer = np.zeros_like(self.img_a, dtype=np.float32)
        self.timer.start(50)
        self.btn_noise.setEnabled(False)

    def averaging_step(self):
        self.avg_count += 1
        
        # Add Gaussian Noise to Image A
        noise = np.random.normal(0, 50, self.img_a.shape)
        noisy_frame = self.img_a + noise
        noisy_frame = np.clip(noisy_frame, 0, 255)
        
        # Accumulate
        self.sum_buffer += noisy_frame
        
        # Calculate Current Average
        current_avg = self.sum_buffer / self.avg_count
        
        # Show Noisy Input in A
        self.show_img(self.lbl_a, noisy_frame)
        self.lbl_a.findChild(QLabel).setText("Image A (Noisy Input)")
        
        # Show Running Average in Result
        self.show_img(self.lbl_res, current_avg)
        
        self.lbl_noise.setText(f"Averaging Frame: {self.avg_count}")
        
        if self.avg_count >= 100:
            self.timer.stop()
            self.btn_noise.setEnabled(True)
            self.lbl_a.findChild(QLabel).setText("Image A (Background)")
