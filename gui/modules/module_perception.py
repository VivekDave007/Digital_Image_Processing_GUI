import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QRadioButton, QButtonGroup, QCheckBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush

class BrightnessGraph(QWidget):
    """
    Custom widget to draw the Weber-Fechner brightness adaptation curve.
    Shows Log(Intensity) vs Subjective Brightness.
    """
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(200)
        self.illumination_log = 1.0 # Current x-axis position
        self.adaptation_level = 1.0 # Current y-axis bias

    def set_illumination(self, value_log):
        self.illumination_log = value_log
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Background
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))
        
        # Axes
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.drawLine(40, h-30, w-20, h-30) # X-axis
        painter.drawLine(40, h-30, 40, 20)     # Y-axis
        
        # Labels
        painter.setPen(QColor(150, 150, 150))
        painter.drawText(w-100, h-10, "Log(Intensity)")
        painter.drawText(10, 15, "Brightness")

        # Draw Curve: A typical S-curve or logarithmic segment shifting with adaptation
        # For this demo, we draw the "range of subjective brightness" at current adaptation
        # The visual system adapts to a mean level Ba.
        
        # We visualize the "curve" as a line of 45 degrees for simplicity of the lesson, 
        # but constrained within a operational window.
        
        center_x = 40 + (w-60) * ((self.illumination_log + 2) / 6.0) # Map -2..4 to Screen
        center_y = (h-30) - (h-60) * 0.5 # Fixed middle for visualization context
        
        # Draw the "Active Sensitivity Range" (The curve section)
        path_pen = QPen(QColor(0, 255, 128), 3)
        painter.setPen(path_pen)
        
        # Draw a small segment representing the current dynamic range
        segment_len = 60
        p1_x = center_x - segment_len
        p1_y = center_y + segment_len
        p2_x = center_x + segment_len
        p2_y = center_y - segment_len
        
        painter.drawLine(int(p1_x), int(p1_y), int(p2_x), int(p2_y))
        
        # Draw dashed line to axes to show operating point
        painter.setPen(QPen(QColor(255, 255, 0), 1, Qt.DashLine))
        painter.drawLine(int(center_x), int(center_y), int(center_x), h-30)
        painter.drawLine(int(center_x), int(center_y), 40, int(center_y))
        
        painter.drawText(int(center_x)+5, h-35, f"I={10**self.illumination_log:.2f}")


class SceneSimulator(QWidget):
    """
    Renders a scene that changes based on simulation parameters.
    e.g. Fades to grayscale in low light (Rods).
    """
    def __init__(self):
        super().__init__()
        self.is_day = True
        self.illumination = 1.0 # 0.0 to 1.0
        self.show_mach_bands = False
        
    def set_params(self, illumination, mach_bands):
        self.illumination = illumination
        self.show_mach_bands = mach_bands
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        
        # If Mach Bands requested, draw gradient pattern
        if self.show_mach_bands:
            self.draw_mach_bands(painter, w, h)
            return

        # Otherwise draw a "Flower" scene
        # Background
        bg_intensity = int(255 * self.illumination)
        painter.fillRect(0, 0, w, h, QColor(bg_intensity, bg_intensity, bg_intensity))
        
        # Draw a simple "Flower"
        # If Illumination is very low (< 0.2), we are in Scotopic (Rod) vision -> Grayscale only & Blurry
        is_scotopic = self.illumination < 0.2
        
        if is_scotopic:
           painter.setPen(Qt.NoPen)
           painter.setBrush(QColor(100, 100, 100)) # Gray flower
        else:
           painter.setPen(Qt.NoPen)
           painter.setBrush(QColor(255, 50, 50)) # Red flower

        # Center
        cx, cy = w//2, h//2
        painter.drawEllipse(cx-50, cy-50, 100, 100)
        
        # Center dot
        if is_scotopic:
            painter.setBrush(QColor(50, 50, 50))
        else:
            painter.setBrush(QColor(255, 255, 0))
        painter.drawEllipse(cx-20, cy-20, 40, 40)
        
        # Overlay text
        painter.setPen(QColor(0, 255, 255))
        painter.setFont(QFont("Segoe UI", 12))
        mode = "SCOTOPIC (Rods)" if is_scotopic else "PHOTOPIC (Cones)"
        painter.drawText(20, 30, f"Vision Mode: {mode}")

    def draw_mach_bands(self, painter, w, h):
        # Draw vertical stripes with varying intensity to show overshoot illusion
        steps = 10
        sw = w / steps
        for i in range(steps):
            val = int(255 * (i/steps) * self.illumination)
            painter.fillRect(int(i*sw), 0, int(sw)+1, h, QColor(val, val, val))
        
        painter.setPen(QColor(255, 0, 0))
        painter.drawText(20, 30, "Mach Bands: Notice the 'overshoot' brightness at edges")

class ModulePerception(QWidget):
    def __init__(self):
        super().__init__()
        
        layout = QHBoxLayout(self)
        
        # -- Controls Panel --
        controls_panel = QWidget()
        controls_panel.setFixedWidth(300)
        controls_panel.setStyleSheet("background-color: #2b2b2b; border-radius: 10px;")
        c_layout = QVBoxLayout(controls_panel)
        
        c_layout.addWidget(QLabel("<h2>1. Visual Perception</h2>"))
        c_layout.addWidget(QLabel("Adjust illumination to see how the eye adapts (Photopic vs Scotopic)."))
        
        c_layout.addSpacing(20)
        c_layout.addWidget(QLabel("Illumination Level (Log)"))
        self.slider_illum = QSlider(Qt.Horizontal)
        self.slider_illum.setRange(-200, 400) # Representing log scale -2.00 to 4.00
        self.slider_illum.setValue(100)
        self.slider_illum.valueChanged.connect(self.update_sim)
        c_layout.addWidget(self.slider_illum)
        
        c_layout.addSpacing(20)
        self.chk_mach = QCheckBox("Show Mach Bands")
        self.chk_mach.stateChanged.connect(self.update_sim)
        c_layout.addWidget(self.chk_mach)
        
        c_layout.addStretch()
        
        # -- Visualization Area --
        viz_panel = QWidget()
        v_layout = QVBoxLayout(viz_panel)
        
        self.scene_sim = SceneSimulator()
        v_layout.addWidget(self.scene_sim, 2)
        
        self.graph = BrightnessGraph()
        v_layout.addWidget(self.graph, 1)
        
        layout.addWidget(controls_panel)
        layout.addWidget(viz_panel)
        
        # Init
        self.update_sim()

    def update_sim(self):
        # Slider value -200..400 -> -2.0 .. 4.0 log scale
        val = self.slider_illum.value() / 100.0
        
        # Map log value to linear 0..1 for scene brightness (clamped for viz)
        # linear approximation for scene brightness
        # 4.0 is super bright, -2.0 is pitch dark
        # Normal range ~0 to 1
        linear_val = (val + 2.0) / 6.0 # Normalize -2..4 to 0..1
        linear_val = max(0.0, min(1.0, linear_val))
        
        self.graph.set_illumination(val)
        self.scene_sim.set_params(linear_val, self.chk_mach.isChecked())
