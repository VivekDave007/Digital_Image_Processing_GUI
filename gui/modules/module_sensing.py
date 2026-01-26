from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QTabWidget, QSlider, QProgressBar)
from PySide6.QtCore import Qt, QTimer, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

class SensorSimulator(QWidget):
    def __init__(self, mode="single"):
        super().__init__()
        self.mode = mode # single, strip, array
        self.resolution = 20 # 20x20 grid
        self.is_capturing = False
        self.progress = 0
        self.scan_x = 0
        self.scan_y = 0
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.game_loop)
        
    def start_capture(self):
        self.is_capturing = True
        self.progress = 0
        self.scan_x = 0
        self.scan_y = 0
        self.timer.start(30) # 30ms

    def game_loop(self):
        # Update simulation state based on mode
        if self.mode == "single":
            # Move pixel by pixel
            self.scan_x += 1
            if self.scan_x >= self.resolution:
                self.scan_x = 0
                self.scan_y += 1
            if self.scan_y >= self.resolution:
                self.finish()
                
        elif self.mode == "strip":
            # Move row by row
            self.scan_y += 1
            if self.scan_y >= self.resolution:
                self.finish()
                
        elif self.mode == "array":
            # Instant
            self.scan_x = self.resolution
            self.scan_y = self.resolution
            self.finish()
            
        self.update()

    def finish(self):
        self.is_capturing = False
        self.timer.stop()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        
        # Define Draw Area (Square in middle)
        size = min(w, h) - 40
        ox = (w - size) // 2
        oy = (h - size) // 2
        
        pixel_size = size / self.resolution
        
        # 1. Draw "Scene" (Ghosted)
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        for y in range(self.resolution):
            for x in range(self.resolution):
                # Simple checker pattern as "Subject"
                is_white = (x + y) % 2 == 0
                col = QColor(60, 60, 60) if is_white else QColor(30, 30, 30)
                painter.fillRect(int(ox + x*pixel_size), int(oy + y*pixel_size), int(pixel_size), int(pixel_size), col)
                painter.drawRect(int(ox + x*pixel_size), int(oy + y*pixel_size), int(pixel_size), int(pixel_size))

        # 2. Draw "Captured" Pixels
        # Logic depends on mode progress
        
        rows_done = self.scan_y
        
        if self.mode == "array" and not self.is_capturing and self.scan_y > 0:
            rows_done = self.resolution # Show full if done

        # Draw fully captured rows
        for y in range(rows_done):
            for x in range(self.resolution):
                self.draw_pixel(painter, ox, oy, x, y, pixel_size)
        
        # Draw partial row for single sensor
        if self.mode == "single" and self.is_capturing:
            for x in range(self.scan_x):
                self.draw_pixel(painter, ox, oy, x, self.scan_y, pixel_size)
                
        # 3. Draw "Sensor/Laser" Indicator
        if self.is_capturing:
            painter.setPen(QColor(255, 0, 0))
            painter.setBrush(Qt.NoBrush)
            
            if self.mode == "single":
                rx = ox + self.scan_x * pixel_size
                ry = oy + self.scan_y * pixel_size
                painter.drawRect(int(rx), int(ry), int(pixel_size), int(pixel_size))
                painter.drawLine(int(rx)+int(pixel_size/2), int(ry), int(rx)+int(pixel_size/2), int(ry)-10) # Laser beam
                
            elif self.mode == "strip":
                ry = oy + self.scan_y * pixel_size
                painter.drawRect(int(ox), int(ry), int(size), int(pixel_size))
                
            elif self.mode == "array":
                painter.setBrush(QColor(255, 255, 255, 100)) # Flash
                painter.drawRect(ox, oy, size, size)

    def draw_pixel(self, painter, ox, oy, x, y, size):
        # Captured Pattern (Bright Checkerboard)
        is_white = (x + y) % 2 == 0
        col = QColor(200, 200, 200) if is_white else QColor(50, 50, 100)
        painter.fillRect(int(ox + x*size), int(oy + y*size), int(size), int(size), col)


class ModuleSensing(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("<h2>3. Sensing & Acquisition</h2>"))
        layout.addWidget(QLabel("Different sensor geometries require different mechanical setups."))
        
        # Tabs
        self.tabs = QTabWidget()
        
        # Single Sensor Tab
        self.sim_single = SensorSimulator("single")
        self.tabs.addTab(self.create_tab(self.sim_single), "Single Sensor (Laser/Drum)")
        
        # Strip Sensor Tab
        self.sim_strip = SensorSimulator("strip")
        self.tabs.addTab(self.create_tab(self.sim_strip), "Sensor Strip (Flatbed)")
        
        # Array Sensor Tab
        self.sim_array = SensorSimulator("array")
        self.tabs.addTab(self.create_tab(self.sim_array), "Sensor Array (Camera)")
        
        layout.addWidget(self.tabs)

    def create_tab(self, sim_widget):
        container = QWidget()
        l = QVBoxLayout(container)
        
        # Controls
        ctrl = QHBoxLayout()
        btn = QPushButton("Trigger Capture")
        btn.clicked.connect(sim_widget.start_capture)
        ctrl.addWidget(btn)
        ctrl.addStretch()
        
        l.addLayout(ctrl)
        l.addWidget(sim_widget)
        return container
