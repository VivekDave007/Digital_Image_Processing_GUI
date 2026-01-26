import math
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QFrame
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QFont, QLinearGradient, QBrush, QPen

class SpectrumBar(QWidget):
    """
    Custom widget visualizing the EM spectrum from Gamma (10^20 Hz) to Radio (10^4 Hz).
    """
    pos_changed = Signal(float)  # Emits log10(frequency)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.min_freq_log = 4.0   # Radio
        self.max_freq_log = 22.0  # Gamma
        self.current_freq_log = 14.5 # Visible-ish (Green)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Draw Background Track
        painter.fillRect(20, 40, w-40, 20, QColor(50, 50, 50))
        
        # Draw "Visible Light" highlighted section
        # Visible is approx 4x10^14 to 8x10^14 Hz -> Log ~14.6 to ~14.9
        # It's a tiny sliver in the grand scheme, we represent it slightly wider for UX
        vis_start_log = 14.5
        vis_end_log = 15.0
        
        start_x = self.log_to_x(vis_start_log, w)
        end_x = self.log_to_x(vis_end_log, w)
        
        # Rainbow Gradient for Visible
        grad = QLinearGradient(start_x, 0, end_x, 0)
        grad.setColorAt(0.0, Qt.red)
        grad.setColorAt(0.2, Qt.yellow)
        grad.setColorAt(0.4, Qt.green)
        grad.setColorAt(0.6, Qt.cyan)
        grad.setColorAt(0.8, Qt.blue)
        grad.setColorAt(1.0, Qt.magenta)
        
        painter.fillRect(int(start_x), 40, int(end_x - start_x), 20, QBrush(grad))
        
        # Labels for bands
        bands = [
            ("Radio", 6, 12), ("Microwave", 10, 11), ("Infrared", 13, 14),
            ("Visible", 14.7, 14.8), ("UV", 15.5, 16), ("X-Ray", 18, 19), ("Gamma", 21, 22)
        ]
        
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor(200, 200, 200))
        for name, log_val, display_priority in bands:
            x = self.log_to_x(log_val, w)
            painter.drawText(int(x)-20, 30, name)
            painter.drawLine(int(x), 40, int(x), 60)

        # Draw Slider Handle
        hx = self.log_to_x(self.current_freq_log, w)
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(hx)-10, 40-5, 20, 30)

    def log_to_x(self, log_val, w):
        # Linearly map log freq to x pixel
        # Low Freq (Radio) on Right? Or Left? 
        # Physics usually Higher Freq on Right.
        pct = (log_val - self.min_freq_log) / (self.max_freq_log - self.min_freq_log)
        return 20 + pct * (w - 40)

    def x_to_log(self, x, w):
        pct = (x - 20) / (w - 40)
        pct = max(0.0, min(1.0, pct))
        return self.min_freq_log + pct * (self.max_freq_log - self.min_freq_log)

    def mousePressEvent(self, event):
        self.update_pos(event.position().x())

    def mouseMoveEvent(self, event):
        self.update_pos(event.position().x())
        
    def update_pos(self, x):
        self.current_freq_log = self.x_to_log(x, self.width())
        self.update()
        self.pos_changed.emit(self.current_freq_log)

class ModuleSpectrum(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # -- Header --
        layout.addWidget(QLabel("<h2>2. The Electromagnetic Spectrum</h2>"))
        layout.addWidget(QLabel("Slide through the spectrum to see the relationship between Frequency ($\nu$), Wavelength ($\lambda$), and Energy ($E$)."))
        
        # -- Spectrum Interaction --
        self.spectrum_bar = SpectrumBar()
        self.spectrum_bar.pos_changed.connect(self.update_stats)
        layout.addWidget(self.spectrum_bar)
        
        # -- Info Panel --
        info_layout = QHBoxLayout()
        
        # Left: Math Box
        math_frame = QFrame()
        math_frame.setStyleSheet("background-color: #2b2b2b; border-radius: 10px; padding: 10px;")
        mf_layout = QVBoxLayout(math_frame)
        self.lbl_freq = QLabel("Frequency (v): ")
        self.lbl_wave = QLabel("Wavelength (λ): ")
        self.lbl_energy = QLabel("Energy (E): ")
        self.lbl_category = QLabel("Band: ")
        
        font = QFont("Consolas", 12)
        for l in [self.lbl_freq, self.lbl_wave, self.lbl_energy, self.lbl_category]:
            l.setFont(font)
            mf_layout.addWidget(l)
            
        mf_layout.addStretch()
        info_layout.addWidget(math_frame, 1)
        
        # Right: Visualization Box (What does seeing this look like?)
        self.viz_frame = QLabel()
        self.viz_frame.setAlignment(Qt.AlignCenter)
        self.viz_frame.setStyleSheet("background-color: #000; color: #fff; font-size: 20px; border-radius: 10px;")
        self.viz_frame.setText("Visualization")
        info_layout.addWidget(self.viz_frame, 2)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Init
        self.update_stats(14.7)

    def update_stats(self, log_freq):
        freq = 10**log_freq
        
        # c = 3 x 10^8 m/s
        c = 3.0e8
        wavelength = c / freq
        
        # E = h*v (h approx 4.135 x 10^-15 eV*s)
        h_const = 4.135e-15
        energy = h_const * freq
        
        self.lbl_freq.setText(f"Frequency (ν): {freq:.2e} Hz")
        self.lbl_wave.setText(f"Wavelength (λ): {wavelength:.2e} m")
        self.lbl_energy.setText(f"Energy (E): {energy:.2e} eV")
        
        # Determine Band
        band = "Unknown"
        color = "#ffffff"
        viz_text = "🌌"
        
        if log_freq < 9:
            band = "Radio Waves"
            viz_text = "📡 Radio/TV Signals"
        elif log_freq < 12:
            band = "Microwaves"
            viz_text = "📡 Rader / WiFi"
        elif log_freq < 14.5:
            band = "Infrared"
            viz_text = "🌡️ Thermal Imaging (Heat)"
            color = "#ff4444"
        elif log_freq < 15.0:
            band = "Visible Light"
            viz_text = "🌈 Human Vision (Colors)"
            color = "#00ff00"
        elif log_freq < 17:
            band = "Ultraviolet"
            viz_text = "☀️ Florescence / Sunburn"
            color = "#aa00ff"
        elif log_freq < 20:
            band = "X-Rays"
            viz_text = "🦴 Medical Imaging (Bones)"
        else:
            band = "Gamma Rays"
            viz_text = "☢️ Nuclear Imaging"
            
        self.lbl_category.setText(f"Band: {band}")
        self.lbl_category.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.viz_frame.setText(viz_text)
