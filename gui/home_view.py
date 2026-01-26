from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QPushButton, QFrame, QScrollArea
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

class ModuleCard(QFrame):
    clicked = Signal(int)

    def __init__(self, module_id, title, description, icon_text="👁️"):
        super().__init__()
        self.module_id = module_id
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border-radius: 10px;
                border: 1px solid #3d3d3d;
            }
            QFrame:hover {
                background-color: #353535;
                border: 1px solid #4d4d4d;
            }
            QLabel {
                color: #e0e0e0;
                border: none;
                background-color: transparent;
            }
        """)

        layout = QVBoxLayout(self)
        
        # Icon / Thumbnail Placeholder
        self.icon_label = QLabel(icon_text)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFont(QFont("Segoe UI", 32))
        layout.addWidget(self.icon_label)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        # Description
        self.desc_label = QLabel(description)
        self.desc_label.setFont(QFont("Segoe UI", 10))
        self.desc_label.setStyleSheet("color: #aaaaaa;")
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)

    def mousePressEvent(self, event):
        self.clicked.emit(self.module_id)

class HomeView(QWidget):
    module_selected = Signal(int)

    def __init__(self):
        super().__init__()
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        # Header
        header = QLabel("Digital Image Processing Fundamentals")
        header.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #ffffff; margin-bottom: 20px;")
        main_layout.addWidget(header)

        sub_header = QLabel("Chapter 2: Interactive Learning System")
        sub_header.setFont(QFont("Segoe UI", 16))
        sub_header.setAlignment(Qt.AlignCenter)
        sub_header.setStyleSheet("color: #cccccc;")
        main_layout.addWidget(sub_header)

        # Grid of Modules
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(20)

        modules = [
            (1, "Elements of Visual Perception", "Eye structure, brightness adaptation, and illusions.", "👁️"),
            (2, "Light & EM Spectrum", "Wavelength, frequency, and visible light.", "🌈"),
            (3, "Image Sensing & Acquisition", " sensors, sampling, and digitization pipelines.", "📷"),
            (4, "Sampling & Quantization", "Spatial and intensity resolution effects.", "🔢"),
            (5, "Pixel Relationships", "Neighbors, adjacency, connectivity, and distance.", "🕸️"),
            (6, "Mathematical Tools", "Matrix operations, logic, and arithmetic.", "➕")
        ]

        row, col = 0, 0
        for mod_id, title, desc, icon in modules:
            card = ModuleCard(mod_id, title, desc, icon)
            card.clicked.connect(self.module_selected.emit)
            grid.addWidget(card, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1

        scroll.setWidget(container)
        main_layout.addWidget(scroll)
