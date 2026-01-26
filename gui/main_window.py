from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QStackedWidget, QToolBar, QLabel, QPushButton
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QIcon

from .home_view import HomeView

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digital Image Processing - Chapter 2 Educational System")
        self.resize(1280, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QToolBar {
                background-color: #252526;
                border-bottom: 1px solid #3d3d3d;
                padding: 5px;
            }
            QLabel {
                color: #cccccc;
            }
        """)

        # Central Widget & Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar / Breadcrumb area
        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        
        # Home Action
        self.home_action = QAction("🏠 Home", self)
        self.home_action.triggered.connect(self.go_home)
        self.toolbar.addAction(self.home_action)
        
        # Spacer
        dummy = QWidget()
        dummy.setFixedWidth(20)
        self.toolbar.addWidget(dummy)
        
        # Title Label in Toolbar
        self.title_label = QLabel("Dashboard")
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.toolbar.addWidget(self.title_label)

        # Stacked Widget for Views
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # Initialize Home View
        self.home_view = HomeView()
        self.home_view.module_selected.connect(self.load_module)
        self.stack.addWidget(self.home_view)
        
        # Placeholders for Modules (will be lazy loaded)
        self.modules = {} 

    def go_home(self):
        self.stack.setCurrentWidget(self.home_view)
        self.title_label.setText("Dashboard")
        self.home_action.setEnabled(False) # Already home

    def load_module(self, module_id):
        # In a real app, we would lazy load the module classes here
        # For now, we will just show a placeholder
        
        if module_id not in self.modules:
            if module_id == 1:
                from .modules.module_perception import ModulePerception
                self.modules[module_id] = ModulePerception()
            elif module_id == 2:
                from .modules.module_spectrum import ModuleSpectrum
                self.modules[module_id] = ModuleSpectrum()
            elif module_id == 3:
                from .modules.module_sensing import ModuleSensing
                self.modules[module_id] = ModuleSensing()
            elif module_id == 4:
                from .modules.module_sampling import ModuleSampling
                self.modules[module_id] = ModuleSampling()
            elif module_id == 5:
                from .modules.module_relationships import ModuleRelationships
                self.modules[module_id] = ModuleRelationships()
            elif module_id == 6:
                from .modules.module_math import ModuleMath
                self.modules[module_id] = ModuleMath()
            else:
                placeholder = self.create_placeholder(module_id)
                self.modules[module_id] = placeholder
                self.stack.addWidget(placeholder)
                
            if module_id in [1, 2, 3, 4, 5, 6]:
                 # Add the newly created module to the stack
                 self.stack.addWidget(self.modules[module_id])
        
        self.stack.setCurrentWidget(self.modules[module_id])
        self.title_label.setText(f"Module {module_id}")
        self.home_action.setEnabled(True)

    def create_placeholder(self, module_id):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(f"Module {module_id} Content Coming Soon...")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 24px; color: #666;")
        layout.addWidget(label)
        return widget
