import math
import collections
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, QButtonGroup, QComboBox, QCheckBox)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush

class GridCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.rows = 10
        self.cols = 10
        self.grid_data = [[0 for _ in range(10)] for _ in range(10)] # 0=Empty, 1=Obstacle
        
        self.mode = "neighbors" # neighbors, distance, path
        self.metric = "cityblock" # euclidean, cityblock, chessboard
        self.connectivity = "4" # 4, 8
        
        self.selected_p = None
        self.selected_q = None
        
        self.setMouseTracking(True)
        self.hover_pos = None

    def set_params(self, mode, metric, connectivity):
        self.mode = mode
        self.metric = metric
        self.connectivity = connectivity
        self.update()

    def mouseMoveEvent(self, event):
        w = self.width() / self.cols
        h = self.height() / self.rows
        c = int(event.position().x() / w)
        r = int(event.position().y() / h)
        
        if 0 <= c < self.cols and 0 <= r < self.rows:
            self.hover_pos = (r, c)
        else:
            self.hover_pos = None
        self.update()

    def mousePressEvent(self, event):
        if not self.hover_pos: return
        
        if self.mode == "path":
            if self.selected_p is None:
                self.selected_p = self.hover_pos
            elif self.selected_q is None:
                self.selected_q = self.hover_pos
            else:
                self.selected_p = self.hover_pos
                self.selected_q = None
        else:
            self.selected_p = self.hover_pos
            
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        wfile = self.width()
        hfile = self.height()
        cw = wfile / self.cols
        ch = hfile / self.rows
        
        font = QFont("Segoe UI", 10)
        painter.setFont(font)

        # Draw Grid
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * cw
                y = r * ch
                
                # Base Color
                painter.setBrush(QColor(30, 30, 30))
                painter.setPen(QColor(60, 60, 60))
                painter.drawRect(x, y, cw, ch)
                
                # Logic Visualization Overlay
                self.draw_overlay(painter, r, c, x, y, cw, ch)

    def draw_overlay(self, p, r, c, x, y, cw, ch):
        # Highlight Selected
        if (r, c) == self.selected_p:
            p.setBrush(QColor(0, 100, 255))
            p.drawRect(x, y, cw, ch)
            p.setPen(Qt.white)
            p.drawText(x+5, y+20, "P")

        if (r, c) == self.selected_q:
            p.setBrush(QColor(255, 100, 0))
            p.drawRect(x, y, cw, ch)
            p.setPen(Qt.white)
            p.drawText(x+5, y+20, "Q")
            
        # Neighbors Mode
        if self.mode == "neighbors" and self.hover_pos:
            hr, hc = self.hover_pos
            is_hover = (r == hr and c == hc)
            
            # Draw Center
            if is_hover:
                p.setBrush(QColor(0, 255, 0)) # Green
                p.drawRect(x, y, cw, ch)
                
            # Check Neighbor
            dx = abs(c - hc)
            dy = abs(r - hr)
            dist = dx + dy
            
            is_n4 = (dist == 1)
            is_nd = (dx == 1 and dy == 1)
            
            if is_n4:
                p.setPen(QPen(QColor(255, 255, 0), 2))
                p.drawText(x+cw/2-10, y+ch/2, "N4")
            if is_nd:
                p.setPen(QPen(QColor(0, 255, 255), 2))
                p.drawText(x+cw/2-10, y+ch/2, "ND")
                
        # Distance Mode
        if self.mode == "distance" and self.selected_p:
            pr, pc = self.selected_p
            d = 0
            if self.metric == "euclidean":
                d = math.sqrt((r-pr)**2 + (c-pc)**2)
                txt = f"{d:.1f}"
            elif self.metric == "cityblock":
                d = abs(r-pr) + abs(c-pc)
                txt = f"{int(d)}"
            elif self.metric == "chessboard":
                d = max(abs(r-pr), abs(c-pc))
                txt = f"{int(d)}"
                
            # Heatmap color
            intensity = min(255, int(d * 20))
            p.setBrush(QColor(intensity, 50, 50, 100))
            p.drawRect(x, y, cw, ch)
            
            p.setPen(Qt.white)
            p.drawText(x+cw/2-10, y+ch/2, txt)

        # Path Mode
        if self.mode == "path" and self.selected_p and self.selected_q:
            # Draw path if on path
            path = self.find_path(self.selected_p, self.selected_q)
            if (r, c) in path:
                p.setBrush(QColor(200, 200, 0, 100))
                p.drawRect(x, y, cw, ch)
                idx = path.index((r, c))
                p.setPen(Qt.white)
                p.drawText(x+cw/2-5, y+ch/2, str(idx))

    def find_path(self, start, end):
        # BFS
        queue = collections.deque([[start]])
        visited = set([start])
        
        while queue:
            path = queue.popleft()
            curr = path[-1]
            if curr == end:
                return path
            
            r, c = curr
            neighbors = []
            
            # Generate candidates based on connectivity
            # 4-conn: (r+1,c), (r-1,c), (r,c+1), (r,c-1)
            candidates = [
                (r+1, c), (r-1, c), (r, c+1), (r, c-1)
            ]
            if self.connectivity == "8":
                candidates += [
                    (r+1, c+1), (r-1, c-1), (r+1, c-1), (r-1, c+1)
                ]
            
            for nr, nc in candidates:
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        new_path = list(path)
                        new_path.append((nr, nc))
                        queue.append(new_path)
        return []

class ModuleRelationships(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        
        # Controls
        ctrl = QWidget()
        ctrl.setFixedWidth(250)
        cl = QVBoxLayout(ctrl)
        
        cl.addWidget(QLabel("<h2>5. Relationships</h2>"))
        cl.addWidget(QLabel("Visualize Neighbors, Distance metrics, and Connectivity."))
        
        # Mode
        cl.addSpacing(20)
        cl.addWidget(QLabel("<b>Input Mode</b>"))
        self.grp_mode = QButtonGroup(self)
        self.r_neigh = QRadioButton("View Neighbors")
        self.r_dist = QRadioButton("Measure Distance")
        self.r_path = QRadioButton("Find Path (BFS)")
        self.r_neigh.setChecked(True)
        
        self.grp_mode.addButton(self.r_neigh, 1)
        self.grp_mode.addButton(self.r_dist, 2)
        self.grp_mode.addButton(self.r_path, 3)
        self.grp_mode.buttonClicked.connect(self.update_canvas)
        
        cl.addWidget(self.r_neigh)
        cl.addWidget(self.r_dist)
        cl.addWidget(self.r_path)
        
        # Metric
        cl.addSpacing(20)
        cl.addWidget(QLabel("<b>Distance Metric</b>"))
        self.combo_metric = QComboBox()
        self.combo_metric.addItems(["Euclidean (L2)", "City-Block (D4)", "Chessboard (D8)"])
        self.combo_metric.currentIndexChanged.connect(self.update_canvas)
        cl.addWidget(self.combo_metric)
        
        # Connectivity
        cl.addSpacing(20)
        cl.addWidget(QLabel("<b>Connectivity</b>"))
        self.combo_conn = QComboBox()
        self.combo_conn.addItems(["4-Connectivity", "8-Connectivity"])
        self.combo_conn.currentIndexChanged.connect(self.update_canvas)
        cl.addWidget(self.combo_conn)

        cl.addStretch()
        layout.addWidget(ctrl)
        
        # Canvas
        self.canvas = GridCanvas()
        layout.addWidget(self.canvas, 1)
        
        self.update_canvas()

    def update_canvas(self):
        mode_map = {1: "neighbors", 2: "distance", 3: "path"}
        mode = mode_map[self.grp_mode.checkedId()]
        
        metric_map = {0: "euclidean", 1: "cityblock", 2: "chessboard"}
        metric = metric_map[self.combo_metric.currentIndex()]
        
        conn_map = {0: "4", 1: "8"}
        conn = conn_map[self.combo_conn.currentIndex()]
        
        self.canvas.set_params(mode, metric, conn)
