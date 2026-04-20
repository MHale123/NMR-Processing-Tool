import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class LauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NMR Analysis Suite")
        self.setFixedSize(480, 280)

        self._t1_window  = None
        self._csp_window = None

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(40, 32, 40, 32)

        # Title
        title = QLabel("NMR Analysis Suite")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Bruker data analysis tools")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        # Module buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(16)

        # T1 module
        t1_col = QVBoxLayout()
        btn_t1 = QPushButton("T1 Relaxation")
        btn_t1.setFixedHeight(60)
        btn_t1.setStyleSheet("font-size: 14px; font-weight: bold;")
        btn_t1.clicked.connect(self.open_t1)
        t1_desc = QLabel("Inversion recovery\nT1 fitting and analysis")
        t1_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t1_desc.setStyleSheet("color: gray; font-size: 11px;")
        t1_col.addWidget(btn_t1)
        t1_col.addWidget(t1_desc)
        btn_layout.addLayout(t1_col)

        # CSP module
        csp_col = QVBoxLayout()
        btn_csp = QPushButton("Chemical Shift Pertubation")
        btn_csp.setFixedHeight(60)
        btn_csp.setStyleSheet("font-size: 14px; font-weight: bold;")
        btn_csp.clicked.connect(self.open_csp)
        csp_desc = QLabel("Create waterfall plots")
        csp_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        csp_desc.setStyleSheet("color: gray; font-size: 11px;")
        csp_col.addWidget(btn_csp)
        csp_col.addWidget(csp_desc)
        btn_layout.addLayout(csp_col)

        layout.addLayout(btn_layout)

    def open_t1(self):
        from app.main_window import MainWindow
        self._t1_window = MainWindow()
        self._t1_window.show()

    def open_csp(self):
        from app.csp_window import CSPWindow
        self._csp_window = CSPWindow()
        self._csp_window.show()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("NMR Analysis Suite")
    app.setStyle("Fusion")

    launcher = LauncherWindow()
    launcher.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()