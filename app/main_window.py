import sys
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from processing.loader import BrukerLoader
from processing.processor import extract_trajectory

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NMR T1 Processor")
        self.setGeometry(100, 100, 800, 600)

        # Main Layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # UI Elements
        self.label = QLabel("No Data Loaded")
        self.layout.addWidget(self.label)

        self.load_btn = QPushButton("Load Bruker Folder")
        self.load_btn.clicked.connect(self.open_folder)
        self.layout.addWidget(self.load_btn)

        # Matplotlib Canvas
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.layout.addWidget(self.canvas)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Experiment Folder (e.g., '1')")
        if folder:
            self.process_data(folder)

    def process_data(self, folder_path):
        try:
            loader = BrukerLoader(folder_path)
            dic, data = loader.load_processed_data()
            
            if data is not None:
                # Integrate a sample window (we will make this interactive later!)
                mid = data.shape[1] // 2
                traj = extract_trajectory(data, mid - 50, mid + 50)
                
                # Filter out the zeros for the plot
                traj = traj[traj != 0]

                self.ax.clear()
                self.ax.plot(traj, 'ro-', label="Integrated Intensity")
                self.ax.set_title(f"T1 Recovery: {folder_path.split('/')[-1]}")
                self.ax.legend()
                self.canvas.draw()
                
                self.label.setText(f"Loaded: {folder_path}")
        except Exception as e:
            self.label.setText(f"Error: {str(e)}")