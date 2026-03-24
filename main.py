import sys
import os
from PyQt6.QtWidgets import QApplication
from app.main_window import MainWindow

def main():
    # 1. Initialize the GUI Application
    app = QApplication(sys.argv)
    app.setApplicationName("NMR T1 Processor")
    
    # 2. Set high-level styling (Optional, but makes it look cleaner)
    app.setStyle("Fusion") 

    # 3. Create and show the Main Window
    window = MainWindow()
    window.show()

    # 4. Start the event loop
    # sys.exit ensures the script closes properly when the window is closed
    sys.exit(app.exec())

if __name__ == "__main__":
    main()