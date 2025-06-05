import os
import sys

from PySide6.QtWidgets import QApplication

parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent)
from interface.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
