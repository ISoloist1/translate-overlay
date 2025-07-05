import sys
import traceback as tb

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from translate_overlay.interface.main_window import MainWindow


def global_exception_hook(exctype, value, traceback):
    msg = "".join(tb.format_exception(exctype, value, traceback))
    msg_box = QMessageBox.critical(None, "Unhandled Exception", msg)
    msg_box.setWindowFlag(Qt.WindowStaysOnTopHint)
    sys.exit(1)

sys.excepthook = global_exception_hook


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
