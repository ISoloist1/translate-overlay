import os
import sys

import keyboard
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QHBoxLayout, QVBoxLayout, QFileDialog, QSpinBox, QComboBox
)

from overlay import FullscreenBlackOverlay
from controller import Controller
from const import SHORTCUT_KEYS

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)
from translate.madlad import MadladTranslator


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Translate Overlay")
        self.setGeometry(100, 100, 600, 200)
        self.setMinimumSize(500, 400)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Text Region Detect model path
        trd_path_layout = QHBoxLayout()
        self.trd_path_edit = QLineEdit()
        self.trd_path_edit.setText("E:\\work\\1-personal\\CRAFT-onnx\\models")
        trd_path_browse = QPushButton("Browse...")
        trd_path_browse.clicked.connect(self._browse_trd_path)
        trd_path_layout.addWidget(QLabel("Text Region Detect model path:"))
        trd_path_layout.addWidget(self.trd_path_edit)
        trd_path_layout.addWidget(trd_path_browse)
        main_layout.addLayout(trd_path_layout)
        main_layout.addSpacing(20)

        # OCR model path
        ocr_path_layout = QHBoxLayout()
        self.ocr_path_edit = QLineEdit()
        self.ocr_path_edit.setText("E:\\work\\1-personal\\Florence-2-base\\onnx\\onnx")
        ocr_path_browse = QPushButton("Browse...")
        ocr_path_browse.clicked.connect(self._browse_ocr_path)
        ocr_path_layout.addWidget(QLabel("OCR model path:"))
        ocr_path_layout.addWidget(self.ocr_path_edit)
        ocr_path_layout.addWidget(ocr_path_browse)
        main_layout.addLayout(ocr_path_layout)

        # Translate model path
        translate_path_layout = QHBoxLayout()
        self.translate_path_edit = QLineEdit()
        self.translate_path_edit.setText("E:\\work\\1-personal\\madlad400-3b-mt\\onnx\\quantization\\optimum\\with_accelerate_weight_dedup")
        translate_path_browse = QPushButton("Browse...")
        translate_path_browse.clicked.connect(self._browse_translate_path)
        translate_path_layout.addWidget(QLabel("Translate model path:"))
        translate_path_layout.addWidget(self.translate_path_edit)
        translate_path_layout.addWidget(translate_path_browse)
        main_layout.addLayout(translate_path_layout)

        # Source language dropdown
        source_lang_layout = QHBoxLayout()
        self.source_lang_dropdown = QComboBox()
        source_lang_layout.addWidget(QLabel("Source Language:"))
        source_lang_layout.addWidget(self.source_lang_dropdown)
        main_layout.addLayout(source_lang_layout)

        # Target language dropdown
        target_lang_layout = QHBoxLayout()
        self.target_lang_dropdown = QComboBox()
        target_lang_layout.addWidget(QLabel("Target Language:"))
        target_lang_layout.addWidget(self.target_lang_dropdown)
        main_layout.addLayout(target_lang_layout)

        # Populate dropdowns with language lists
        self.source_langs = MadladTranslator.source_lang_list()
        self.target_langs = MadladTranslator.target_lang_list()
        self.source_lang_dropdown.addItems(self.source_langs)
        self.target_lang_dropdown.addItems(self.target_langs)
        # print(f"Selected: {target_lang_dropdown.itemText(index)}")
        # self.target_lang_dropdown.currentText()

        # Beam size
        beam_layout = QHBoxLayout()
        self.beam_size_spin = QSpinBox()
        self.beam_size_spin.setMinimum(1)
        self.beam_size_spin.setMaximum(10)
        self.beam_size_spin.setValue(3)
        self.beam_size_spin.setFixedWidth(100)
        beam_layout.addWidget(QLabel("Beam Size:"))
        beam_layout.addWidget(self.beam_size_spin)
        main_layout.addLayout(beam_layout)
        
        # Instruction label
        instruction_label = QLabel(f"Shortcut key: {SHORTCUT_KEYS["Overlay"]}")
        main_layout.addWidget(instruction_label, alignment=Qt.AlignBottom)

        # Start button
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._start_overlay)
        main_layout.addWidget(self.start_button, alignment=Qt.AlignRight)
        
        self.overlay_window = None
        self.controller = Controller()


    def closeEvent(self, event):
        if self.overlay_window is not None:
            self.overlay_window.close()
        event.accept()


    def _start_overlay(self):
        if self.overlay_window is None or not self.overlay_window.isVisible():
            self.start_button.setEnabled(False)
            self.start_button.setText("Starting...")

            self.controller.init_finished.connect(self._on_init_finished)
            self.controller.init_worker(
                self.trd_path_edit.text(),
                self.ocr_path_edit.text(), 
                self.translate_path_edit.text(), 
                self.beam_size_spin.value(),
                self.source_lang_dropdown.currentText(),
                self.target_lang_dropdown.currentText()
            )


    def _on_init_finished(self):
        self.overlay_window = FullscreenBlackOverlay(self.controller)
        keyboard.add_hotkey(SHORTCUT_KEYS["Overlay"].lower(), lambda: self.overlay_window.trigger_fade.emit())
        
        self.overlay_window.window_closed.connect(self._overlay_closed)
        self.overlay_window.show()
        
        self.start_button.setText("Stop")
        self.start_button.clicked.disconnect()
        self.start_button.clicked.connect(self._stop_overlay)
        self.start_button.setEnabled(True)
    

    def _stop_overlay(self):
        self.controller.clean_up()

        if self.overlay_window is not None:
            self.overlay_window.close()
            # _overlay_closed will be called by destroyed signal


    def _overlay_closed(self):
        self.overlay_window = None
        keyboard.remove_hotkey(SHORTCUT_KEYS["Overlay"].lower())

        self.start_button.setText("Start")
        self.start_button.clicked.disconnect()
        self.start_button.clicked.connect(self._start_overlay)


    def _browse_trd_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.trd_path_edit.setText(directory)


    def _browse_ocr_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.ocr_path_edit.setText(directory)


    def _browse_translate_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.translate_path_edit.setText(directory)


# For testing the window independently
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
