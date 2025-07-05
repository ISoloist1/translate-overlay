import sys

import keyboard
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QHBoxLayout, QVBoxLayout, QFileDialog, QSpinBox, QComboBox, QMessageBox
)

from translate_overlay.translate import TRANSLATERS
from translate_overlay.ocr import TEXT_RECOGNIZERS
from translate_overlay.interface.overlay import FullscreenBlackOverlay
from translate_overlay.interface.controller import Controller
from translate_overlay.interface.const import SHORTCUT_KEYS, ACTIONS


class MainWindow(QMainWindow):
    def __init__(self, trd_path="", ocr_path="", translate_path=""):
        super().__init__()
        self.setWindowTitle("Translate Overlay")
        self.setGeometry(100, 100, 600, 200)
        self.setMinimumSize(500, 500)
        self.screens = QApplication.screens()

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)


        # Text Region Detect model path
        trd_path_layout = QHBoxLayout()
        self.trd_path_edit = QLineEdit()
        self.trd_path_edit.setText(trd_path)
        trd_path_browse = QPushButton("Browse...")
        trd_path_browse.clicked.connect(self._browse_trd_path)
        trd_path_layout.addWidget(QLabel("Text Region Detect model path:"))
        trd_path_layout.addWidget(self.trd_path_edit)
        trd_path_layout.addWidget(trd_path_browse)
        main_layout.addLayout(trd_path_layout)
        main_layout.addSpacing(20)


        # OCR model list
        ocr_model_layout = QHBoxLayout()
        self.ocr_model_dropdown = QComboBox()
        self.ocr_model_dropdown.addItems(TEXT_RECOGNIZERS.keys())
        ocr_model_layout.addWidget(QLabel("OCR model:"))
        ocr_model_layout.addWidget(self.ocr_model_dropdown)
        main_layout.addLayout(ocr_model_layout)

        # OCR model path
        ocr_path_layout = QHBoxLayout()
        self.ocr_path_edit = QLineEdit()
        self.ocr_path_edit.setText(ocr_path)
        ocr_path_browse = QPushButton("Browse...")
        ocr_path_browse.clicked.connect(self._browse_ocr_path)
        ocr_path_layout.addWidget(QLabel("OCR model path:"))
        ocr_path_layout.addWidget(self.ocr_path_edit)
        ocr_path_layout.addWidget(ocr_path_browse)
        main_layout.addLayout(ocr_path_layout)

        main_layout.addSpacing(20)


        # Translate model list
        translate_model_layout = QHBoxLayout()
        self.translate_model_dropdown = QComboBox()
        self.translate_model_dropdown.addItems(TRANSLATERS.keys())
        translate_model_layout.addWidget(QLabel("Translate model:"))
        translate_model_layout.addWidget(self.translate_model_dropdown)
        main_layout.addLayout(translate_model_layout)

        # Translate model path
        translate_path_layout = QHBoxLayout()
        self.translate_path_edit = QLineEdit()
        self.translate_path_edit.setText(translate_path)
        translate_path_browse = QPushButton("Browse...")
        translate_path_browse.clicked.connect(self._browse_translate_path)
        translate_path_layout.addWidget(QLabel("Translate model path:"))
        translate_path_layout.addWidget(self.translate_path_edit)
        translate_path_layout.addWidget(translate_path_browse)
        main_layout.addLayout(translate_path_layout)

        main_layout.addSpacing(20)


        # Source language dropdown
        source_lang_layout = QHBoxLayout()
        self.source_lang_dropdown = QComboBox()
        self.source_langs = TRANSLATERS[self.translate_model_dropdown.currentText()].source_lang_list()
        self.source_lang_dropdown.addItems(self.source_langs)
        source_lang_layout.addWidget(QLabel("Source Language:"))
        source_lang_layout.addWidget(self.source_lang_dropdown)
        main_layout.addLayout(source_lang_layout)

        # Target language dropdown
        target_lang_layout = QHBoxLayout()
        self.target_lang_dropdown = QComboBox()
        self.target_langs = TRANSLATERS[self.translate_model_dropdown.currentText()].target_lang_list()
        self.target_lang_dropdown.addItems(self.target_langs)
        target_lang_layout.addWidget(QLabel("Target Language:"))
        target_lang_layout.addWidget(self.target_lang_dropdown)
        main_layout.addLayout(target_lang_layout)
        

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

        main_layout.addSpacing(20)


        # Screen list
        screen_list_layout = QHBoxLayout()
        self.screen_list_dropdown = QComboBox()
        self.screen_list_dropdown.addItems([screen.name() for screen in self.screens])
        screen_list_layout.addWidget(QLabel("Screen:"))
        screen_list_layout.addWidget(self.screen_list_dropdown)
        main_layout.addLayout(screen_list_layout)


        self.translate_model_dropdown.currentIndexChanged.connect(self._update_lang_list)
        self.source_lang_dropdown.currentIndexChanged.connect(self._update_source_lang)
        self.target_lang_dropdown.currentIndexChanged.connect(self._update_target_lang)
        self.screen_list_dropdown.currentIndexChanged.connect(self._update_screen)


        # Instruction label
        instruction_label = QLabel("\n".join([
            f"Shortcut key:", 
            f"\tCapture screen: {SHORTCUT_KEYS['Overlay']}",
            f"\t(In overlay) Start OCR: {SHORTCUT_KEYS['OCR']}",
            f"\tShow previous OCR result: {SHORTCUT_KEYS['PrevResult']}",
        ]))
        main_layout.addWidget(instruction_label, alignment=Qt.AlignBottom)

        # Start button
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._start_overlay)
        main_layout.addWidget(self.start_button, alignment=Qt.AlignRight)
        
        self.overlay_window = None
        self.controller = None


    def closeEvent(self, event):
        if self.overlay_window is not None:
            self.overlay_window.close()
        event.accept()


    def _start_overlay(self):
        if self.overlay_window is None or not self.overlay_window.isVisible():
            self.start_button.setEnabled(False)
            self.start_button.setText("Starting...")

            self.controller = Controller()
            self.controller.init_finished.connect(self._on_init_finished)
            self.controller.error_signal.connect(self._handle_error_message)
            self.controller.init_worker(
                self.trd_path_edit.text(),
                self.ocr_model_dropdown.currentText(),
                self.ocr_path_edit.text(), 
                self.translate_model_dropdown.currentText(),
                self.translate_path_edit.text(), 
                self.source_lang_dropdown.currentText(),
                self.target_lang_dropdown.currentText(),
                self.beam_size_spin.value(),
            )


    def _on_init_finished(self):
        self.overlay_window = FullscreenBlackOverlay(self.controller, self.screens[self.screen_list_dropdown.currentIndex()])
        keyboard.add_hotkey(SHORTCUT_KEYS["Overlay"].lower(), lambda: self.overlay_window.trigger_fade.emit(ACTIONS["Capture_Screen"]))
        keyboard.add_hotkey(SHORTCUT_KEYS["PrevResult"].lower(), lambda: self.overlay_window.trigger_fade.emit(ACTIONS["Previous_Result"]))
        
        self.overlay_window.window_closed.connect(self._overlay_closed)
        self.overlay_window.show()
        
        self.start_button.setText("Stop")
        self.start_button.clicked.disconnect()
        self.start_button.clicked.connect(self._stop_overlay)
        self.start_button.setEnabled(True)
    

    def _stop_overlay(self):
        self.start_button.setEnabled(False)
        
        if self.controller is not None:
            self.controller.clean_up()
            self.controller.deleteLater()
            self.controller = None

        if self.overlay_window is not None:
            self.overlay_window.close()
            # _overlay_closed will be called by destroyed signal

        self.start_button.setText("Start")
        self.start_button.clicked.disconnect()
        self.start_button.clicked.connect(self._start_overlay)

        self.start_button.setEnabled(True)


    def _overlay_closed(self):
        self.overlay_window = None
        keyboard.remove_hotkey(SHORTCUT_KEYS["Overlay"].lower())


    def _handle_error_message(self, error_message):
        self._stop_overlay()

        msg_box = QMessageBox(QMessageBox.Critical, error_message.source, error_message.traceback, parent=self)
        msg_box.setWindowFlag(Qt.WindowStaysOnTopHint)
        msg_box.exec()
        

    def _update_lang_list(self):
        self.source_lang_dropdown.clear()
        self.target_lang_dropdown.clear()
        self.source_langs = TRANSLATERS[self.translate_model_dropdown.currentText()].source_lang_list()
        self.source_lang_dropdown.addItems(self.source_langs)
        self.target_langs = TRANSLATERS[self.translate_model_dropdown.currentText()].target_lang_list()
        self.target_lang_dropdown.addItems(self.target_langs)


    def _update_source_lang(self):
        if self.controller:
            self.controller.update_source_lang(self.source_lang_dropdown.currentText())


    def _update_target_lang(self):
        if self.controller:
            self.controller.update_target_lang(self.target_lang_dropdown.currentText())


    def _update_screen(self):
        if self.overlay_window:
            self.overlay_window.update_screen(self.screens[self.screen_list_dropdown.currentIndex()])


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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(
        trd_path="E:\\work\\1-personal\\CRAFT-onnx\\models",
        ocr_path="E:\\work\\1-personal\\Florence-2-base\\onnx\\onnx",
        # translate_path="E:\\work\\1-personal\\madlad400-3b-mt\\onnx\\quantization\\optimum\\with_accelerate_weight_dedup",
        # translate_path="E:\\work\\1-personal\\gemma-3-1b-it-ONNX\\onnx",
        translate_path="E:\\work\\1-personal\\gemma-3n-E2B-it-ONNX\\onnx",
    )
    window.show()
    sys.exit(app.exec())
