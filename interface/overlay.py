from PIL import ImageQt
from PySide6.QtWidgets import QApplication, QWidget, QLabel
from PySide6.QtGui import QPalette, QColor, QPainter, QFontMetrics, QFont
from PySide6.QtCore import Qt, QTimer, Signal, Slot
import keyboard

from const import SHORTCUT_KEYS


class TranslateLabel(QLabel):
    translate_done_signal = Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_text = self.text()
        self.translated_text = None
        self.in_translating = False
        self.translate_done_signal.connect(self.set_translate_text)


    def mousePressEvent(self, event):
        if self.in_translating:
            return 
        
        if event.button() == Qt.LeftButton:
            self.in_translating = True
            if self.translated_text:
                self.set_translate_text(self.translated_text)
            else:
                self.parent().controller.translate_process(self.text(), self.translate_done_signal)
                self.setText("Translating...")

        elif event.button() == Qt.RightButton:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.text())

        super().mousePressEvent(event)


    @Slot(object)
    def set_translate_text(self, translated_text):
        if self.translated_text is None:
            self.translated_text = self.original_text
        else:
            self.translated_text = self.text()

        self.setText(translated_text)
        # self.adjustSize()
        self.stretch_label()
        self.in_translating = False
    

    def stretch_label(self):
        font = self.font()
        metrics = QFontMetrics(font)
        font_width = metrics.horizontalAdvance(self.text())
        label_width = self.width()

        current_font = self.font()
        current_font.setStretch(QFont.SemiCondensed)
        self.setFont(current_font)


class FullscreenBlackOverlay(QWidget):
    trigger_fade = Signal()
    window_closed = Signal()
    ocr_done_signal = Signal(object)
    ocr_start_signal = Signal()

    def __init__(self, controller):
        super().__init__()
        self.screenshot = None
        self._init_ui()
        self.setWindowOpacity(0.0)
        self.fading_in = False
        self.fading_out = False
        self.fade_timer = None
        self.fade_duration = 20
        self.text_label_list = []
        self.toggle_count = 0
        self.controller = controller
        self.ocr_start_signal.connect(self.start_ocr)
        self.ocr_done_signal.connect(self.handle_ocr_result)
        self.in_ocr = False

        self.message_label = None

        # Connect the signal to the slot
        self.trigger_fade.connect(self._toggle_fade)


    def show_text_box(
        self, 
        text: str, 
        text_region: tuple,
        font_size: int=16
    ):
        x_min, y_min, x_max, y_max = text_region
        text_label = TranslateLabel(text, self)
        text_label.setFixedHeight(y_max-y_min)
        text_label.setFixedWidth(x_max-x_min)
        text_label.setStyleSheet(
            f"background-color: rgba(0, 0, 0, 180); color: white; border-radius: 3px; padding: 1px; padding-top: 1px; padding-left: 1px; font-size: {font_size}px;"
        )
        text_label.adjustSize()
        text_label.move(x_min, y_min)
        text_label.stretch_label()
        text_label.show()

        return text_label

    
    @Slot()
    def start_ocr(self):
        self.clean_old_labels()
        self.remove_ocr_hotkey()
        self.in_ocr = True
        self.message_label = self.show_text_box(f"OCRing...", (900, 200, 1100, 300), 30)

        image = ImageQt.fromqpixmap(self.screenshot)
        self.controller.start_ocr(image, self.ocr_done_signal)


    @Slot(object)
    def handle_ocr_result(self, ocr_results):
        # print(ocr_results)
        for ocr_text, region_box in ocr_results:
            # self.set_ocr_text_signal.emit(ocr_text, region_box, int(region_box[3] - region_box[1]))
            self.text_label_list.append(self.show_text_box(ocr_text, region_box, int(region_box[3] - region_box[1]) - 3))

        self.in_ocr = False
        keyboard.add_hotkey(SHORTCUT_KEYS["OCR"].lower(), lambda: self.ocr_start_signal.emit())
        if self.message_label:
            self.message_label.deleteLater()
            self.message_label = None


    def closeEvent(self, event):
        self.window_closed.emit()  # Emit the signal
        self.remove_ocr_hotkey()
        self.clean_old_labels()
        super().closeEvent(event)
        

    def remove_ocr_hotkey(self):
        try:
            keyboard.remove_hotkey(SHORTCUT_KEYS["OCR"].lower())
        except Exception as e:
            print(f"Error removing hotkey: {e}")


    def clean_old_labels(self):
        # Clean up text labels
        for label in self.text_label_list:
            label.deleteLater()
        self.text_label_list.clear()


    @Slot()
    def _toggle_fade(self):
        if self.windowOpacity() < 1.0 and not self.fading_in:
            if not self.screenshot:
                self._capture_screen()

            # self.update()
            self._start_fade_in()
            if not self.in_ocr:
                keyboard.add_hotkey(SHORTCUT_KEYS["OCR"].lower(), lambda: self.ocr_start_signal.emit())

            # Test text box
            # self.toggle_count += 1
            # self.message_label = self.show_text_box(
            #     f"The count is now {self.toggle_count}", 
            #     (
            #         100 * self.toggle_count, 
            #         100 * self.toggle_count,
            #         100 * self.toggle_count + 100, 
            #         100 * self.toggle_count + 50,
            #     )
            # )
        
        elif self.windowOpacity() > 0.0 and not self.fading_out:
            self._start_fade_out()
            self.remove_ocr_hotkey()


    def _capture_screen(self):
        screen = QApplication.primaryScreen()
        self.screenshot = screen.grabWindow(0)
        self.update()


    def _start_fade_in(self):
        self.fading_in = True
        self.fading_out = False
        if self.fade_timer:
            self.fade_timer.stop()
        self.fade_timer = QTimer(self)
        self.fade_timer.timeout.connect(self._fade_in_step)
        self.fade_timer.start(self.fade_duration)


    def _fade_in_step(self):
        opacity = self.windowOpacity()
        if opacity < 1.0:
            self.setWindowOpacity(min(opacity + 0.05, 1.0))
        else:
            self.fade_timer.stop()
            self.fading_in = False


    def _start_fade_out(self):
        self.fading_out = True
        self.fading_in = False
        if self.fade_timer:
            self.fade_timer.stop()
        self.fade_timer = QTimer(self)
        self.fade_timer.timeout.connect(self._fade_out_step)
        self.fade_timer.start(self.fade_duration)


    def _fade_out_step(self):
        opacity = self.windowOpacity()
        if opacity > 0.0:
            self.setWindowOpacity(max(opacity - 0.05, 0.0))
        else:
            self.fade_timer.stop()
            self.fading_out = False


    def paintEvent(self, event):
        painter = QPainter(self)
        if self.screenshot:
            painter.drawPixmap(0, 0, self.screenshot.scaled(self.size()))
            painter.fillRect(self.rect(), QColor(0, 0, 0, 128))


    def _init_ui(self):
        # Set black background
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        # Fullscreen on main monitor
        self.showFullScreen()


if __name__ == "__main__":
    import sys
    import keyboard

    app = QApplication(sys.argv)
    window = FullscreenBlackOverlay()
    window.show()

    # Use a lambda to emit the signal from the hotkey thread
    keyboard.add_hotkey('f9', lambda: window.trigger_fade.emit())

    sys.exit(app.exec())
