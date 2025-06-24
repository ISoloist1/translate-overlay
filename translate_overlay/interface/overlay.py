import os
import sys

import keyboard
from PIL import ImageQt
import sentencepiece as spm
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QPalette, QColor, QPainter
from PySide6.QtCore import Qt, QTimer, Signal, Slot

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)
from interface.const import SHORTCUT_KEYS, ACTIONS
from utils.logger import setup_logger
from utils.misc import is_mostly_inside
from interface.translate_label import TranslateLabel, TranslateLabelGroup


logger = setup_logger()
TOKENIZER_MODEL = os.path.join(parent, "utils", "spm", "spiece.model")


class FullscreenBlackOverlay(QWidget):
    trigger_fade = Signal(object)
    window_closed = Signal()
    ocr_done_signal = Signal(object)
    ocr_start_signal = Signal(object, object, object, )
    split_label_group_signal = Signal(object)

    def __init__(self, controller=None, screen=None):
        super().__init__()
        self.screenshot = None
        self.previous_screenshot = None
        self.controller = controller
        self.screen = screen

        self._init_ui()
        self.setWindowOpacity(0.0)
        self.fading_in = False
        self.fading_out = False
        self.fade_timer = None
        self.fade_duration = 20
        
        self.snip_mode = False
        self.snip_start = None
        self.snip_end = None
        self.snip_rect = None
        self.snip_result = None

        self.text_label_list = []
        self.text_label_group_list = []
        self.ocr_start_signal.connect(self.start_ocr)
        self.ocr_done_signal.connect(self.handle_ocr_result)
        self.in_ocr = False

        self.message_label = None

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(TOKENIZER_MODEL)

        # Connect the signal to the slot
        self.trigger_fade.connect(self._toggle_fade)
        self.split_label_group_signal.connect(self._handle_label_clicked)


    def show_text_box(
        self,
        text: str, 
        text_region: tuple,
    ):
        x_min, y_min, x_max, y_max = text_region
        text_label = TranslateLabel(text, self)
        text_label.setFixedHeight(y_max-y_min)
        text_label.setFixedWidth(x_max-x_min)
        text_label.setStyleSheet((
            "background-color: rgba(0, 0, 0, 180); "
            "color: white; "
            # "text-align: center; "
            "border-radius: 3px; "
            # "padding: 1px; "
            # f"font-size: {font_size}px; "
        ))
        text_label.move(x_min, y_min)
        text_label.fit_text_to_label()
        text_label.show()

        return text_label

    
    @Slot(object, object, object, )
    def start_ocr(self, image, x_offset, y_offset):
        if self.controller:
            self.remove_ocr_hotkey()
            if self.previous_screenshot != self.screenshot:
                self.clean_old_labels()
                self.previous_screenshot = self.screenshot

            self.in_ocr = True
            self.message_label = self.show_text_box(f"OCRing...", (900, 200, 1100, 250))

            image = ImageQt.fromqpixmap(image)
            self.controller.start_ocr(image, self.ocr_done_signal, x_offset, y_offset)


    @Slot(object)
    def handle_ocr_result(self, ocr_result_groups):
        self.create_text_label_group(ocr_result_groups)

        self.in_ocr = False
        self.set_ocr_hotkey()
        self.enable_snip_mode()
        if self.message_label:
            self.message_label.deleteLater()
            self.message_label = None


    @Slot(object)
    def _handle_label_clicked(self, group):
        text_box_list = group.get_text_box_list()
        self.text_label_group_list.remove(group)
        group.clean_up()
        group.deleteLater()

        self.create_text_label_group([[item] for item in text_box_list])


    def create_text_label_group(self, ocr_result_groups):
        for group in ocr_result_groups:
            self.text_label_group_list.append(
                TranslateLabelGroup(group, self, self.sp_model)
            )


    def closeEvent(self, event):
        self.window_closed.emit()  # Emit the signal
        self.remove_ocr_hotkey()
        self.clean_old_labels()
        super().closeEvent(event)


    def set_ocr_hotkey(self):
        keyboard.add_hotkey(
            SHORTCUT_KEYS["OCR"].lower(), 
            lambda: self.ocr_start_signal.emit(self.screenshot, 0, 0)
        )
        

    def remove_ocr_hotkey(self):
        try:
            keyboard.remove_hotkey(SHORTCUT_KEYS["OCR"].lower())
        except Exception as e:
            logger.warning(f"Error removing hotkey: {e}")


    def clean_old_labels(self):
        # Clean up text labels
        for label in self.text_label_list:
            label.deleteLater()
        self.text_label_list.clear()

        for group in self.text_label_group_list:
            group.clean_up()
        self.text_label_group_list.clear()


    def _capture_screen(self):
        self.screenshot = self.screen.grabWindow(0)


    def update_screen(self, screen):
        self.screen = screen


    def _set_labels_hidden(self, state):
        for label in self.text_label_list:
            label.setHidden(state)

        for group in self.text_label_group_list:
            group.set_hidden(state)


    @Slot(object)
    def _toggle_fade(self, action):
        if self.windowOpacity() < 1.0 and not self.fading_in:
            if action == ACTIONS["Capture_Screen"] and not self.screenshot:
                self._capture_screen()
                self._set_labels_hidden(True)

            elif action == ACTIONS["Previous_Result"] and self.previous_screenshot:
                self.screenshot = self.previous_screenshot
                self._set_labels_hidden(False)

            self.update()
            self._start_fade_in()
            self.enable_snip_mode()
            if not self.in_ocr:
                self.set_ocr_hotkey()
        
        elif self.windowOpacity() > 0.0 and not self.fading_out:
            self.disable_snip_mode()
            self._start_fade_out()
            self.remove_ocr_hotkey()


    ########## Fade animation section
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
            self.screenshot = None


    ########## Snip animation section
    def enable_snip_mode(self):
        self.snip_mode = True
        self.snip_start = None
        self.snip_end = None
        self.snip_rect = None
        self.snip_result = None
        self.update()


    def disable_snip_mode(self):
        self.snip_mode = False
        self.snip_start = None
        self.snip_end = None
        self.snip_rect = None
        self.update()


    def mousePressEvent(self, event):
        if self.snip_mode and event.button() == Qt.LeftButton:
            self.snip_start = event.pos()
            self.snip_end = event.pos()
            self.update()

        super().mousePressEvent(event)


    def mouseMoveEvent(self, event):
        if self.snip_mode and self.snip_start:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.snip_end = event.pos()
            self.update()

        super().mouseMoveEvent(event)


    def mouseReleaseEvent(self, event):
        if self.snip_mode and event.button() == Qt.LeftButton and self.snip_start:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            ctrl_held = event.modifiers() & Qt.KeyboardModifier.ControlModifier
            self.snip_end = event.pos()
            self.snip_rect = self._get_snip_rect()
            self.snip_result = self._get_snip_image()
            left, top, width, height = self.snip_rect
            right = left + width
            bottom = top + height
            self.disable_snip_mode()

            # Do something with self.snip_result (QPixmap)
            if width >= 10 and height >= 10:
                if not ctrl_held:
                    self.ocr_start_signal.emit(self.snip_result, left, top)
                else:
                    selected_text_box = list()
                    for group in self.text_label_group_list[:]:
                        box_xyxy = group.get_box_xyxy()
                        if any([
                            box_xyxy[0] >= left,
                            box_xyxy[1] >= top,
                            box_xyxy[2] <= right,
                            box_xyxy[3] <= bottom,
                        ]) and is_mostly_inside(
                            box_xyxy,
                            (left, top, right, bottom),
                            0.5
                        ):
                            selected_text_box += group.get_text_box_list()
                            self.text_label_group_list.remove(group)
                            group.clean_up()
                            group.deleteLater()

                    if len(selected_text_box) > 0:
                        self.create_text_label_group([selected_text_box])
                    self.enable_snip_mode()
            else:
                self.enable_snip_mode()

        super().mouseReleaseEvent(event)


    def _get_snip_rect(self):
        if not self.snip_start or not self.snip_end:
            return None
        
        x1, y1 = self.snip_start.x(), self.snip_start.y()
        x2, y2 = self.snip_end.x(), self.snip_end.y()
        left, top = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)

        return (left, top, width, height)


    def _get_snip_image(self):
        if self.screenshot and self.snip_rect:
            left, top, width, height = self.snip_rect
            return self.screenshot.copy(left, top, width, height)
        
        return None


    ########## Overlay paint event
    def paintEvent(self, event):
        painter = QPainter(self)
        if self.screenshot:
            # Draw the dimmed screenshot first
            painter.drawPixmap(0, 0, self.screenshot.scaled(self.size()))
            painter.fillRect(self.rect(), QColor(0, 0, 0, 128))

            if self.snip_mode and self.snip_start and self.snip_end:
                rect = self._get_snip_rect()
                if rect:
                    left, top, width, height = rect
                    # Only draw if width and height are both > 0
                    if width > 0 and height > 0:
                        source_pixmap = self.screenshot.scaled(self.size())
                        painter.drawPixmap(left, top, source_pixmap.copy(left, top, width, height))
                        # Draw the white border
                        pen = painter.pen()
                        pen.setColor(QColor("white"))
                        pen.setWidth(2)
                        painter.setPen(pen)
                        painter.drawRect(left, top, width, height)


    def _init_ui(self):
        # Set black background
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        # Fullscreen on main monitor
        self.move(self.screen.geometry().topLeft())
        self.showFullScreen()


if __name__ == "__main__":
    import sys
    import keyboard

    app = QApplication(sys.argv)
    screen = QApplication.primaryScreen()
    window = FullscreenBlackOverlay(screen=screen)
    window.show()

    # Use a lambda to emit the signal from the hotkey thread
    keyboard.add_hotkey(SHORTCUT_KEYS["Overlay"].lower(), lambda: window.trigger_fade.emit(ACTIONS["Capture_Screen"]))

    sys.exit(app.exec())
