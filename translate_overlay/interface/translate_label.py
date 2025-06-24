import os
import sys

from PySide6.QtWidgets import QApplication, QLabel, QWidget
from PySide6.QtGui import QFontMetrics, QPainter, QColor
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QObject, QRect

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)
from utils.logger import setup_logger


logger = setup_logger()


class TranslateLabelGroupBackground(QWidget):
    def __init__(self, rect, parent=None):
        super().__init__(parent)
        self.setGeometry(rect)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: transparent;")
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        color = QColor(0, 0, 0, 100)  # Semi-transparent black
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())


class TranslateLabelGroup(QObject):
    copy_signal = Signal()
    translate_signal = Signal()
    translate_done_signal = Signal(object)

    def __init__(self, ocr_result_group, parent, tokenizer):
        super().__init__()

        self.parent = parent
        self.tokenizer = tokenizer
        self.in_action = False
        self.translate_done_signal.connect(self.handle_translate_result)

        ocr_result_group.sort(key=lambda item: item[1][1])

        self.text_label_list = list()
        for ocr_text, region_box in ocr_result_group:
            self.text_label_list.append(self.show_text_box(
                ocr_text, 
                region_box, 
                parent
            ))

        xyxy_list = list(zip(*[item[1] for item in ocr_result_group]))
        x_min = min(xyxy_list[0])
        y_min = min(xyxy_list[1])
        x_max = max(xyxy_list[2])
        y_max = max(xyxy_list[3])
        self.box_xyxy = (x_min, y_min, x_max, y_max)
        self.group_bg = TranslateLabelGroupBackground(QRect(x_min, y_min, x_max-x_min, y_max-y_min), parent)
        self.group_bg.lower()  # Ensure background is below labels
        
        self.set_current_text()


    def get_box_xyxy(self):
        return self.box_xyxy
    

    def get_text_box_list(self):
        return [(label.original_text, label.box_xyxy) for label in self.text_label_list]


    def set_current_text(self):
        self.current_text = ' '.join([label.text() for label in self.text_label_list])


    def set_message_text(self, text_list):
        len_diff = len(self.text_label_list) - len(text_list)
        assert len_diff >= 0

        for label, text in zip(self.text_label_list, text_list + [""]*len_diff):
            label.setText(text)


    def set_hidden(self, state):
        for label in self.text_label_list:
            label.setHidden(state)

        self.group_bg.setHidden(state)


    def clean_up(self):
        for label in self.text_label_list:
            label.deleteLater()

        if hasattr(self, "group_bg"):
            self.group_bg.deleteLater()


    def copy_text(self):
        if self.in_action:
            return
        self.in_action = True

        self.set_current_text()
        clipboard = QApplication.clipboard()
        clipboard.setText(self.current_text)
        self.set_message_text(["Copied!"])
        QTimer.singleShot(500, self._restore_text)

        
    def _restore_text(self):
        for label in self.text_label_list:
            label.setText(label.current_text)

        self.in_action = False


    def split_group(self):
        self.parent.split_label_group_signal.emit(self)


    def translate(self):
        if self.in_action:
            return
        self.in_action = True
        
        if self.text_label_list[0].replace_text:
            for label in self.text_label_list:
                label.set_translate_text(label.replace_text)
            self.in_action = False
        else:
            self.set_message_text(["Translating..."])
            logger.info(f"Translate: {self.current_text}")
            self.parent.controller.translate_process(self.current_text, self.translate_done_signal)


    @Slot(object)
    def handle_translate_result(self, translate_result):
        translate_result_list = self.split_text(translate_result)
        
        logger.info(f"Translated text: {translate_result}")
        logger.info(f"Splitted text: {translate_result_list}")
        
        for label, text in zip(self.text_label_list, translate_result_list):
            label.set_translate_text(text)

        self.in_action = False


    def split_text(self, text):
        # Break down long sentence into short segments with similar length
        # Number of segments is the same as self.text_label_list
        # Use sentencepiece tokenizer to tokenize long sentence into pieces
        # Accumulate length of characters from each piece to get segment length
        # If space is avaiable near the segment point, segment at space first, so words in languages like English are complete
        # Only for languages like Chinese, Japanese, Thai where space is not need, cut after segment length is long enough
        num_segments = len(self.text_label_list)
        if num_segments <= 1 or not text:
            return [text] + [""] * (num_segments - 1)

        # Tokenize the text
        pieces = self.tokenizer.EncodeAsPieces(text)
        piece_lens = [len(piece) for piece in pieces]
        total_chars = sum(piece_lens)
        target_len = max(1, total_chars // num_segments + 1)

        segments = []
        current = []    
        current_len = 0

        for i, (piece, piece_len) in enumerate(zip(pieces, piece_lens)):
            if current and current_len >= target_len and len(segments) <= num_segments:
                if not piece.startswith('\u2581'):
                    if any([future_piece.startswith('\u2581') for future_piece in pieces[i+1:i+4]]):
                        pass
                    elif any([past_piece.startswith('\u2581') for past_piece in current[-3:]]):
                        for idx, past_piece in enumerate(current[::-1][:3]):
                            if past_piece.startswith('\u2581'):
                                segments.append(current[:len(current)-idx-1])
                                current = current[len(current)-idx-1:]
                                current_len = len(current)
                    else:
                        segments.append(current)
                        current = []
                        current_len = 0
                else:
                    segments.append(current)
                    current = []
                    current_len = 0

            current.append(piece)
            current_len += piece_len

        if current:
            segments.append(current)

        segment_piece_count = sum([len(segment) for segment in segments])
        assert segment_piece_count == len(pieces)

        segments = [self.tokenizer.DecodePieces(segment) for segment in segments]

        return segments


    def show_text_box(
        self,
        text: str, 
        text_region: tuple,
        parent: object,
    ):
        x_min, y_min, x_max, y_max = text_region
        text_label = TranslateLabel(text, parent)
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
        text_label.set_group(self)
        text_label.set_box_xyxy(text_region)
        text_label.show()

        return text_label


class TranslateLabel(QLabel):
    translate_done_signal = Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_text = self.text()
        self.current_text = self.text()
        self.replace_text = None
        self.group = None
        self.in_action = False
        self.setWordWrap(False)
        self.setContentsMargins(0, 0, 0, 0)
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.translate_done_signal.connect(self.set_translate_text)


    def mousePressEvent(self, event):
        if self.in_action:
            return 
        
        ctrl_held = event.modifiers() & Qt.KeyboardModifier.ControlModifier
        if event.button() == Qt.LeftButton:
            # self.in_action = True
            # if self.replace_text:
            #     self.set_translate_text(self.replace_text)
            # else:
            #     self.parent().controller.translate_process(self.text(), self.translate_done_signal)
            #     self.setText("Translating...")
            if not ctrl_held:
                self.group.translate()
            else: 
                self.group.split_group()

        elif event.button() == Qt.RightButton:
            # self.in_action = True
            # clipboard = QApplication.clipboard()
            # clipboard.setText(self.text())
            # self.setText("Copied!")
            # QTimer.singleShot(500, self._restore_text)
            self.group.copy_text()

        super().mousePressEvent(event)


    def _restore_text(self):
        self.setText(self.current_text)
        self.in_action = False


    def set_box_xyxy(self, box_xyxy):
        self.box_xyxy = box_xyxy


    def get_box_xyxy(self):
        return self.box_xyxy


    def set_group(self, group):
        self.group = group


    @Slot(object)
    def set_translate_text(self, replace_text):
        if self.replace_text is None:
            self.replace_text = self.current_text
        else:
            self.replace_text = self.text()

        self.setText(replace_text)
        self.fit_text_to_label()
        self.in_action = False


    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_text_to_label()


    def fit_text_to_label(self):
        """Dynamically adjust font size to fit text within label's width and height."""
        if not self.text():
            return

        # Start from a large font size and decrease until it fits
        min_font_size = 6
        max_font_size = min(self.height(), self.width())  # Reasonable upper bound
        font = self.font()
        metrics = QFontMetrics(font)
        text = self.text()

        # Binary search for best font size
        best_size = min_font_size
        left, right = min_font_size, max_font_size
        while left <= right:
            mid = (left + right) // 2
            font.setPointSize(mid)
            metrics = QFontMetrics(font)
            # Use boundingRect for more accurate measurement (includes overhangs)
            rect = metrics.boundingRect(self.rect(), Qt.AlignCenter, text)
            if rect.width() <= self.width() and rect.height() <= self.height():
                best_size = mid
                left = mid + 1
            else:
                right = mid - 1

        font.setPointSize(best_size)
        self.setFont(font)
