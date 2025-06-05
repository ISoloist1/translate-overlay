import os
import sys
from PySide6.QtCore import QObject, Signal, Slot


parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)
from text_region_detect.craft import CRAFT
from ocr.florence2 import Florence2OCR


class Worker(QObject):
    initialized = Signal()
    finished = Signal()
    result = Signal(object, object)

    def __init__(self):
        super().__init__()


class TextRegionDetectWorker(Worker):
    def __init__(self, trd_model_path, beam_size):
        super().__init__()
        self.trd_model_path = trd_model_path
        self.beam_size = beam_size


    @Slot()
    def init_worker(self):
        self.trd = CRAFT(
            model_path=self.trd_model_path, 
        )
        self.initialized.emit()


    @Slot(object, object)
    def run(self, data, data_id):
        # Perform trd processing
        result = self.trd.recognize(data)
        self.result.emit(result, data_id)
        self.finished.emit()


class OCRWorker(Worker):
    def __init__(self, ocr_model_path, beam_size):
        super().__init__()
        self.ocr_model_path = ocr_model_path
        self.beam_size = beam_size


    @Slot()
    def init_worker(self):
        self.ocr = Florence2OCR(
            model_path=self.ocr_model_path, 
            beam_size=self.beam_size,
            output_region=True
        )
        self.initialized.emit()


    @Slot(object, object)
    def run(self, data, data_id):
        # Perform OCR processing
        result = self.ocr.recognize(data)
        self.result.emit(result, data_id)
        self.finished.emit()


class TranslateWorker(Worker):
    def __init__(self, translate_model, translate_model_path, beam_size, source_lang, target_lang):
        super().__init__()
        self.translate_model = translate_model
        self.translate_model_path = translate_model_path
        self.beam_size = beam_size
        self.source_lang = source_lang
        self.target_lang = target_lang


    @Slot()
    def init_worker(self):
        self.translator = self.translate_model(
            model_path=self.translate_model_path, 
            beam_size=self.beam_size,
        )
        self.initialized.emit()


    @Slot(object, object)
    def run(self, data, data_id):
        # Perform translation processing
        result = self.translator.translate(
            data, 
            self.target_lang, 
            self.source_lang
        )
        self.result.emit(result, data_id)
        self.finished.emit()

