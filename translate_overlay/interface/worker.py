import traceback

from PySide6.QtCore import QObject, Signal, Slot

from translate_overlay.text_region_detect.craft import CRAFT
from translate_overlay.utils.misc import ErrorMessage, split_text_tokens
from translate_overlay.utils.logger import setup_logger


logger = setup_logger()


class Worker(QObject):
    initialized = Signal()
    finished = Signal()
    result = Signal(object, object)
    error = Signal(object)

    def __init__(self):
        super().__init__()


class TRDWorker(Worker):
    def __init__(self, trd_model_path, beam_size):
        super().__init__()
        self.trd_model_path = trd_model_path
        self.beam_size = beam_size


    @Slot()
    def init_worker(self):
        try:
            self.trd = CRAFT(
                model_path=self.trd_model_path, 
            )
            self.initialized.emit()
        except Exception as e:
            error_message = ErrorMessage(e, traceback.format_exc(), "TRD Worker")
            self.error.emit(error_message)


    @Slot(object, object)
    def run(self, data, data_id):
        # Perform trd processing
        try:
            result = self.trd.recognize(data)
            self.result.emit(result, data_id)
        except Exception as e:
            error_message = ErrorMessage(e, traceback.format_exc(), "TRD Worker")
            self.error.emit(error_message)


class OCRWorker(Worker):
    def __init__(self, ocr_model, ocr_model_path, beam_size):
        super().__init__()
        self.ocr_model = ocr_model
        self.ocr_model_path = ocr_model_path
        self.beam_size = beam_size


    @Slot()
    def init_worker(self):
        try:
            self.ocr = self.ocr_model(
                model_path=self.ocr_model_path, 
                beam_size=self.beam_size,
                output_region=True
            )
            self.initialized.emit()
        except Exception as e:
            error_message = ErrorMessage(e, traceback.format_exc(), "OCR Worker")
            self.error.emit(error_message)


    @Slot(object, object)
    def run(self, data, data_id):
        # Perform OCR processing
        try:
            result = self.ocr.recognize(data)
            self.result.emit(result, data_id)
        except Exception as e:
            error_message = ErrorMessage(e, traceback.format_exc(), "OCR Worker")
            self.error.emit(error_message)


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
        try:
            self.translator = self.translate_model(
                model_path=self.translate_model_path, 
                beam_size=self.beam_size,
            )
            self.initialized.emit()
        except Exception as e:
            error_message = ErrorMessage(e, traceback.format_exc(), "Translate Worker")
            self.error.emit(error_message)


    @Slot(object, object)
    def run(self, data, box_list, data_id):
        # Perform translation processing
        try:
            result = self.translator.translate(
                data, 
                self.target_lang, 
                self.source_lang
            )
            result = self.split_text(result, box_list)
            self.result.emit(result, data_id)
        except Exception as e:
            error_message = ErrorMessage(e, traceback.format_exc(), "Translate Worker")
            self.error.emit(error_message)

            
    def split_text(self, text, box_list):
        num_segments = len(box_list)
        if num_segments <= 1 or not text:
            return [text] + [""] * (num_segments - 1)

        # Tokenize the text
        pieces = self.translator.encode_tokens(text)

        segments = [self.translator.decode_tokens(segment) for segment in split_text_tokens(pieces, box_list)]

        logger.info(f"Translation: {text}")
        logger.info(f"Segments: {segments}")

        return segments

