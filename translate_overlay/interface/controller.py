import os
import sys

from PySide6.QtCore import QThread, Signal, QObject

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)
from ocr import TEXT_RECOGNIZERS
from translate import TRANSLATERS
from interface.worker import OCRWorker, TranslateWorker, TextRegionDetectWorker
from utils.misc import pad_image_to_square, merge_text_images, map_florence2_to_trd_result
from utils.logger import setup_logger


logger = setup_logger()


class Controller(QObject):
    init_finished = Signal()
    trd_data_signal = Signal(object, object)
    ocr_data_signal = Signal(object, object)
    translate_data_signal = Signal(object, object)

    def __init__(self):
        super().__init__()

        self.thread = QThread()
        
        self.processing_image = None
        self.image_x_offset = None
        self.image_y_offset = None

        self.trd_worker = None
        self.trd_id = 0

        self.ocr_model_name = None
        self.ocr_worker = None
        self.ocr_task_dict = dict()
        self.ocr_id = 0
        self.ocr_results = list()

        self.translate_model_name = None
        self.translate_worker = None
        self.translate_task_dict = dict()
        self.translate_id = 0


    def init_worker(
            self, 
            trd_path, 
            ocr_model_name, 
            ocr_path, 
            translate_model_name, 
            translate_path, 
            source_lang, 
            target_lang,
            beam_size, 
    ):
        self.trd_worker = TextRegionDetectWorker(trd_path, beam_size)
        self.trd_worker.moveToThread(self.thread)
        self.trd_worker.initialized.connect(self._on_trd_initialized)
        self.trd_worker.result.connect(self.handle_trd_result)
        self.trd_data_signal.connect(self.trd_worker.run)
        self.thread.started.connect(self.trd_worker.init_worker)

        self.ocr_model_name = ocr_model_name
        self.ocr_worker = OCRWorker(TEXT_RECOGNIZERS[ocr_model_name], ocr_path, beam_size)
        self.ocr_worker.moveToThread(self.thread)
        self.ocr_worker.initialized.connect(self._on_ocr_initialized)
        self.ocr_worker.result.connect(self.handle_text_reco_result)
        self.ocr_data_signal.connect(self.ocr_worker.run)
        self.thread.started.connect(self.ocr_worker.init_worker)

        self.translate_model_name = translate_model_name
        self.translate_worker = TranslateWorker(TRANSLATERS[translate_model_name], translate_path, beam_size, source_lang, target_lang)
        self.translate_worker.moveToThread(self.thread)
        self.translate_worker.initialized.connect(self._on_translate_initialized)
        self.translate_worker.result.connect(self.handle_translate_result)
        self.translate_data_signal.connect(self.translate_worker.run)
        self.thread.started.connect(self.translate_worker.init_worker)

        self._trd_ready = False
        self._ocr_ready = False
        self._translate_ready = False

        self.thread.start()

        
    def _on_trd_initialized(self):
        self._trd_ready = True
        self._check_init_finished()


    def _on_ocr_initialized(self):
        self._ocr_ready = True
        self._check_init_finished()


    def _on_translate_initialized(self):
        self._translate_ready = True
        self._check_init_finished()


    def _check_init_finished(self):
        logger.info(f"TRD ready: {self._trd_ready}, OCR ready: {self._ocr_ready}, Translate ready: {self._translate_ready}")
        if all([self._trd_ready, self._ocr_ready, self._translate_ready]):
            self.init_finished.emit()


    def clean_up(self):
        if self.trd_worker is not None:
            self.trd_worker.deleteLater()
            self.trd_worker = None

        if self.ocr_worker is not None:
            self.ocr_worker.deleteLater()
            self.ocr_worker = None

        if self.translate_worker is not None:
            self.translate_worker.deleteLater()
            self.translate_worker = None

        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()


    def update_source_lang(self, source_lang):
        if self.translate_worker:
            self.translate_worker.source_lang = source_lang


    def update_target_lang(self, target_lang):
        if self.translate_worker:
            self.translate_worker.target_lang = target_lang

        
    def start_ocr(self, data, done_signal, x_offset, y_offset):
        self.processing_image = data
        self.image_x_offset = x_offset
        self.image_y_offset = y_offset
        self.ocr_done_signal = done_signal
        self.trd_process(data)


    def trd_process(self, data):
        if self.trd_worker is not None:
            logger.info("TRD process")
            self.trd_id += 1
            self.trd_data_signal.emit(data, self.trd_id)
        
    
    def handle_trd_result(self, result, data_id):
        logger.info("Handle TRD result")
        cropped_image_box_list = list()
        for box_xyxy in result:
            cropped_image = self.processing_image.crop(box_xyxy)
            cropped_image_box_list.append((cropped_image, box_xyxy))

        if self.ocr_model_name == "Florence-2-base":
            merged_image_box_list = merge_text_images(cropped_image_box_list)
            for idx, item in merged_image_box_list.items():
                self.text_reco_process(item["merged_image"], item["trd_box_list"], item["merged_box_list"])
        else:
            for cropped_image, box_xyxy in cropped_image_box_list:
                self.text_reco_process(cropped_image, box_xyxy)


    def text_reco_process(self, data, trd_box_xyxy, merged_box_list=None):
        if self.ocr_worker is not None:
            logger.info("OCR process")
            self.ocr_id += 1
            self.ocr_task_dict[self.ocr_id] = {
                "trd_box_xyxy": trd_box_xyxy,
            }
            
            if self.ocr_model_name == "Florence-2-base":
                self.ocr_data_signal.emit(data, self.ocr_id)
                self.ocr_task_dict[self.ocr_id].update({
                    "merged_box_list": merged_box_list,
                })
            else:
                padded_image, x_offset, y_offset = pad_image_to_square(data)
                self.ocr_task_dict[self.ocr_id].update({
                    "pad_x_offset": x_offset,
                    "pad_y_offset": y_offset,
                })
                self.ocr_data_signal.emit(padded_image, self.ocr_id)
        
    
    def handle_text_reco_result(self, result, data_id):
        logger.info("Handle OCR result")
        if data_id in self.ocr_task_dict:
            if self.ocr_model_name == "Florence-2-base": 
                result = map_florence2_to_trd_result(
                    result, 
                    self.ocr_task_dict[data_id]["trd_box_xyxy"],
                    self.ocr_task_dict[data_id]["merged_box_list"],
                )
                for text, box_xyxy in result:
                    self.ocr_results.append({
                        "ocr_result": text,
                        "trd_box_xyxy": box_xyxy,
                    })
            else:
                self.ocr_task_dict[data_id]["ocr_result"] = " ".join(result)
                self.ocr_results.append(self.ocr_task_dict[data_id])

            del self.ocr_task_dict[data_id]
            if len(self.ocr_task_dict) == 0:
                self.ocr_finish()

        
    def ocr_finish(self):
        results = []
        for item in self.ocr_results:
            trd_box_xyxy = [
                item["trd_box_xyxy"][0] + self.image_x_offset,
                item["trd_box_xyxy"][1] + self.image_y_offset,
                item["trd_box_xyxy"][2] + self.image_x_offset,
                item["trd_box_xyxy"][3] + self.image_y_offset,
            ]
            results.append((item["ocr_result"], trd_box_xyxy))

        self.ocr_done_signal.emit(results)
        self.processing_image = None
        self.image_x_offset = None
        self.image_y_offset = None
        self.ocr_results = list()


    def translate_process(self, data, done_signal):
        if self.translate_worker is not None:
            self.translate_id += 1
            self.translate_task_dict[self.translate_id] = done_signal
            self.translate_data_signal.emit(data, self.translate_id)
        

    def handle_translate_result(self, result, data_id):
        if data_id in self.translate_task_dict:
            done_signal = self.translate_task_dict[data_id]
            done_signal.emit(result)
            del self.translate_task_dict[data_id]
