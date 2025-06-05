import os
import sys
from PySide6.QtCore import QThread, Signal, QObject


parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)
from interface.worker import OCRWorker, TranslateWorker, TextRegionDetectWorker
from ocr.florence2 import Florence2OCR
from utils.misc import pad_image_to_square, merge_text_images, map_florence2_to_trd_result

class Controller(QObject):
    init_finished = Signal()
    trd_data_signal = Signal(object, object)
    ocr_data_signal = Signal(object, object)
    translate_data_signal = Signal(object, object)

    def __init__(self):
        super().__init__()

        self.thread = QThread()
        
        self.processing_image = None

        self.trd_worker = None
        self.trd_id = 0

        self.ocr_worker = None
        self.ocr_task_dict = dict()
        self.ocr_id = 0
        self.ocr_results = list()

        self.translate_worker = None
        self.translate_task_dict = dict()
        self.translate_id = 0


    def init_worker(self, trd_path, ocr_path, translate_model, translate_path, beam_size, source_lang, target_lang):
        self.trd_worker = TextRegionDetectWorker(trd_path, beam_size)
        self.trd_worker.moveToThread(self.thread)
        self.trd_worker.initialized.connect(self._on_trd_initialized)
        self.trd_worker.result.connect(self.handle_trd_result)
        self.trd_data_signal.connect(self.trd_worker.run)
        self.thread.started.connect(self.trd_worker.init_worker)

        self.ocr_worker = OCRWorker(ocr_path, beam_size)
        self.ocr_worker.moveToThread(self.thread)
        self.ocr_worker.initialized.connect(self._on_ocr_initialized)
        self.ocr_worker.result.connect(self.handle_ocr_result)
        self.ocr_data_signal.connect(self.ocr_worker.run)
        self.thread.started.connect(self.ocr_worker.init_worker)

        self.translate_worker = TranslateWorker(translate_model, translate_path, beam_size, source_lang, target_lang)
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
        print(f"TRD ready: {self._trd_ready}, OCR ready: {self._ocr_ready}, Translate ready: {self._translate_ready}")
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

        
    def start_ocr(self, data, done_signal):
        self.processing_image = data
        self.ocr_done_signal = done_signal
        self.trd_process(data)


    def trd_process(self, data):
        if self.trd_worker is not None:
            print("TRD process")
            self.trd_id += 1
            self.trd_data_signal.emit(data, self.trd_id)
        
    
    def handle_trd_result(self, result, data_id):
        print("Handle TRD result")
        cropped_image_box_list = list()
        for box_xyxy in result:
            cropped_image = self.processing_image.crop(box_xyxy)
            cropped_image_box_list.append((cropped_image, box_xyxy))

        if type(self.ocr_worker.ocr) == Florence2OCR:
            merged_image_box_list = merge_text_images(cropped_image_box_list)
            for idx, item in merged_image_box_list.items():
                self.ocr_process(item["merged_image"], item["trd_box_list"], item["merged_box_list"])
        else:
            for cropped_image, box_xyxy in cropped_image_box_list:
                self.ocr_process(cropped_image, box_xyxy)


    def ocr_process(self, data, trd_box_xyxy, merged_box_list=None):
        if self.ocr_worker is not None:
            print("OCR process")
            self.ocr_id += 1
            self.ocr_task_dict[self.ocr_id] = {
                "trd_box_xyxy": trd_box_xyxy,
            }
            
            if type(self.ocr_worker.ocr) == Florence2OCR:
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
        
    
    def handle_ocr_result(self, result, data_id):
        print("Handle OCR result")
        if data_id in self.ocr_task_dict:
            if result:
                if type(self.ocr_worker.ocr) == Florence2OCR: 
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
            results.append((item["ocr_result"], item["trd_box_xyxy"]))

        self.ocr_done_signal.emit(results)
        self.processing_image = None
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
