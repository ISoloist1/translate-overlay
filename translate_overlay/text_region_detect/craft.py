import os
import sys
import math

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

from translate_overlay.text_region_detect.base import BaseTextRegionDetection
from translate_overlay.utils.logger import setup_logger, log_timing


logger = setup_logger()


class CRAFT(BaseTextRegionDetection):
    def __init__(self, model_path: str):
        self.model_path = model_path

        self._load_model()


    @log_timing(logger, __name__, "Load model")
    def _load_model(self):
        reco_model_path = os.path.join(self.model_path, "craftmlt25k.onnx")
        refine_model_path = os.path.join(self.model_path, "refine.onnx")

        for model_path in [
            reco_model_path,
            refine_model_path
        ]:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False

        self.reco_session = ort.InferenceSession(reco_model_path, sess_options=sess_options)
        self.refine_session = ort.InferenceSession(refine_model_path, sess_options=sess_options)
        

    @log_timing(logger, __name__, "Preprocess")
    def _preprocess(self, input_image):
        def resize_aspect_retio(input_image, square_size, mag_ratio):
            width, height = input_image.size
            target_size = mag_ratio * max(height, width)

            if target_size > square_size:
                target_size = square_size

            ratio = target_size / max(height, width)    
            target_h, target_w = int(height * ratio), int(width * ratio)
            target_h32, target_w32 = target_h, target_w
            if target_h % 32 != 0:
                target_h32 = target_h + (32 - target_h % 32)
            if target_w % 32 != 0:
                target_w32 = target_w + (32 - target_w % 32)

            resized_image = input_image.resize((target_w, target_h))
            pad_resized_image = Image.new(
                resized_image.mode, 
                (target_w32, target_h32), 
                (0, 0, 0)
            )
            pad_resized_image.paste(resized_image)

            return pad_resized_image, ratio
        

        def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
            # should be RGB order
            img = in_img.copy().astype(np.float32)
            img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
            img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
            return img


        pad_resized_image, ratio = resize_aspect_retio(input_image, 1920, 1.5)
        image_array = np.array(pad_resized_image)

        if image_array.shape[2] == 4: 
            image_array = image_array[:,:,:3]

        processed_image_array = np.zeros((
            pad_resized_image.height, 
            pad_resized_image.width, 
            image_array.shape[2]
        ), dtype=np.float32)
        processed_image_array[0:pad_resized_image.height, 0:pad_resized_image.width, :] = image_array

        processed_image_array = normalizeMeanVariance(processed_image_array)
        processed_image_array = np.transpose(processed_image_array, (2, 0, 1))
        processed_image_array = np.expand_dims(processed_image_array, axis=0)

        return processed_image_array, ratio


    @log_timing(logger, __name__, "Inference")
    def _inference(self, input_array):
        reco_input_name = self.reco_session.get_inputs()[0].name
        y, feature = self.reco_session.run(None, {reco_input_name: input_array})

        refinput_name_y = self.refine_session.get_inputs()[0].name
        refinput_name_feature = self.refine_session.get_inputs()[1].name
        y_refiner = self.refine_session.run(None, {refinput_name_y: y, refinput_name_feature: feature})[0]

        return y, y_refiner


    @log_timing(logger, __name__, "Postprocess")
    def _postprocess(self, infer_output, refine_output, input_image, ratio):
        def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
            # prepare data
            linkmap = linkmap.copy()
            textmap = textmap.copy()
            img_h, img_w = textmap.shape

            """ labeling method """
            ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
            ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

            text_score_comb = np.clip(text_score + link_score, 0, 1)
            nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

            boxes = []
            for k in range(1,nLabels):
                # size filtering
                size = stats[k, cv2.CC_STAT_AREA]
                if size < 10: continue

                # thresholding
                if np.max(textmap[labels==k]) < text_threshold: continue

                # make segmentation map
                segmap = np.zeros(textmap.shape, dtype=np.uint8)
                segmap[labels==k] = 255
                segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
                x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
                w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
                niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
                sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
                # boundary check
                if sx < 0 : sx = 0
                if sy < 0 : sy = 0
                if ex >= img_w: ex = img_w
                if ey >= img_h: ey = img_h
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
                segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

                # make box
                np_contours = np.roll(np.array(np.where(segmap!=0)), 1, axis=0).transpose().reshape(-1,2)
                rectangle = cv2.minAreaRect(np_contours)
                box = cv2.boxPoints(rectangle)

                # align diamond-shape
                w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
                box_ratio = max(w, h) / (min(w, h) + 1e-5)
                if abs(1 - box_ratio) <= 0.1:
                    l, r = min(np_contours[:,0]), max(np_contours[:,0])
                    t, b = min(np_contours[:,1]), max(np_contours[:,1])
                    box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

                # make clock-wise order
                startidx = box.sum(axis=1).argmin()
                box = np.roll(box, 4-startidx, 0)
                box = np.array(box)

                boxes.append(box)

            return boxes
        

        def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
            if len(polys) > 0:
                polys = np.array(polys)
                for k in range(len(polys)):
                    if polys[k] is not None:
                        polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
            return polys
        
        
        score_text = infer_output[0,:,:,0]
        score_link = refine_output[0,:,:,0]
        ratio = 1 / ratio
        boxes = getDetBoxes_core(score_text, score_link, 0.5, 0.4, 0.4)
        adjusted_boxes = adjustResultCoordinates(boxes, ratio, ratio)

        boxes_xxyy = []
        w, h = input_image.size

        for box in adjusted_boxes:
            x_min = max(math.floor(min(box, key=lambda x: x[0])[0]), 1)
            x_max = min(math.ceil(max(box, key=lambda x: x[0])[0]), w-1)
            y_min = max(math.floor(min(box, key=lambda x: x[1])[1]), 3)
            y_max = min(math.ceil(max(box, key=lambda x: x[1])[1]), h-2)    
            boxes_xxyy.append([x_min-1, y_min-1, x_max+1, y_max+1])

        return boxes_xxyy


    def recognize(self, input_image):
        
        # Preprocess the image
        input_array, ratio = self._preprocess(input_image)
        
        # Perform inference
        infer_output, refine_output = self._inference(input_array)
        
        # Postprocess the outputs to get the final text
        boxes_xyxy = self._postprocess(infer_output, refine_output, input_image, ratio)
        
        return boxes_xyxy




if __name__ == "__main__":
    from utils.misc import draw_boxes, merge_text_images, group_boxes_to_paragraphs
    # Example usage
    model_path = sys.argv[1]
    image = Image.open(sys.argv[2])

    text_detect = CRAFT(model_path)
    boxes_xyxy = text_detect.recognize(image)


    # Test crop
    text_box_list = [("", i) for i in boxes_xyxy]
    merged_boxes_xyxy = []
    merged_original_cluster_groups = []

    grouped_boxes_xyxy, original_cluster_groups = group_boxes_to_paragraphs(text_box_list)
    for i in grouped_boxes_xyxy:
        box_list = [j[1] for j in i]
        box_list_transposed = list(zip(*box_list))

        merged_boxes_xyxy.append((
            min(box_list_transposed[0]), 
            min(box_list_transposed[1]), 
            max(box_list_transposed[2]), 
            max(box_list_transposed[3]), 
        ))

    for i in original_cluster_groups:
        box_list = [j[1] for j in i]
        box_list_transposed = list(zip(*box_list))

        merged_original_cluster_groups.append((
            min(box_list_transposed[0]), 
            min(box_list_transposed[1]), 
            max(box_list_transposed[2]), 
            max(box_list_transposed[3]), 
        ))
    # End test crop


    # Test merge images
    # crop_image_list = list()
    # for box_xyxy in boxes_xyxy:
    #     new_image = image.crop(box_xyxy)
    #     crop_image_list.append((new_image, box_xyxy))

    # merged_image_dict = merge_text_images(crop_image_list)
    # for idx, item in merged_image_dict.items():
    #     item["merged_image"].show()

    # End merge images



    image = draw_boxes(image, boxes_xyxy, "yellow")
    image = draw_boxes(image, merged_original_cluster_groups, "green")
    image = draw_boxes(image, merged_boxes_xyxy, "red")

    logger.info(boxes_xyxy)
    image.show()

