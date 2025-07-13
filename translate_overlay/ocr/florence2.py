# Regex pattern, coordinate quantizer, mean/std, etc. from Florence 2 code implementation by Microsoft:
# https://huggingface.co/microsoft/Florence-2-base/blob/main/processing_florence2.py

import os
import re
import sys
from typing import Any, List, Dict, Tuple

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from translate_overlay.ocr.base import BaseOCR
from translate_overlay.utils.onnx_decode import create_init_past_key_values, batched_beam_search_with_past_new
from translate_overlay.utils.logger import setup_logger, log_timing


logger = setup_logger()


class Florence2OCR(BaseOCR):
    """
    Florence2 OCR class for Optical Character Recognition using the Florence2 model.
    """

    def __init__(self, model_path: str, beam_size: int = None, output_region: bool = False):
        """
        Initialize the Florence2OCR class.
        """

        self.model_path = model_path
        self.beam_size = beam_size
        self.tokenizer_max_length = 1024
        self.task = "<OCR_WITH_REGION>"
        self.prompt = "What is the text in the image, with regions?"
        self.pattern = r'(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
        self.output_region = output_region
        
        self.bos_id = 0
        self.pad_id = 1
        self.eos_id = 2
        self.unk_id = 3

        # base
        self.max_length = 1024
        self.num_layers = 6
        self.num_heads = 12
        self.head_dim = 64

        # large
        # self.max_length = 4096
        # self.num_layers = 12
        # self.num_heads = 16
        # self.head_dim = 64

        self._load_model()

    
    @log_timing(logger, __name__, "Load model")
    def _load_model(self) -> None:
        """
        Load the Florence2 model and processor.
        """

        encoder_model_path = os.path.join(self.model_path, "encoder_model_uint8.onnx")
        decoder_merged_model_path = os.path.join(self.model_path, "decoder_model_merged_uint8.onnx")
        embedding_model_path = os.path.join(self.model_path, "embed_tokens_uint8.onnx")
        vision_model_path = os.path.join(self.model_path, "vision_encoder_uint8.onnx")
        tokenizer_path = os.path.join(self.model_path, "tokenizer.json")

        for model_path in [
            encoder_model_path,
            decoder_merged_model_path,
            embedding_model_path,
            vision_model_path,
            tokenizer_path
        ]:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        self.enc_session = ort.InferenceSession(encoder_model_path, sess_options=sess_options)
        self.dec_merged_session = ort.InferenceSession(decoder_merged_model_path, sess_options=sess_options)
        self.emb_session = ort.InferenceSession(embedding_model_path, sess_options=sess_options)
        self.vis_session = ort.InferenceSession(vision_model_path, sess_options=sess_options)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=self.tokenizer_max_length)
        self.tokenizer.add_tokens(
            ['<od>', '</od>', '<ocr>', '</ocr>'] + \
            [f'<loc_{x}>' for x in range(1000)] + \
            [
                '<cap>', '</cap>', '<ncap>', '</ncap>','<dcap>', '</dcap>', '<grounding>', '</grounding>', \
                '<seg>', '</seg>', '<sep>', '<region_cap>', '</region_cap>', '<region_to_desciption>', \
                '</region_to_desciption>', '<proposal>', '</proposal>', '<poly>', '</poly>', '<and>'
            ]
        )
        self.coordinates_quantizer = CoordinatesQuantizer(
            "floor",
            (1000, 1000),
        )
        

    def _merge_input_ids_with_image_features(
        self, image_features: np.ndarray, inputs_embeds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge image features with input IDs and create attention masks.
        """
        
        batch_size, image_token_length = image_features.shape[:-1]
        image_attention_mask = np.ones((batch_size, image_token_length), dtype=np.int64)

        # task_prefix_embeds: [batch_size, padded_context_length, hidden_size]
        # task_prefix_attention_mask: [batch_size, context_length]
        if inputs_embeds is None:
            return image_features, image_attention_mask

        task_prefix_embeds = inputs_embeds
        task_prefix_attention_mask = np.ones((batch_size, task_prefix_embeds.shape[1]), dtype=np.int64)

        if len(task_prefix_attention_mask.shape) == 3:
            task_prefix_attention_mask = task_prefix_attention_mask[:, 0]

        # concat [image embeds, task prefix embeds]
        inputs_embeds = np.concatenate([image_features, task_prefix_embeds], axis=1)
        attention_mask = np.concatenate([image_attention_mask, task_prefix_attention_mask], axis=1)

        return inputs_embeds, attention_mask
    

    def parse_ocr_from_text_and_spans(
            self, text, pattern, image_size, area_threshold=-1.0,
        ):
        bboxes = []
        labels = []
        text = text.replace('<s>', '')
        # ocr with regions
        parsed = re.findall(pattern, text)
        instances = []
        image_width, image_height = image_size

        for ocr_line in parsed:
            ocr_content = ocr_line[0]
            quad_box = ocr_line[1:]
            quad_box = [int(i) for i in quad_box]
            quad_box = self.coordinates_quantizer.dequantize(
                np.array(quad_box).reshape(-1, 2),
                size=image_size
            ).reshape(-1).tolist()

            if area_threshold > 0:
                x_coords = [i for i in quad_box[0::2]]
                y_coords = [i for i in quad_box[1::2]]

                # apply the Shoelace formula
                area = 0.5 * abs(sum(x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i] for i in range(4 - 1)))

                if area < (image_width * image_height) * area_threshold:
                    continue

            bboxes.append(quad_box)
            labels.append(ocr_content)
            instances.append((ocr_content, quad_box))

        return instances


    def pixel_values_normalize(self, pixel_values, mean, std) -> np.ndarray:
        """
        Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

        image = (image - mean) / std
        """

        return (pixel_values - mean) / std


    @log_timing(logger, __name__, "Inference")
    def _inference(self, inputs_text: np.ndarray, inputs_pixel_values: np.ndarray) -> np.ndarray:
        """
        Perform inference using the ONNX model.
        """

        embed_input_name = self.emb_session.get_inputs()[0].name
        embed_output = self.emb_session.run(
            None, {embed_input_name: inputs_text}
        )[0]

        vision_input_name = self.vis_session.get_inputs()[0].name
        vision_output = self.vis_session.run(
            None, {vision_input_name: inputs_pixel_values}
        )[0]

        inputs_embeds, encoder_attention_mask = self._merge_input_ids_with_image_features(
            vision_output, 
            embed_output
        )

        encoder_outputs = self.enc_session.run(None, {
            "inputs_embeds": inputs_embeds, 
            "attention_mask": encoder_attention_mask,
        })
        encoder_hidden_states = encoder_outputs[0]
        input_ids = np.array([[self.eos_id]], dtype=np.int64)
        init_past_key_values = create_init_past_key_values(self.num_heads, self.head_dim)  # (1, num_heads, 0, head_dim)

        decoder_cache_name_dict = {
            f'present.{layer}.{kv}': f'past_key_values.{layer}.{kv}'
            for layer in range(self.num_layers)
            for kv in ('encoder.key', 'encoder.value', 'decoder.key', 'decoder.value')
        }

        past_key_values = {
            cache_name: init_past_key_values
            for cache_name in decoder_cache_name_dict.values()
        }

        decoder_inputs = {
            "encoder_hidden_states": np.repeat(encoder_hidden_states, 1, axis=0),
            "encoder_attention_mask": np.repeat(encoder_attention_mask, 1, axis=0),
            "use_cache_branch": np.array([False]),
            **past_key_values
        }

        return batched_beam_search_with_past_new(
            self.dec_merged_session,
            "inputs_embeds",
            input_ids,
            decoder_inputs,
            decoder_cache_name_dict,
            self.max_length,
            self.beam_size,
            self.eos_id,
            self._embedding,
        )
    

    def _embedding(self, input_ids: np.ndarray) -> Dict:
        inputs_embeds = self.emb_session.run(None, {"input_ids": input_ids})[0]

        return {"inputs_embeds": inputs_embeds}
    

    @log_timing(logger, __name__, "Preprocess")
    def _preprocess(self, image) -> np.ndarray:
        """
        Preprocess the input image for the model.
        """
        
        # Convert image to RGB if not already in that mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        inputs = dict()

        resized_image = image.resize((768, 768), resample=3, reducing_gap=None)
        pixel_values = np.array(resized_image, dtype=np.float64) / 255
        pixel_values = pixel_values.astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        pixel_values = self.pixel_values_normalize(pixel_values, mean, std)
        pixel_values = pixel_values.transpose([2, 0, 1])
        pixel_values = np.expand_dims(pixel_values, axis=0)
        inputs["pixel_values"] = pixel_values

        prompt_ids = self.tokenizer.encode(self.prompt).ids
        prompt_ids = np.array([prompt_ids], dtype=np.int64)
        inputs["input_ids"] = prompt_ids
        
        return inputs
    

    @log_timing(logger, __name__, "Postprocess")
    def _postprocess(self, image, outputs: np.ndarray) -> str:
        """
        Postprocess the model outputs to get the final text.
        """
        # Convert the output IDs to text using the processor
        # decoded_text = self.processor.batch_decode(outputs, skip_special_tokens=False)[0]
        output_tokens = [item for item in outputs[0].tolist() if item not in [self.eos_id]]
        decoded_text = self.tokenizer.decode(output_tokens, skip_special_tokens=False)
        if decoded_text.startswith("</s>"):
            decoded_text = decoded_text.replace("</s>", "", 1)

        parsed_results = self.parse_ocr_from_text_and_spans(
            decoded_text,
            self.pattern,
            (image.width, image.height),
            0.0
        )
        
        return parsed_results


    def recognize(self, image: Any) -> List[Dict]:
        """
        Recognize text from the input image.
        
        :param image: Input image for OCR.
        :return: Recognized text.
        """
        
        # Preprocess the image
        inputs = self._preprocess(image)
        
        # Perform inference
        outputs = self._inference(inputs["input_ids"], inputs["pixel_values"])
        
        # Postprocess the outputs to get the final text
        parsed_results = self._postprocess(image, outputs)
        
        return parsed_results


class CoordinatesQuantizer(object):
    """
    Quantize coornidates (Nx2)
    """

    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def quantize(self, coordinates: np.ndarray, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size       # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        assert coordinates.shape[-1] == 2, 'coordinates should be shape (N, 2)'
        x = coordinates[..., 0:1]  # Shape: (N, 1)
        y = coordinates[..., 1:2]  # Shape: (N, 1)

        if self.mode == 'floor':
            quantized_x = np.floor(x / size_per_bin_w).clip(0, bins_w - 1)
            quantized_y = np.floor(y / size_per_bin_h).clip(0, bins_h - 1)

        elif self.mode == 'round':
            raise NotImplementedError()

        else:
            raise ValueError('Incorrect quantization type.')

        quantized_coordinates = np.concatenate(
            (quantized_x, quantized_y), axis=-1
        ).astype(int)

        return quantized_coordinates

    def dequantize(self, coordinates: np.ndarray, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size       # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        assert coordinates.shape[-1] == 2, 'coordinates should be shape (N, 2)'
        x = coordinates[..., 0:1]  # Shape: (N, 1)
        y = coordinates[..., 1:2]  # Shape: (N, 1)

        if self.mode == 'floor':
            # Add 0.5 to use the center position of the bin as the coordinate.
            dequantized_x = (x + 0.5) * size_per_bin_w
            dequantized_y = (y + 0.5) * size_per_bin_h

        elif self.mode == 'round':
            raise NotImplementedError()

        else:
            raise ValueError('Incorrect quantization type.')

        dequantized_coordinates = np.concatenate(
            (dequantized_x, dequantized_y), axis=-1
        )

        return dequantized_coordinates


if __name__ == "__main__":
    from PIL import Image
    from translate_overlay.utils.misc import draw_boxes
    
    # Example usage
    model_path = sys.argv[1]
    # url = "http://ecx.images-amazon.com/images/I/51UUzBDAMsL.jpg?download=true"
    # image = Image.open("E:\\work\\1-personal\\Florence-2-base\\Screenshot_2025-05-18_172138.png")
    image = Image.open(sys.argv[2])

    ocr = Florence2OCR(model_path, beam_size=3, output_region=True)
    reco_result = ocr.recognize(image)
    boxes_xyxy = [(i[1][0], i[1][1], i[1][4], i[1][5]) for i in reco_result]
    logger.info(reco_result)
    image = draw_boxes(image, boxes_xyxy)
    image.show()
