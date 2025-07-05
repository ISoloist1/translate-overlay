import os
import sys
import time
import numpy as np
import onnxruntime as ort
from typing import Any, List, Dict, Tuple
from tokenizers import Tokenizer
from transformers import AutoProcessor

from translate_overlay.ocr.base import BaseOCR
from translate_overlay.utils.onnx_decode import create_init_past_key_values, greedy_search, batched_beam_search_with_past
from translate_overlay.utils.logger import setup_logger


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

        t0 = time.time()
        self._load_model()
        t1 = time.time()
        logger.info(f"Load model: {t1 - t0:.4f} seconds")

    
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

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        

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
        init_past_key_values = create_init_past_key_values(self.num_heads, self.head_dim).squeeze(0)  # (1, num_heads, 0, head_dim)
        
        if self.beam_size is None:
            # Greedy search
            return greedy_search(
                self.dec_merged_session,
                encoder_hidden_states,
                encoder_attention_mask,
                self.max_length,
                self.eos_id,
                self.eos_id,
            )
        else:
            # Beam search
            return batched_beam_search_with_past(
                self.dec_merged_session,
                "inputs_embeds",
                encoder_hidden_states,
                encoder_attention_mask,
                init_past_key_values,
                self.max_length,
                self.beam_size,
                self.eos_id,
                self.eos_id,
                self.num_layers,
                self.emb_session,
            )
    

    def _preprocess(self, image) -> np.ndarray:
        """
        Preprocess the input image for the model.
        """
        
        # Convert image to RGB if not already in that mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize and normalize the image
        inputs = self.processor(
            self.prompt, 
            image, 
            return_tensors="np",
        )

        prompt_ids = self.tokenizer.encode(self.prompt).ids
        prompt_ids = np.array([prompt_ids], dtype=np.int64)
        inputs["input_ids"] = prompt_ids
        
        return inputs
    

    def _postprocess(self, image, outputs: np.ndarray) -> str:
        """
        Postprocess the model outputs to get the final text.
        """
        # Convert the output IDs to text using the processor
        decoded_text = self.processor.batch_decode(outputs, skip_special_tokens=False)[0]
        if decoded_text.startswith("</s>"):
            decoded_text = decoded_text.replace("</s>", "", 1)
        
        parsed_results = self.processor.post_process_generation(
            decoded_text, 
            task=self.task, 
            image_size=(image.width, image.height)
        )

        if self.output_region:
            parsed_results = list(zip(parsed_results[self.task]["labels"], parsed_results[self.task]["quad_boxes"]))
        else:
            parsed_results = parsed_results[self.task]["labels"]
        
        return parsed_results


    def recognize(self, image: Any) -> List[Dict]:
        """
        Recognize text from the input image.
        
        :param image: Input image for OCR.
        :return: Recognized text.
        """
        
        # Preprocess the image
        t0 = time.time()
        inputs = self._preprocess(image)
        t1 = time.time()
        logger.info(f"Preprocess: {t1 - t0:.4f} seconds")
        
        # Perform inference
        t2 = time.time()
        outputs = self._inference(inputs["input_ids"], inputs["pixel_values"])
        t3 = time.time()
        logger.info(f"Inference: {t3 - t2:.4f} seconds")
        
        # Postprocess the outputs to get the final text
        t4 = time.time()
        parsed_results = self._postprocess(image, outputs)
        t5 = time.time()
        logger.info(f"Postprocess: {t5 - t4:.4f} seconds")
        
        return parsed_results


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
