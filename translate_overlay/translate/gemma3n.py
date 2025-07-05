import os
import re
import sys
import time
import json
from typing import List

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from translate_overlay.translate.base import BaseTranslator
from translate_overlay.utils.onnx_decode import create_init_past_key_values, batched_beam_search_with_past_new
from translate_overlay.utils.logger import setup_logger


logger = setup_logger()


TARGET_LANG_MAP = {
    "English": "English",
    "Spanish": "Spanish",
    "French": "French",
    "German": "German",
    "Italian": "Italian",
    "Portuguese": "Portuguese",
    "Dutch": "Dutch",
    "Cantonese": "Cantonese",
    "Taiwanese": "Taiwanese",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "Arabic": "Arabic",
    "Turkish": "Turkish",
    "Hindi": "Hindi",
    "Ukrainian": "Ukrainian",
    "Russian": "Russian",
    "Simplified Chinese": "Simplified Chinese",
    # Add more language mappings as needed
}

class Gemma3nTranslator(BaseTranslator):
    def __init__(self, model_path: str, beam_size: int = None):
        """
        Initialize the Gemma3Translator with the path to the local model.

        :param model_path: Path to the local Gemma3 model directory.
        """

        self.model_path = model_path
        self.beam_size = beam_size
        self.tokenizer = None

        # inputs
        # '<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite me a poem about Machine Learning.<end_of_turn>\n<start_of_turn>model\n'
        #  {'input_ids': array([[     2,    105,   2364,    107,   3048,    659,    496,  11045,
        #          16326, 236761,    108,   6974,    786,    496,  27355,   1003,
        #          15313,  19180, 236761,    106,    107,    105,   4368,    107]]), 
        # 'attention_mask': array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
        self.eos_id = 106  # <end_of_turn>

        self.max_length = 1000

        self.num_layers = 30
        self.num_heads = 2
        self.head_dim = 256
        
        t0 = time.time()
        self._load_model()
        t1 = time.time()
        logger.info(f"Load model: {t1 - t0:.4f} seconds")

        # self.prompt_template = (
        #     '<start_of_turn>user\nYou are a professional translator. Please help translating input text to {language} '
        #     'Requirements: '
        #     '1. Identify the source langauge from input text. '
        #     '2. Translate all parts in the input text to {language} accurately. '
        #     '3. Put traslated text in a json with format: '
        #     '```json{{"SourceLang": (source language you identiied here), "TargetLang": {language}, "Translation": (translated {language} text here)}}```. '
        #     '4. Do not output additional information, only json is needed. '
        #     # '\n'
        #     'Example: '
        #     'User input: Please translate: "..." '
        #     'Output: ```json{{"SourceLang": "...", "TargetLang": "{language}", "Translation": "..."}}``` '
        #     'End of instructions, requirements and exampls. Start of actual input \n\n'
        #     'User input: Please translate: "{text}"'
        #     '<end_of_turn>\n<start_of_turn>model\n'
        # )
        self.prompt_template = (
            '<start_of_turn>user\nYou are a professional translator.\n'
            '### Task: Please help translating input text to {language}.\n'
            '### Requirements:\n'
            '1. Identify the source langauge from input text.\n'
            '2. Translate all parts in the input text to {language} accurately.\n'
            '3. Return a json with fields: "source_language", "target_language", and "translation".\n'
            '4. Do not output additional information, only json is needed.\n'
            '### Example:\n'
            'Input: Please translate: "Cooking the Perfect 10-Pound Turkey: A Comprehensive Guide"\n'
            'Output: ```json{{"source_language": "English", '
            '"target_language": "Japanese", '
            '"translation": "完璧な10ポンドの鶏肉を調理する。完全なガイド"}}```\n'
            # 'End of instructions, requirements and exampls. Start of actual input \n\n'
            '\n'
            # '### User input:\n'
            'Please translate: "{text}"\n'
            '<end_of_turn>\n<start_of_turn>model\n'
        )
        self.result_regex = re.compile(r'```json(.+?)```', re.S)


    def _load_model(self) -> None:
        """
        Load the ONNX model from the specified path.

        :return: None
        """

        embed_model_path = os.path.join(self.model_path, "embed_tokens_uint8.onnx")
        decoder_merged_model_path = os.path.join(self.model_path, "decoder_model_merged_q4.onnx")
        tokenizer_path = os.path.join(self.model_path, "tokenizer.json")

        for model_path in [decoder_merged_model_path, tokenizer_path, embed_model_path]:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
        
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        self.embed_session = ort.InferenceSession(embed_model_path, sess_options=sess_options)
        self.dec_merged_session = ort.InferenceSession(decoder_merged_model_path, sess_options=sess_options)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)


    def _inference(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Perform inference using the ONNX model.
        If beam_size is None, use greedy search; otherwise, use beam search.

        :param input_ids: Input IDs for the model.
        :return: Best sequence of token IDs.
        """

        init_past_key_values = create_init_past_key_values(self.num_heads, self.head_dim)  # (1, num_heads, 0, head_dim)

        decoder_cache_name_dict = {
            f'present.{layer}.{kv}': f'past_key_values.{layer}.{kv}'
            for layer in range(self.num_layers)
            for kv in ('key', 'value')
        }

        past_key_values = {
            cache_name: init_past_key_values
            for cache_name in decoder_cache_name_dict.values()
        }

        decoder_inputs = {
            "position_ids": np.tile(np.arange(1, input_ids.shape[-1] + 1), (1, 1)),
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
    

    def _embedding(self, input_ids: np.ndarray) -> np.ndarray:
        inputs_embeds, per_layer_inputs = self.embed_session.run(None, {"input_ids": input_ids})

        return {"inputs_embeds": inputs_embeds, "per_layer_inputs": per_layer_inputs}
    

    def encode_tokens(self, text: str) -> List:
        return self.tokenizer.encode(text).tokens
    

    def decode_tokens(self, tokens: List) -> str:
        return self.tokenizer.decode([self.tokenizer.token_to_id(token) for token in tokens])


    def _encode_token_ids(self, text: str) -> List:
        """
        Tokenize the input text.

        :param text: The text to tokenize.
        :return: Tokenized text as a list.
        """

        return self.tokenizer.encode(text).ids
    

    def _decode_token_ids(self, token_ids: List) -> str:
        """
        Decode the tokenized text back to string.

        :param tokens: The tokenized text as a numpy array.
        :return: Decoded text as a string.
        """

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    

    def _parse_response(self, model_response: str) -> str:
        match = self.result_regex.search(model_response.replace('\n', ' '))

        if match:
            try:
                result = json.loads(match.group(1).strip())
            except Exception:
                result = dict()

            if "source_language" not in result:
                result["source_language"] = "N/A"
            logger.info(f"Model recognized source language: {result["source_language"]}")
            if "target_language" not in result:
                result["target_language"] = "N/A"
            logger.info(f"Model recognized target language: {result["target_language"]}")

            if "translation" in result:
                return result["translation"]

        logger.warning("Unable to parse the model response:")
        logger.warning(model_response)
        return "Translation failed"


    def translate(self, text: str, target_lang: str, source_lang: str) -> str:
        """
        Translate text from source_lang to target_lang using Madlad API.

        :param text: The text to translate.
        :param source_lang: The source language code.
        :param target_lang: The target language code.
        :return: The translated text.
        """
        
        assert target_lang in TARGET_LANG_MAP, f"Unsupported target language: {target_lang}"

        t0 = time.time()
        text = self.prompt_template.format(language=target_lang, text=text)
        input_tokens = self._encode_token_ids(text)
        self.max_length = len(input_tokens) * 2
        input_ids = np.array([input_tokens], dtype=np.int64)
        t1 = time.time()
        logger.info(f"Tokenization: {t1 - t0:.4f} seconds")

        t2 = time.time()
        output_ids = self._inference(input_ids)
        t3 = time.time()
        logger.info(f"Inference: {t3 - t2:.4f} seconds")

        output_tokens = [item for item in output_ids[0].tolist() if item not in [self.eos_id]]
        output_text = self._decode_token_ids(output_tokens)
        output_text = self._parse_response(output_text)
        t4 = time.time()
        logger.info(f"Decoding: {t4 - t3:.4f} seconds")

        logger.info(f"Total: {t4 - t0:.4f} seconds")

        return output_text


    @staticmethod
    def source_lang_list() -> list:
        """
        Get the list of supported source languages.

        :return: List of supported source languages.
        """

        return ['Any']


    @staticmethod
    def target_lang_list() -> list:
        """
        Get the list of supported languages.

        :return: List of supported languages.
        """

        return sorted(list(TARGET_LANG_MAP.keys()))


if __name__ == "__main__":
    # Example usage
    text = "SciPy is a Python module that provides algorithms for mathematics, science, and engineering. It works with NumPy arrays and offers routines for statistics, optimization, integration, linear algebra, and more."
    text = "Cooking the Perfect 10-Pound Turkey: A Comprehensive Guide"
    text = "The Ultimate Guide to Cooking Meatloaf: Temperature and Time"
    # text = "Stir-fries are a staple in many cuisines, and adding eggs to the mix can elevate the dish to a whole new level. However, cooking eggs in a stir-fry can be a bit tricky, especially for beginners. In this article, we will explore the different techniques and tips for cooking eggs in a stir-fry, so you can achieve a perfect, fluffy, and delicious result every time."
    # model_path = r"E:\work\1-personal\madlad400-3b-mt\onnx"
    model_path = sys.argv[1]
    target_lang = sys.argv[2]
    translator = Gemma3nTranslator(model_path=model_path, beam_size=3)

    if target_lang == "test":
        for target_lang in TARGET_LANG_MAP.values():
            print(target_lang)
            translated_text = translator.translate(text, target_lang, "Any")
            logger.info(translated_text)

    else:
        translated_text = translator.translate(text, target_lang, "Any")
        logger.info(translated_text)
    


