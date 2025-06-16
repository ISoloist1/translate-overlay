import os
import sys
import time
import heapq
import numpy as np
import onnxruntime as ort
import sentencepiece as spm

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)
from translate.base import BaseTranslator
from utils.onnx_decode import log_softmax, create_init_past_key_values, batched_beam_search_with_past
from utils.logger import setup_logger


logger = setup_logger()


# Madlad 400 Dataset & languages list: https://arxiv.org/pdf/2309.04662
# IETF BCP 47 language tag: https://en.wikipedia.org/wiki/IETF_language_tag
TARGET_LANG_MAP = {
    "English": "<2en>",
    "Spanish": "<2es>",
    "French": "<2fr>",
    "German": "<2de>",
    "Italian": "<2it>",
    "Portuguese": "<2pt>",
    "Dutch": "<2nl>",
    "Russian": "<2ru>",
    "Chinese": "<2zh>",
    "Japanese": "<2ja>",
    "Korean": "<2ko>",
    "Arabic": "<2ar>",
    "Turkish": "<2tr>",
    "Hindi": "<2hi>",
    # Add more language mappings as needed
}


class MadladTranslator(BaseTranslator):
    def __init__(self, model_path: str, beam_size: int = None):
        """
        Initialize the MadladTranslator with the path to the local model.

        :param model_path: Path to the local Madlad model directory.
        """

        self.model_path = model_path
        self.beam_size = beam_size
        self.sp_model = spm.SentencePieceProcessor()
        # tensor([805, 116, 908, 10108, 88792, 918, 2]])
        # >>> sp_model.EncodeAsIds("<2pt> I love pizza!")
        # [805, 116, 908, 10108, 88792, 918]
        self.unk_id = 0
        self.bos_id = 1
        self.eos_id = 2

        self.max_length = 100

        self.num_layers = 32
        self.num_heads = 16
        self.head_dim = 128
        
        t0 = time.time()
        self._load_model()
        t1 = time.time()
        logger.info(f"Load model: {t1 - t0:.4f} seconds")


    def _load_model(self) -> None:
        """
        Load the ONNX model from the specified path.

        :return: None
        """

        encoder_model_path = os.path.join(self.model_path, "encoder_model.onnx")
        decoder_model_path = os.path.join(self.model_path, "decoder_model.onnx")
        decoder_merged_model_path = os.path.join(self.model_path, "decoder_model_merged.onnx")
        spm_model_path = os.path.join(self.model_path, "spiece.model")

        for model_path in [encoder_model_path, decoder_model_path, decoder_merged_model_path, spm_model_path]:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
        
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        self.enc_session = ort.InferenceSession(encoder_model_path, sess_options=sess_options)
        # self.dec_session = ort.InferenceSession(decoder_model_path)
        self.dec_merged_session = ort.InferenceSession(decoder_merged_model_path, sess_options=sess_options)
        self.sp_model.Load(spm_model_path)


    def _inference(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Perform inference using the ONNX model.
        If beam_size is None, use greedy search; otherwise, use beam search.

        :param input_ids: Input IDs for the model.
        :return: Best sequence of token IDs.
        """

        # 1. Prepare input for encoder
        encoder_attention_mask = np.ones(input_ids.shape, dtype=np.int64)  # (1, seq_len)

        # 1. Run encoder
        encoder_outputs = self.enc_session.run(
            None, {
                "input_ids": input_ids,
                "attention_mask": encoder_attention_mask,
            }
        )
        encoder_hidden_states = encoder_outputs[0]
        init_past_key_values = create_init_past_key_values(self.num_heads, self.head_dim).squeeze(0)  # (1, num_heads, 0, head_dim)
        
        if self.beam_size is None:
            # Greedy search
            return self._greedy_search(encoder_hidden_states, encoder_attention_mask)
        else:
            # Beam search
            return batched_beam_search_with_past(
                self.dec_merged_session,
                "input_ids",
                encoder_hidden_states,
                encoder_attention_mask,
                init_past_key_values,
                self.max_length,
                self.beam_size,
                self.unk_id,
                self.eos_id,
                self.num_layers,
            )


    def _tokenize(self, text: str) -> np.ndarray:
        """
        Tokenize the input text using the SentencePiece model.

        :param text: The text to tokenize.
        :return: Tokenized text as a numpy array.
        """

        tokens = self.sp_model.EncodeAsIds(text)
        tokens.append(self.eos_id)
        return tokens
    

    def _decode_tokens(self, tokens: np.ndarray) -> str:
        """
        Decode the tokenized text back to string using the SentencePiece model.

        :param tokens: The tokenized text as a numpy array.
        :return: Decoded text as a string.
        """

        return self.sp_model.DecodeIds(tokens)


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
        text = f"{TARGET_LANG_MAP[target_lang]} " + text.strip()
        input_tokens = self._tokenize(text)
        self.max_length = len(input_tokens) * 2
        input_ids = np.array([input_tokens], dtype=np.int64)
        t1 = time.time()
        logger.info(f"Tokenization: {t1 - t0:.4f} seconds")

        t2 = time.time()
        output_ids = self._inference(input_ids)
        t3 = time.time()
        logger.info(f"Inference: {t3 - t2:.4f} seconds")

        output_tokens = [item for item in output_ids[0].tolist() if item not in [self.unk_id, self.bos_id, self.eos_id]]
        output_text = self._decode_tokens(output_tokens)
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
    # model_path = r"E:\work\1-personal\madlad400-3b-mt\onnx"
    model_path = sys.argv[1]
    translator = MadladTranslator(model_path=model_path, beam_size=3)
    translated_text = translator.translate(text, "Chinese", "Any")
    logger.info(translated_text)

