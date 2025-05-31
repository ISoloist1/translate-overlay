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
        print(f"Load model: {t1 - t0:.4f} seconds")


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
            # return self._beam_search(encoder_hidden_states, encoder_attention_mask)
            # return self._beam_search_with_past(encoder_hidden_states, encoder_attention_mask)
            # return self._batched_beam_search_with_past(encoder_hidden_states, encoder_attention_mask)
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


    def _greedy_search(self, encoder_hidden_states: np.ndarray, encoder_attention_mask: np.ndarray) -> np.ndarray:
        # 2. Prepare decoder input (start with <pad> or <bos>)
        decoder_input_ids = np.array([[self.unk_id]], dtype=np.int64)

        # 3. Run decoder step-by-step (greedy decoding)
        for _ in range(self.max_length):
            decoder_outputs = self.dec_session.run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": encoder_attention_mask,
                }
            )
            next_token_logits = decoder_outputs[0][:, -1, :]
            next_token_id = np.argmax(next_token_logits, axis=-1)
            decoder_input_ids = np.concatenate(
                [decoder_input_ids, next_token_id[:, None]], axis=-1
            )
            if next_token_id[0] == self.eos_id:
                break

        return decoder_input_ids
    

    def _init_past_key_values(self, batch_size: int = 1) -> dict:
        """
        Initialize past_key_values for the decoder.
        :param batch_size: Batch size for the past_key_values.
        :return: Initialized past_key_values.
        """
        
        num_heads = 16  # or your model's value
        seq_len = 0     # for first step, usually 0 or 1
        head_dim = 128  # or your model's value
        kv_cache = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
        
        return kv_cache
    

    def _batched_beam_search_with_past(self, encoder_hidden_states, encoder_attention_mask) -> np.ndarray:
        """
        Batched beam search for decoder with past_key_values support (merged decoder model).
        """
        num_layers = 32
        kv_cache = self._init_past_key_values().squeeze(0)  # (1, num_heads, 0, head_dim)

        # Each beam: (score, sequence, past_key_values)
        beams = [(0.0, np.array([[self.unk_id]], dtype=np.int64), None)]
        finished = []

        for step in range(self.max_length):
            # Prepare batch inputs for all beams
            batch_input_ids = np.concatenate([seq[:, -1:] for _, seq, _ in beams], axis=0)  # (beam, 1)
            batch_size = len(beams)
            batch_past = []

            for _, _, past in beams:
                if past is None:
                    batch_past.append(kv_cache)
                else:
                    batch_past.append(past)

            # Prepare ONNX inputs
            inputs = {
                "input_ids": batch_input_ids,
                "encoder_hidden_states": np.repeat(encoder_hidden_states, batch_size, axis=0),
                "encoder_attention_mask": np.repeat(encoder_attention_mask, batch_size, axis=0),
                "use_cache_branch": np.array([False if past is None else True])
            }
            # Add all past_key_values, stacking along batch dimension
            for i in range(num_layers):
                for typ in ["encoder.key", "encoder.value", "decoder.key", "decoder.value"]:
                    key = f"past_key_values.{i}.{typ}"
                    stacked = np.stack([p[f"present.{i}.{typ}"] if p is not None and f"present.{i}.{typ}" in p else kv_cache
                                    for p in batch_past], axis=0)
                    inputs[key] = stacked

            # Run decoder
            outputs = self.dec_merged_session.run(None, inputs)
            next_token_logits = outputs[0][:, -1, :]  # (beam, vocab_size)
            output_names = [node.name for node in self.dec_merged_session.get_outputs()]
            output_dicts = []
            for b in range(batch_size):
                output_dict = {}
                for idx, name in enumerate(output_names[1:]):  # skip logits
                    if outputs[idx + 1].shape[0] == 0:
                        output_dict[name] = batch_past[b][name]
                    else:
                        output_dict[name] = outputs[idx + 1][b]
                output_dicts.append(output_dict)

            all_candidates = []
            for beam_idx, (score, seq, past) in enumerate(beams):
                if seq[0, -1] == self.eos_id:
                    finished.append((score, seq))
                    continue

                log_probs = log_softmax(next_token_logits[beam_idx], axis=-1)
                topk_ids = np.argsort(log_probs)[::-1][:self.beam_size]

                for token_id in topk_ids:
                    new_seq = np.concatenate([seq, [[token_id]]], axis=-1)
                    new_score = score + log_probs[token_id]
                    new_past = output_dicts[beam_idx]
                    all_candidates.append((new_score, new_seq, new_past))

            # Keep only the best beam_size sequences
            beams = heapq.nlargest(self.beam_size, all_candidates, key=lambda x: x[0])

            # Early stopping
            if finished:
                best_finished_score = max(score for score, _ in finished)
                # Check if any unfinished beam is better than the best finished beam
                best_unfinished_score = max(
                    score for score, _, _ in beams
                )
                if best_finished_score >= best_unfinished_score:
                    break

            if all(seq[0, -1] == self.eos_id for _, seq, _ in beams):
                finished.extend(beams)
                break

        if finished:
            return max(finished, key=lambda x: x[0])[1]
        else:
            return beams[0][1]
    

    def _beam_search_with_past(self, encoder_hidden_states, encoder_attention_mask) -> np.ndarray:
        """
        Beam search for decoder with past_key_values support (merged decoder model).
        """

        # Each beam: (score, sequence, past_key_values)
        beams = [(0.0, np.array([[self.unk_id]], dtype=np.int64), None)]
        finished = []
        kv_cache = self._init_past_key_values()
        num_layers = 32

        for _ in range(self.max_length):
            # print(f"iteration {_}")

            all_candidates = []
            for score, seq, past in beams:
                if seq[0, -1] == self.eos_id:
                    finished.append((score, seq))
                    continue
                # print(f"iteration {_}")

                # Prepare inputs
                inputs = {
                    "input_ids": seq[:, -1:],  # Only last token
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": encoder_attention_mask,
                    "use_cache_branch": np.array([True])
                }

                if past is None:
                    inputs["use_cache_branch"] = np.array([False])

                # Subsequent steps: use past_key_values
                # Add past_key_values to inputs
                for i in range(num_layers):
                    inputs[f"past_key_values.{i}.encoder.key"] = kv_cache \
                        if past is None else past[f"present.{i}.encoder.key"]
                    inputs[f"past_key_values.{i}.encoder.value"] = kv_cache \
                        if past is None else past[f"present.{i}.encoder.value"]
                    inputs[f"past_key_values.{i}.decoder.key"] = kv_cache \
                        if past is None else past[f"present.{i}.decoder.key"]
                    inputs[f"past_key_values.{i}.decoder.value"] = kv_cache \
                        if past is None else past[f"present.{i}.decoder.value"]

                # Run decoder
                outputs = self.dec_merged_session.run(None, inputs)
                next_token_logits = outputs[0][:, -1, :]  # (1, vocab_size)
                # Extract new past_key_values from outputs (assuming they are outputs[1:])
                new_past = dict(zip([node.name for node in self.dec_merged_session.get_outputs()],
                                    outputs))
                # If batch size is 0, use the original past
                for key in new_past:
                    if new_past[key].shape[0] == 0:
                        new_past[key] = past[key]

                log_probs = log_softmax(next_token_logits, axis=-1)
                topk_ids = np.argsort(log_probs[0])[::-1][:self.beam_size]

                for token_id in topk_ids:
                    new_seq = np.concatenate([seq, [[token_id]]], axis=-1)
                    new_score = score + log_probs[0, token_id]
                    all_candidates.append((new_score, new_seq, new_past))

            # Keep only the best beam_size sequences
            beams = heapq.nlargest(self.beam_size, all_candidates, key=lambda x: x[0])

            # After updating beams and finished
            if len(finished) >= self.beam_size * 2:
                best_finished_score = max(score for score, _ in finished)

                # Check if any unfinished beam is better than the best finished beam
                best_unfinished_score = max(
                    score for score, _, _ in beams
                )

                # If the best finished beam is at least as good as any unfinished, stop
                if best_finished_score >= best_unfinished_score:
                    break

            # Early stop if all beams finished
            if all(seq[0, -1] == self.eos_id for _, seq, _ in beams):
                finished.extend(beams)
                break

        # Return the best finished sequence
        if finished:
            return max(finished, key=lambda x: x[0])[1]
        else:
            return beams[0][1]


    def _beam_search(self, encoder_hidden_states: np.ndarray, encoder_attention_mask: np.ndarray) -> np.ndarray:
        # Each beam is a tuple: (score, sequence)
        beams = [(0.0, np.array([[self.unk_id]], dtype=np.int64))]

        for _ in range(self.max_length):
            all_candidates = []
            for score, seq in beams:
                if seq[0, -1] == self.eos_id:
                    # If EOS already generated, keep as is
                    all_candidates.append((score, seq))
                    continue

                decoder_outputs = self.dec_session.run(
                    None,
                    {
                        "input_ids": seq,
                        "encoder_hidden_states": encoder_hidden_states,
                        "encoder_attention_mask": encoder_attention_mask,
                    }
                )
                next_token_logits = decoder_outputs[0][:, -1, :]  # (1, vocab_size)
                log_probs = log_softmax(next_token_logits, axis=-1)  # (1, vocab_size)
                topk_ids = np.argsort(log_probs[0])[::-1][:self.beam_size]

                for token_id in topk_ids:
                    new_seq = np.concatenate([seq, [[token_id]]], axis=-1)
                    new_score = score + log_probs[0, token_id]
                    all_candidates.append((new_score, new_seq))

            # Keep only the best beam_size sequences
            beams = heapq.nlargest(self.beam_size, all_candidates, key=lambda x: x[0])

            # If all beams end with EOS, stop early
            if all(seq[0, -1] == self.eos_id for _, seq in beams):
                break

        # Return the sequence with the highest score
        return beams[0][1]


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
        print(f"Tokenization: {t1 - t0:.4f} seconds")

        t2 = time.time()
        output_ids = self._inference(input_ids)
        t3 = time.time()
        print(f"Inference: {t3 - t2:.4f} seconds")

        output_tokens = [item for item in output_ids[0].tolist() if item not in [self.unk_id, self.bos_id, self.eos_id]]
        output_text = self._decode_tokens(output_tokens)
        t4 = time.time()
        print(f"Decoding: {t4 - t3:.4f} seconds")

        print(f"Total: {t4 - t0:.4f} seconds")

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
    print(translated_text)
