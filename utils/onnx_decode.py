import heapq
import numpy as np


def log_softmax(x, axis=-1) -> np.ndarray:
    """
    Compute the log softmax of the input tensor.
    :param x: Input tensor.
    :param axis: Axis along which to compute the softmax.
    :return: Log softmax of the input tensor.
    """

    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def create_init_past_key_values(num_heads: int, head_dim: int) -> dict:
    """
    Initialize past_key_values for the decoder.
    :param num_heads: Number of attention heads.
    :param head_dim: Dimension of each attention head.
    :return: Initialized past_key_values.
    """

    seq_len = 0     # for first step, usually 0 or 1
    batch_size = 1  # for first step, usually 1
    init_past_key_values = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
    
    return init_past_key_values


def greedy_search(
    dec_merged_session: object, 
    encoder_hidden_states: np.ndarray, 
    encoder_attention_mask: np.ndarray,
    max_length: int,
    start_id: int,
    eos_id: int,
) -> np.ndarray:
    # 2. Prepare decoder input (start with <pad> or <bos>)
    decoder_input_ids = np.array([[start_id]], dtype=np.int64)

    # 3. Run decoder step-by-step (greedy decoding)
    for _ in range(max_length):
        decoder_outputs = dec_merged_session.run(
            None,
            {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": encoder_attention_mask,
                "use_cache_branch": np.array([False]),
            }
        )
        next_token_logits = decoder_outputs[0][:, -1, :]
        next_token_id = np.argmax(next_token_logits, axis=-1)
        decoder_input_ids = np.concatenate(
            [decoder_input_ids, next_token_id[:, None]], axis=-1
        )
        if next_token_id[0] == eos_id:
            break

    return decoder_input_ids


def batched_beam_search_with_past(
    dec_merged_session: object, 
    input_name: str,
    encoder_hidden_states: np.ndarray, 
    encoder_attention_mask: np.ndarray,
    init_past_key_values: np.ndarray,
    max_length: int,
    beam_size: int,
    start_id: int,
    eos_id: int,
    num_layers: int,
    embed_session: object = None,
) -> np.ndarray:
    """
    Batched beam search for decoder with past_key_values support (merged decoder model).
    """

    # Each beam: (score, sequence, past_key_values)
    beams = [(0.0, np.array([[start_id]], dtype=np.int64), None)]
    finished = []

    for _ in range(max_length):
        # Prepare batch inputs for all beams
        batch_input_ids = np.concatenate([seq[:, -1:] for _, seq, _ in beams], axis=0)  # (beam, 1)
        if embed_session is not None:
            embed_input_name = embed_session.get_inputs()[0].name
            batch_input_ids = embed_session.run(
                None, {embed_input_name: batch_input_ids}
            )[0]
        batch_size = len(beams)
        batch_past = []

        for _, _, past in beams:
            if past is None:
                batch_past.append(init_past_key_values)
            else:
                batch_past.append(past)

        # Prepare ONNX inputs
        inputs = {
            input_name: batch_input_ids,
            "encoder_hidden_states": np.repeat(encoder_hidden_states, batch_size, axis=0),
            "encoder_attention_mask": np.repeat(encoder_attention_mask, batch_size, axis=0),
            "use_cache_branch": np.array([False if past is None else True])
        }
        # Add all past_key_values, stacking along batch dimension
        for i in range(num_layers):
            for typ in ["encoder.key", "encoder.value", "decoder.key", "decoder.value"]:
                key = f"past_key_values.{i}.{typ}"
                stacked = np.stack([p[f"present.{i}.{typ}"] if p is not None and f"present.{i}.{typ}" in p else init_past_key_values
                                for p in batch_past], axis=0)
                inputs[key] = stacked

        # Run decoder
        outputs = dec_merged_session.run(None, inputs)
        next_token_logits = outputs[0][:, -1, :]  # (beam, vocab_size)
        output_names = [node.name for node in dec_merged_session.get_outputs()]
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
            if seq.shape[1] > 2 and seq[0, -1] == eos_id:
                finished.append((score, seq))
                continue

            log_probs = log_softmax(next_token_logits[beam_idx], axis=-1)
            topk_ids = np.argsort(log_probs)[::-1][:beam_size]

            for token_id in topk_ids:
                new_seq = np.concatenate([seq, [[token_id]]], axis=-1)
                new_score = score + log_probs[token_id]
                new_past = output_dicts[beam_idx]
                all_candidates.append((new_score, new_seq, new_past))

        # Keep only the best beam_size sequences
        beams = heapq.nlargest(beam_size, all_candidates, key=lambda x: x[0])

        # Early stopping
        if finished:
            best_finished_score = max(score for score, _ in finished)
            # Check if any unfinished beam is better than the best finished beam
            best_unfinished_score = max(
                score for score, _, _ in beams
            )
            if best_finished_score >= best_unfinished_score:
                break

        if all(seq[0, -1] == eos_id for _, seq, _ in beams):
            finished.extend(beams)
            break

    if finished:
        return max(finished, key=lambda x: x[0])[1]
    else:
        return beams[0][1]

