"""Frozen XLM-R word encoding; complete-word chunks and mean subword pooling."""

import hashlib
import json
import time


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Encoding deadline reached")


DIMENSION = 768
CONTENT_CAPACITY = 510


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def text_key(words):
    if (
        not isinstance(words, list)
        or not words
        or not all(isinstance(w, str) and w.strip() for w in words)
    ):
        raise ValueError("Complete original words must be a nonempty list of strings")
    # No normalization: this key preserves the exact canonical AOI tokenization.
    return _digest(words)


def word_chunks(tokenizer, words, capacity=CONTENT_CAPACITY):
    """Preserve every content token; split only between complete source words."""
    text_key(words)
    if type(capacity) is not int or not 1 <= capacity <= CONTENT_CAPACITY:
        raise ValueError("Content capacity must be an integer in 1..510")
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    ids, owners = encoding["input_ids"], encoding.word_ids()
    if len(ids) != len(owners) or not ids:
        raise ValueError("Tokenizer content IDs and word ownership disagree")
    by_word = [[] for _ in words]
    previous_owner = -1
    for token, owner in zip(ids, owners):
        if (
            type(owner) is not int
            or not 0 <= owner < len(words)
            or owner < previous_owner
        ):
            raise ValueError("Tokenizer returned unmapped or reordered content tokens")
        if type(token) is not int or token < 0:
            raise ValueError("Tokenizer returned an invalid token ID")
        previous_owner = owner
        by_word[owner].append(token)
    if any(not tokens for tokens in by_word):
        raise ValueError("Tokenizer dropped a source word")
    if any(len(tokens) > capacity for tokens in by_word):
        raise ValueError(
            "A complete source word exceeds the content-token capacity; no truncation"
        )
    chunks, chunk_ids, chunk_owners = [], [], []
    for word_index, tokens in enumerate(by_word):
        if chunk_ids and len(chunk_ids) + len(tokens) > capacity:
            chunks.append((chunk_ids, chunk_owners))
            chunk_ids, chunk_owners = [], []
        chunk_ids.extend(tokens)
        chunk_owners.extend([word_index] * len(tokens))
    if chunk_ids:
        chunks.append((chunk_ids, chunk_owners))
    return chunks


def encode_words(model, tokenizer, words, deadline_monotonic, device="cuda"):
    """Encode one complete text; return CPU float32 words and coverage counts."""
    import torch

    chunks = word_chunks(tokenizer, words)
    device_type = torch.device(device).type
    sums = torch.zeros((len(words), DIMENSION), dtype=torch.float32, device=device)
    counts = torch.zeros(len(words), dtype=torch.float32, device=device)
    n_subwords = 0
    with torch.inference_mode():
        for ids, owners in chunks:
            check_deadline(deadline_monotonic)
            full_ids = tokenizer.build_inputs_with_special_tokens(ids)
            if (
                full_ids != [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
                or len(full_ids) > 512
            ):
                raise ValueError(
                    "Unexpected XLM-R BOS/EOS construction or window length"
                )
            inputs = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=device_type == "cuda",
            ):
                output = model(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs)
                ).last_hidden_state[0]
            hidden = output[1:-1].float()
            if hidden.shape != (len(ids), DIMENSION) or not bool(
                torch.isfinite(hidden).all()
            ):
                raise ValueError("Frozen encoder returned invalid content features")
            owner_tensor = torch.tensor(owners, dtype=torch.long, device=device)
            sums.index_add_(0, owner_tensor, hidden)
            counts.index_add_(0, owner_tensor, torch.ones(len(owners), device=device))
            n_subwords += len(ids)
            check_deadline(deadline_monotonic)
    if not bool((counts > 0).all()):
        raise ValueError(
            "At least one original word is missing from the encoder output"
        )
    tensor = (sums / counts[:, None]).cpu()
    if tensor.shape != (len(words), DIMENSION) or not bool(
        torch.isfinite(tensor).all()
    ):
        raise ValueError("Invalid pooled word tensor")
    check_deadline(deadline_monotonic)
    return tensor, len(chunks), n_subwords
