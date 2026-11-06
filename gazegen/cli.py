"""Run fixed GazeGen inference from exact words and an optional visible question."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .model import GazeGen
from .validation import validate_cache_key, validate_record

ROOT = Path(__file__).resolve().parents[1]
REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"


def read(path):
    """Read a UTF-8 JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    """Write strict JSON; NaN and infinity are rejected."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def key(words):
    return hashlib.sha256(
        json.dumps(
            words, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def batch(records, feature_cache, device="cpu"):
    """Collate exact word features for fixed-model inference.

    NR questions are always masked; answers and annotation fields are ignored.
    """
    if not records:
        raise ValueError("Cannot collate an empty record batch")
    for record in records:
        validate_record(record)
    max_words = max(map(lambda r: len(r["words"]), records))
    word_embeddings = torch.zeros(len(records), max_words, 768)
    word_mask = torch.zeros(len(records), max_words, dtype=torch.bool)
    question_embeddings = torch.zeros(len(records), 768)
    task_ids = []
    for i, r in enumerate(records):
        k = r.get("text_key") or key(r["words"])
        cached_words = feature_cache[k]
        if not isinstance(cached_words, torch.Tensor) or cached_words.shape != (
            len(r["words"]),
            768,
        ):
            raise ValueError("Cached text features must have shape [word_count, 768]")
        word_embeddings[i, : len(cached_words)] = cached_words
        word_mask[i, : len(cached_words)] = True
        task_ids.append(1 if r["task"] == "NR" else 2)
        if r["task"] == "IS":
            if not r.get("question", "").strip():
                raise ValueError(
                    "Information seeking requires a question visible before reading."
                )
            question_embeddings[i] = feature_cache[
                r.get("question_key") or key(r["question"].split())
            ].mean(0)
    inputs = dict(
        word_embeddings=word_embeddings,
        word_mask=word_mask,
        question_embeddings=question_embeddings,
        task_ids=torch.tensor(task_ids),
        question_mask=torch.tensor(task_ids) == 2,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def cache_load(folder, needed=None):
    """Load tensor-only cache files using safe checkpoint deserialization."""
    if needed is not None:
        for cache_key in needed:
            validate_cache_key(cache_key)
    fs = (
        list(Path(folder).glob("*.pt"))
        if needed is None
        else [Path(folder) / (k + ".pt") for k in sorted(needed)]
    )
    return {f.stem: torch.load(f, map_location="cpu", weights_only=True) for f in fs}


def needed(records):
    """Return the exact text and visible-question keys required by records."""
    for record in records:
        validate_record(record)
    return {r.get("text_key") or key(r["words"]) for r in records} | {
        r.get("question_key") or key(r["question"].split())
        for r in records
        if r["task"] == "IS"
    }


def load_model(device="cpu", checkpoint=None):
    """Load the archived state dictionary without downloading an encoder."""
    model = GazeGen().to(device)
    model.load_state_dict(
        torch.load(
            checkpoint or ROOT / "weights/gazegen.pt",
            map_location=device,
            weights_only=True,
        )
    )
    return model.eval()


def sample(model, records, cache, seeds, device="cpu", max_events=2048):
    with torch.inference_mode():
        return model.sample_batch(
            batch(records, cache, device), seeds, max_events=max_events
        )


def encode_missing(words_by_key, cache_dir, encoder, device="cpu"):
    """Encode missing exact-word caches only after explicit encoder selection."""
    for cache_key, words in words_by_key.items():
        validate_cache_key(cache_key)
        if key(words) != cache_key:
            raise ValueError("Cache key does not match the supplied words")
    from transformers import AutoModel, AutoTokenizer

    from .encoding import encode_words

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = {
        k: w for k, w in words_by_key.items() if not (cache_dir / (k + ".pt")).exists()
    }
    if not todo:
        return
    # A remote encoder is downloaded only through this explicit infer/serve option.
    opts = (
        {"local_files_only": True} if Path(encoder).exists() else {"revision": REVISION}
    )
    tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True, **opts)
    encoder_model = (
        AutoModel.from_pretrained(encoder, **opts)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    for k, w in todo.items():
        word_features, _, _ = encode_words(
            encoder_model, tokenizer, w, float("inf"), device=device
        )
        torch.save(word_features, cache_dir / (k + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    infer = sub.add_parser(
        "infer", help="Sample trajectories from the supplied checkpoint"
    )
    infer.add_argument("--inputs", default=str(ROOT / "examples/inputs.json"))
    infer.add_argument("--cache", default=str(ROOT / "examples/cache"))
    infer.add_argument("--output", required=True)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--samples", type=int, default=3)
    infer.add_argument("--index", type=int, default=0)
    infer.add_argument("--device", default="cpu")
    infer.add_argument(
        "--encoder", help="Local XLM-R directory or FacebookAI/xlm-roberta-base"
    )
    infer.add_argument("--max-events", type=int, default=2048)
    args = parser.parse_args()
    if (
        args.samples < 1
        or not 0 <= args.seed < 2**63
        or args.seed + args.samples > 2**63
    ):
        parser.error("samples must be positive and all seeds must be in [0, 2**63)")
    if not 1 <= args.max_events <= 2048:
        parser.error("max-events must be in 1..2048")
    input_data = read(args.inputs)
    if isinstance(input_data, list) and not 0 <= args.index < len(input_data):
        parser.error("index is outside the supplied input list")
    record = input_data[args.index] if isinstance(input_data, list) else input_data
    validate_record(record)
    words_by_key = {record.get("text_key") or key(record["words"]): record["words"]}
    if record["task"] == "IS":
        words_by_key[record.get("question_key") or key(record["question"].split())] = (
            record["question"].split()
        )
    torch.set_num_threads(2)
    if args.encoder:
        encode_missing(words_by_key, args.cache, args.encoder, args.device)
    missing = [
        k for k in words_by_key if not (Path(args.cache) / (k + ".pt")).is_file()
    ]
    if missing:
        parser.error(
            "This input is not cached. Supply --encoder /path/to/xlm-roberta-base to encode new text or questions."
        )
    features = cache_load(args.cache, needed([record]))
    model = load_model(args.device)
    paths = sample(
        model,
        [record] * args.samples,
        features,
        list(range(args.seed, args.seed + args.samples)),
        args.device,
        args.max_events,
    )
    save(args.output, {"input": record, "model": "GazeGen", "paths": paths})


if __name__ == "__main__":
    main()
