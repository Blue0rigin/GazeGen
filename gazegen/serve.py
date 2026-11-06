"""Serve the local research demo and serialize optional fixed-model inference.

This loopback-only server is a development prototype, not a production service.
It does not train a model, expose arbitrary files, or enable cross-origin API use.
"""

import argparse
import json
import logging
import threading
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .cli import ROOT, cache_load, encode_missing, key, load_model, needed, sample
from .validation import validate_record

LOGGER = logging.getLogger(__name__)
INFERENCE_LOCK = threading.Lock()
MAX_REQUEST_BYTES = 50_000


def parse_generation_request(payload: object) -> tuple[dict, int]:
    """Validate a bounded custom-question request and construct model-only fields."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    words = payload.get("words")
    question = payload.get("question")
    if not isinstance(words, list) or not 1 <= len(words) <= 600:
        raise ValueError("words must contain between 1 and 600 tokens")
    if any(
        not isinstance(word, str) or not 1 <= len(word.strip()) <= 200 for word in words
    ):
        raise ValueError(
            "Every word must be a nonempty string of at most 200 characters"
        )
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 500:
        raise ValueError("question must contain between 1 and 500 characters")
    if payload.get("task", "IS") != "IS":
        raise ValueError("The custom-question endpoint supports IS only")
    seed = payload.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    question = question.strip()
    record = {
        "words": words,
        "task": "IS",
        "question": question,
        "text_key": key(words),
        "question_key": key(question.split()),
    }
    validate_record(record)
    return record, seed


def create_server(port: int, *, model=None, encoder=None, cache=None, device="cpu"):
    """Build a loopback server; port 0 is supported for isolated integration tests."""
    cache = cache or ROOT / "examples/cache"

    class Handler(SimpleHTTPRequestHandler):
        def send_json(self, value: dict, status=HTTPStatus.OK) -> None:
            body = json.dumps(value, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/api/status":
                self.send_json({"inference": model is not None, "model": "GazeGen"})
                return
            super().do_GET()

        def do_POST(self) -> None:
            if self.path != "/api/generate":
                self.send_json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
                return
            actual_port = self.server.server_port
            allowed_hosts = {f"127.0.0.1:{actual_port}", f"localhost:{actual_port}"}
            origin = self.headers.get("Origin")
            if self.headers.get("Host") not in allowed_hosts or (
                origin is not None
                and origin not in {f"http://{host}" for host in allowed_hosts}
            ):
                self.send_json(
                    {"error": "Only same-origin loopback requests are accepted"},
                    HTTPStatus.FORBIDDEN,
                )
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(
                    {"error": "Content-Type must be application/json"},
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                )
                return
            if self.headers.get("Transfer-Encoding") is not None:
                self.send_json(
                    {"error": "Chunked request bodies are not supported"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_json(
                    {"error": "Invalid Content-Length"}, HTTPStatus.BAD_REQUEST
                )
                return
            if not 0 < length <= MAX_REQUEST_BYTES:
                self.send_json(
                    {"error": "Request body must be 1..50000 bytes"},
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
                return
            if model is None:
                self.send_json(
                    {"error": "Inference requires --encoder"},
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            try:
                self.connection.settimeout(15)
                record, seed = parse_generation_request(
                    json.loads(self.rfile.read(length))
                )
            except (ValueError, UnicodeError, TimeoutError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            try:
                # The model caches per-call question scores, so shared inference
                # and cache creation must not interleave across HTTP threads.
                with INFERENCE_LOCK:
                    encode_missing(
                        {
                            record["text_key"]: record["words"],
                            record["question_key"]: record["question"].split(),
                        },
                        cache,
                        encoder,
                        device,
                    )
                    features = cache_load(cache, needed([record]))
                    raw = sample(model, [record], features, [seed], device)[0]
                if not raw["primary_eligible"]:
                    self.send_json(
                        {
                            "error": "The sampled path did not finish as an eligible trajectory",
                            "failure_reason": raw["failure_reason"],
                        },
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                    )
                    return
                events = [
                    {"word_id": word, "duration_ms": duration}
                    for word, duration in zip(raw["words"], raw["durations_ms"])
                    if duration >= 50
                ]
                self.send_json(
                    {
                        "paths": [
                            {
                                "events": events,
                                "seed": str(seed),
                                "terminated": raw["terminated_by_stop"],
                            }
                        ],
                        "native_record": raw,
                    }
                )
            except Exception:
                # Keep local diagnostics, but do not return stack traces or paths
                # to a caller. Invalid requests have already been handled above.
                LOGGER.exception("Fixed-model inference failed")
                self.send_json(
                    {"error": "Inference failed; check the local server log"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return ThreadingHTTPServer(
        ("127.0.0.1", port), partial(Handler, directory=str(ROOT / "demo"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--encoder")
    parser.add_argument("--cache", default=str(ROOT / "examples/cache"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    model = None
    if args.encoder:
        import torch

        torch.set_num_threads(2)
        model = load_model(args.device)
    with create_server(
        args.port,
        model=model,
        encoder=args.encoder,
        cache=args.cache,
        device=args.device,
    ) as server:
        print(
            f"GazeGen demo: http://127.0.0.1:{server.server_port} · inference={model is not None}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
