"""GazeGen autoregressive decoding core. Training objectives are not distributed.

The parameter names and shapes match the supplied checkpoint. Sampling retains
the original position, duration, stopping, and seeded-randomness calculations.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

TASK_ONESTOP_ORDINARY = 1
TASK_ONESTOP_SEARCH = 2
MAX_EVENTS = 2048
RELATIVE_CLIP = 32
RELATIVE_BUCKETS = 2 * RELATIVE_CLIP + 1
BOS_EXACT_POSITIONS = 32
BOS_ABSOLUTE_BUCKETS = 48
PROGRESS_NAMES = (
    "previous_position",
    "furthest_visited_position",
    "word_coverage",
    "events_per_word",
    "is_bos",
)


class HistoryFeatures(NamedTuple):
    """Prefix-only tensors with a shared [B,T] leading shape."""

    pointer_features: Tensor  # [B,T,N,3], original offset / visited / frequency
    relative_buckets: Tensor  # [B,T,N], int64
    absolute_buckets: Tensor  # [B,T,N], int64
    is_bos: Tensor  # [B,T], bool; not equivalent to previous_id == 0
    progress: Tensor  # [B,T,5], float32


def relative_displacement_buckets(displacement: Tensor) -> Tensor:
    return (
        displacement.clamp(min=-RELATIVE_CLIP, max=RELATIVE_CLIP).long() + RELATIVE_CLIP
    )


def bos_absolute_buckets(position: Tensor) -> Tensor:
    # Exact small integers; tail buckets are per candidate, not frequency priors.
    tail = (
        BOS_EXACT_POSITIONS
        + torch.floor(
            torch.log2(position.clamp(min=BOS_EXACT_POSITIONS).float())
        ).long()
        - 5
    )
    return torch.where(
        position < BOS_EXACT_POSITIONS,
        position.long(),
        tail.clamp(max=BOS_ABSOLUTE_BUCKETS - 1),
    )


class GazeGenCore(nn.Module):
    """Encode text and task conditions, then sample fixation events.

    Word and question features come from frozen XLM-R. The reserved background
    embedding is retained to load the checkpoint and is always masked for NR/IS.
    """

    def __init__(
        self,
        mask_conditions: bool = False,
        num_background_categories: int = 8,
    ) -> None:
        super().__init__()
        if num_background_categories < 1:
            raise ValueError("Background count must be positive")
        self.mask_conditions = bool(mask_conditions)
        self.num_background_categories = int(num_background_categories)
        self.word_projection = nn.Linear(768, 256)
        self.task_embedding = nn.Embedding(3, 16)
        self.background_embedding = nn.Embedding(num_background_categories, 16)
        self.question_projection = nn.Linear(768, 32)
        self.initial_state = nn.Linear(256 + 64, 256)
        self.bos = nn.Parameter(torch.empty(256))
        nn.init.normal_(self.bos, mean=0.0, std=0.02)
        self.gru = nn.GRU(
            256 + 1 + 64, 256, num_layers=1, batch_first=True, dropout=0.0
        )
        self.pointer_hidden = nn.Linear(256 + 256 + 64 + 3, 128)
        self.pointer_out = nn.Linear(128, 1)
        self.stop_head = nn.Sequential(
            nn.Linear(256 + 64, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.duration_head = nn.Sequential(
            nn.Linear(256 + 256 + 64, 128), nn.Tanh(), nn.Linear(128, 2)
        )
        # Retain the checkpoint state keys and shapes.
        self.relative_displacement_bias = nn.Parameter(torch.zeros(RELATIVE_BUCKETS))
        self.bos_absolute_bias = nn.Parameter(torch.zeros(BOS_ABSOLUTE_BUCKETS))
        self.stop_progress = nn.Linear(len(PROGRESS_NAMES), 1, bias=False)
        nn.init.zeros_(self.stop_progress.weight)

    def _validate_inputs(self, batch: Mapping[str, Tensor]) -> tuple[int, int]:
        required = (
            "word_embeddings",
            "word_mask",
            "task_ids",
            "background_ids",
            "question_embeddings",
            "question_mask",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise ValueError(f"Missing model inputs: {missing}")
        words, mask = batch["word_embeddings"], batch["word_mask"]
        if words.ndim != 3 or words.shape[-1] != 768:
            raise ValueError("word_embeddings must have shape [B,N,768]")
        b, n, _ = words.shape
        if b == 0 or n == 0 or mask.shape != (b, n) or mask.dtype != torch.bool:
            raise ValueError("word_mask must be nonempty bool [B,N]")
        lengths = mask.sum(1)
        expected_mask = torch.arange(n, device=mask.device)[None, :] < lengths[:, None]
        if bool((lengths == 0).any()) or not torch.equal(mask, expected_mask):
            raise ValueError("word_mask must describe nonempty contiguous prefixes")
        for name in ("task_ids", "background_ids"):
            if batch[name].shape != (b,) or batch[name].dtype != torch.long:
                raise ValueError(f"{name} must be int64 [B]")
        if batch["question_embeddings"].shape != (b, 768):
            raise ValueError("question_embeddings must be [B,768]")
        if (
            batch["question_mask"].shape != (b,)
            or batch["question_mask"].dtype != torch.bool
        ):
            raise ValueError("question_mask must be bool [B]")
        if any(batch[name].device != words.device for name in required):
            raise ValueError("All model inputs must share a device")
        if words.device != self.bos.device:
            raise ValueError("Inputs and model must share a device")
        tasks, backgrounds, qmask = (
            batch["task_ids"],
            batch["background_ids"],
            batch["question_mask"],
        )
        if not bool(
            ((tasks == TASK_ONESTOP_ORDINARY) | (tasks == TASK_ONESTOP_SEARCH)).all()
        ):
            raise ValueError("Only NR and IS are supported")
        if not torch.equal(qmask, tasks == TASK_ONESTOP_SEARCH):
            raise ValueError("Only information-seeking inputs have a question")
        if bool((backgrounds != -1).any()):
            raise ValueError("Reader background is not an input to this checkpoint")
        if not bool(torch.isfinite(words).all()) or not bool(
            torch.isfinite(batch["question_embeddings"]).all()
        ):
            raise ValueError("Frozen encoder features must be finite")
        if bool((batch["question_embeddings"][~qmask] != 0).any()):
            raise ValueError("Unavailable questions must have zero features")
        return b, n

    def _encode(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Recompute every trainable projection for this model and forward call."""
        words = self.word_projection(batch["word_embeddings"].detach())
        tasks = self.task_embedding(batch["task_ids"])
        ids = batch["background_ids"]
        backgrounds = self.background_embedding(ids.clamp(min=0))
        backgrounds = backgrounds * (ids >= 0).unsqueeze(-1)
        questions = self.question_projection(batch["question_embeddings"].detach())
        questions = questions * batch["question_mask"].unsqueeze(-1)
        conditions = torch.cat((tasks, backgrounds, questions), dim=-1)
        if self.mask_conditions:
            # Retain the same parameter graph/shapes, but remove all conditions.
            conditions = conditions * 0.0
        mask = batch["word_mask"].unsqueeze(-1)
        word_mean = (words * mask).sum(1) / mask.sum(1)
        h0 = torch.tanh(self.initial_state(torch.cat((word_mean, conditions), -1)))
        return words, conditions, h0

    def _run_gru(self, inputs: Tensor, h0: Tensor) -> Tensor:
        # Keep recurrent state arithmetic in float32 under an outer bf16 context.
        # Projections/pointer heads can still use bf16; probability math cannot.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            states, _ = self.gru(inputs.float(), h0.float().unsqueeze(0))
        return states

    @staticmethod
    def _history_features(
        counts: Tensor, previous_ids: Tensor, word_mask: Tensor
    ) -> HistoryFeatures:
        """Use only the already generated fixation prefix.

        counts[B,T,N] contains prior fixation counts, previous_ids[B,T] contains
        the prior word (placeholder 0 at BOS), and word_mask[B,N] excludes padding.
        The empty-prefix flag is derived from counts, so a return to word 0 can
        never reactivate the BOS bias. No future target or total path length is
        consumed. Generation passes T=1.
        """
        if (
            counts.ndim != 3
            or previous_ids.shape != counts.shape[:2]
            or word_mask.shape != (counts.shape[0], counts.shape[2])
        ):
            raise ValueError(
                "History shapes must be counts[B,T,N], previous_ids[B,T], mask[B,N]"
            )
        b, t, n = counts.shape
        counts = counts.float() * word_mask[:, None, :]
        word_lengths = word_mask.sum(1).float()
        event_counts = counts.sum(-1)
        is_bos = event_counts == 0
        positions = torch.arange(n, device=counts.device, dtype=torch.long)[
            None, None, :
        ]
        displacement = positions - previous_ids[:, :, None]
        denominator = (word_lengths - 1).clamp(min=1)[:, None, None]
        offsets = displacement.float() / denominator
        offsets = torch.where(is_bos[:, :, None], torch.zeros_like(offsets), offsets)
        visited = counts > 0
        original = torch.stack(
            (offsets, visited.float(), counts / event_counts.clamp(min=1)[:, :, None]),
            -1,
        )
        furthest = torch.where(visited, positions, -1).amax(-1).clamp(min=0)
        previous_position = previous_ids.float() / denominator.squeeze(-1)
        previous_position = torch.where(
            is_bos, torch.zeros_like(previous_position), previous_position
        )
        furthest_position = furthest.float() / denominator.squeeze(-1)
        coverage = visited.sum(-1).float() / word_lengths[:, None]
        events_per_word = event_counts / word_lengths[:, None]
        progress = torch.stack(
            (
                previous_position,
                furthest_position,
                coverage,
                events_per_word,
                is_bos.float(),
            ),
            -1,
        )
        return HistoryFeatures(
            original,
            relative_displacement_buckets(displacement),
            bos_absolute_buckets(positions).expand(b, t, n),
            is_bos,
            progress,
        )

    def _pointer_logits(
        self,
        states: Tensor,
        words: Tensor,
        conditions: Tensor,
        history: HistoryFeatures,
        word_mask: Tensor,
    ) -> Tensor:
        """Vectorized [B,T,N+1] scores; affine factoring is exactly concatenation."""
        weight = self.pointer_hidden.weight
        state_part = F.linear(states, weight[:, :256], self.pointer_hidden.bias)
        word_part = F.linear(words, weight[:, 256:512])
        condition_part = F.linear(conditions, weight[:, 512:576])
        history_part = F.linear(history.pointer_features, weight[:, 576:])
        hidden = torch.tanh(
            state_part[:, :, None, :]
            + word_part[:, None, :, :]
            + condition_part[:, None, None, :]
            + history_part
        )
        word_scores = self.pointer_out(hidden).squeeze(-1).float()
        relative_bias = self.relative_displacement_bias[
            history.relative_buckets
        ].float()
        absolute_bias = self.bos_absolute_bias[history.absolute_buckets].float()
        word_scores = word_scores + torch.where(
            history.is_bos[:, :, None], absolute_bias, relative_bias
        )
        word_scores = word_scores.masked_fill(~word_mask[:, None, :], -torch.inf)
        c = conditions[:, None, :].expand(-1, states.shape[1], -1)
        stop_scores = self.stop_head(torch.cat((states, c), -1)).float()
        # Accumulate the explicit progress contribution in probability precision.
        with torch.autocast(device_type=states.device.type, enabled=False):
            stop_scores = (
                stop_scores + self.stop_progress(history.progress.float()).float()
            )
        return torch.cat((word_scores, stop_scores), dim=-1)

    def _duration_parameters(
        self, states: Tensor, selected_words: Tensor, conditions: Tensor
    ) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=states.device.type, enabled=False):
            outputs = self.duration_head(
                torch.cat(
                    (states.float(), selected_words.float(), conditions.float()), dim=-1
                )
            )
            return outputs[..., 0], F.softplus(outputs[..., 1]) + 0.001

    @torch.no_grad()
    def sample_batch(
        self,
        batch: Mapping[str, Tensor],
        seeds: Sequence[int],
        max_events: int = MAX_EVENTS,
        deadline_monotonic: float | None = None,
    ) -> list[dict[str, Any]]:
        """Batch-vectorized autoregression, one seeded attempt per row.

        The caller supplies one seed for each requested trajectory.
        Each CPU generator first draws max_events+1 action uniforms, then
        max_events normal duration innovations. Thus batching, condition changes,
        another row's stopping time and global RNG state cannot alter its stream.
        A deadline returns unfinished prefixes, never replacement attempts.
        """
        b, n = self._validate_inputs(batch)
        if len(seeds) != b or not 1 <= max_events <= MAX_EVENTS:
            raise ValueError(
                "Exactly one seed per row and max_events in 1..2048 are required"
            )
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
            for seed in seeds
        ):
            raise ValueError("Seeds must be nonnegative Python integers below 2**63")
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            return [
                {
                    "words": [],
                    "durations_ms": [],
                    "seed": int(seed),
                    "length": 0,
                    "numerical_valid": True,
                    "terminated_by_stop": False,
                    "censored": False,
                    "attempt_finished": False,
                    "primary_eligible": False,
                    "failure_reason": "deadline_exceeded",
                    "failure_event": None,
                }
                for seed in seeds
            ]
        uniforms, innovations = [], []
        for seed in seeds:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            uniforms.append(
                torch.rand(max_events + 1, generator=generator, dtype=torch.float32)
            )
            innovations.append(
                torch.randn(max_events, generator=generator, dtype=torch.float32)
            )
        action_noise = torch.stack(uniforms).to(self.bos.device)
        duration_noise = torch.stack(innovations).to(self.bos.device)
        words, conditions, state = self._encode(batch)
        device = words.device
        word_mask = batch["word_mask"]
        word_lengths = word_mask.sum(1)
        recorded_words = torch.full(
            (b, max_events), -1, dtype=torch.long, device=device
        )
        recorded_duration = torch.full(
            (b, max_events), torch.nan, dtype=torch.float32, device=device
        )
        lengths = torch.zeros(b, dtype=torch.long, device=device)
        active = torch.ones(b, dtype=torch.bool, device=device)
        valid = active.clone()
        terminated = torch.zeros_like(active)
        censored = torch.zeros_like(active)
        finished = torch.zeros_like(active)
        # 0=none; 1=invalid logits; 2=invalid duration; 3=censor; 4=deadline.
        reason = torch.zeros(b, dtype=torch.long, device=device)
        failure_word = torch.full((b,), -1, dtype=torch.long, device=device)
        failure_duration = torch.full(
            (b,), torch.nan, dtype=torch.float32, device=device
        )
        failure_log_duration = torch.full(
            (b,), torch.nan, dtype=torch.float32, device=device
        )
        visits = torch.zeros((b, n), dtype=torch.float32, device=device)
        previous_word = self.bos.expand(b, -1)
        previous_log_duration = torch.zeros(b, dtype=torch.float32, device=device)
        previous_id = torch.zeros(b, dtype=torch.long, device=device)

        for step in range(max_events + 1):
            if (
                deadline_monotonic is not None
                and time.monotonic() >= deadline_monotonic
            ):
                reason = torch.where(active, 4, reason)
                break
            if not bool(active.any()):
                break
            inputs = torch.cat(
                (previous_word, previous_log_duration[:, None], conditions), -1
            )
            next_state = self._run_gru(inputs[:, None, :], state)[:, 0]
            state = torch.where(active[:, None], next_state, state)
            history = self._history_features(
                visits[:, None], previous_id[:, None], word_mask
            )
            logits = self._pointer_logits(
                state[:, None], words, conditions, history, word_mask
            )[:, 0]
            legal = torch.cat(
                (word_mask, torch.ones((b, 1), dtype=torch.bool, device=device)), 1
            )
            bad_logits = active & ~torch.isfinite(logits.masked_fill(~legal, 0.0)).all(
                1
            )
            valid = valid & ~bad_logits
            finished = finished | bad_logits
            reason = torch.where(bad_logits, 1, reason)
            active = active & ~bad_logits
            safe_logits = torch.where(
                bad_logits[:, None], torch.zeros_like(logits), logits
            )
            # Inactive rows may contain stale nonfinite state; never let them
            # produce invalid CDF indexing even though their result is ignored.
            safe_logits = torch.where(
                active[:, None], safe_logits, torch.zeros_like(safe_logits)
            )
            safe_logits = safe_logits.masked_fill(~legal, -torch.inf)
            with torch.autocast(device_type=device.type, enabled=False):
                probabilities = F.softmax(safe_logits.float(), dim=1)
                cdf = probabilities.cumsum(1)
                cdf[:, -1] = 1.0
                actions = (cdf <= action_noise[:, step, None]).sum(1)
            stop = active & (actions == n)
            terminated = terminated | stop
            finished = finished | stop
            active = active & ~stop
            if step == max_events:
                censored = censored | active
                finished = finished | active
                reason = torch.where(active, 3, reason)
                active = active & False
                break
            illegal_word = active & ((actions >= word_lengths) | (actions < 0))
            valid = valid & ~illegal_word
            finished = finished | illegal_word
            reason = torch.where(illegal_word, 1, reason)
            failure_word = torch.where(illegal_word, actions, failure_word)
            active = active & ~illegal_word
            safe_action = actions.clamp(min=0, max=n - 1)
            selected = words[torch.arange(b, device=device), safe_action]
            mu, sigma = self._duration_parameters(state, selected, conditions)
            with torch.autocast(device_type=device.type, enabled=False):
                log_d = mu + sigma * duration_noise[:, step]
                duration = torch.exp(log_d)
            bad_duration = active & (
                (duration <= 0)
                | ~torch.isfinite(duration)
                | ~torch.isfinite(mu)
                | ~torch.isfinite(sigma)
            )
            failure_word = torch.where(bad_duration, actions, failure_word)
            failure_duration = torch.where(bad_duration, duration, failure_duration)
            failure_log_duration = torch.where(
                bad_duration, log_d, failure_log_duration
            )
            valid = valid & ~bad_duration
            finished = finished | bad_duration
            reason = torch.where(bad_duration, 2, reason)
            active = active & ~bad_duration
            recorded_words[:, step] = torch.where(
                active, actions, recorded_words[:, step]
            )
            recorded_duration[:, step] = torch.where(
                active, duration, recorded_duration[:, step]
            )
            lengths = lengths + active.long()
            visits.scatter_add_(1, safe_action[:, None], active.float()[:, None])
            previous_word = torch.where(active[:, None], selected, previous_word)
            previous_log_duration = torch.where(active, log_d, previous_log_duration)
            previous_id = torch.where(active, actions, previous_id)

        arrays = [
            value.detach().cpu()
            for value in (
                recorded_words,
                recorded_duration,
                lengths,
                valid,
                terminated,
                censored,
                finished,
                reason,
                failure_word,
                failure_duration,
                failure_log_duration,
            )
        ]
        names = {
            0: None,
            1: "invalid_action_scores_or_word",
            2: "invalid_duration",
            3: "right_censored",
            4: "deadline_exceeded",
        }
        results = []
        for row, seed in enumerate(seeds):
            length = int(arrays[2][row])
            is_finished, is_valid, is_stopped = (
                bool(arrays[6][row]),
                bool(arrays[3][row]),
                bool(arrays[4][row]),
            )
            code = int(arrays[7][row])
            failure_reason = names[code]
            if code == 0 and is_stopped and length < 2:
                failure_reason = "empty_path" if length == 0 else "short_path"
            record = {
                "words": arrays[0][row, :length].tolist(),
                "durations_ms": arrays[1][row, :length].tolist(),
                "seed": int(seed),
                "length": length,
                "numerical_valid": is_valid,
                "terminated_by_stop": is_stopped,
                "censored": bool(arrays[5][row]),
                "attempt_finished": is_finished,
                "primary_eligible": is_finished
                and is_valid
                and is_stopped
                and 2 <= length <= MAX_EVENTS,
                "failure_reason": failure_reason,
                "failure_event": None,
            }
            if int(arrays[8][row]) >= 0:
                raw_d, raw_log = float(arrays[9][row]), float(arrays[10][row])
                record["failure_event"] = {
                    "event_index": length,
                    "word_id": int(arrays[8][row]),
                    "duration_ms": raw_d if math.isfinite(raw_d) else None,
                    "log_duration_ms": raw_log if math.isfinite(raw_log) else None,
                    "duration_status": "finite"
                    if math.isfinite(raw_d)
                    else repr(raw_d),
                }
            results.append(record)
        return results
