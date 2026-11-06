"""GazeGen: task-conditioned autoregressive fixation event generator."""

import math

import torch
from torch import nn

from .core import GazeGenCore as Core


class GazeGen(Core):
    """Condition the shared autoregressive backbone on NR or a visible IS question.

    A model instance stores question scores during a call; serialize concurrent
    calls on that instance, as the bundled local server does.
    """

    def __init__(self):
        super().__init__(mask_conditions=False)
        self.background_embedding.requires_grad_(False)
        self.question_word = nn.Linear(256, 64, bias=False)
        self.question_query = nn.Linear(768, 64, bias=False)
        self.question_gain = nn.Parameter(torch.zeros(()))

    def _safe(self, b):
        allowed = {
            "word_embeddings",
            "word_mask",
            "task_ids",
            "question_embeddings",
            "question_mask",
        }
        if set(b) - allowed:
            raise ValueError("Unexpected input fields")
        required = {
            "word_embeddings",
            "word_mask",
            "task_ids",
            "question_embeddings",
            "question_mask",
        }
        if not required.issubset(b):
            raise ValueError("Missing required model inputs")
        b = dict(b)
        t = b["task_ids"]
        if not bool(((t == 1) | (t == 2)).all()):
            raise ValueError("Only NR and IS task IDs are supported")
        if not torch.equal(b["question_mask"], t == 2):
            raise ValueError("question_mask must be true exactly for IS")
        b["question_embeddings"] = torch.where(
            (t == 2)[:, None],
            b["question_embeddings"],
            torch.zeros_like(b["question_embeddings"]),
        )
        if bool(((t == 2) & (b["question_embeddings"].abs().sum(-1) == 0)).any()):
            raise ValueError("Information seeking requires a visible question")
        b["background_ids"] = torch.full_like(t, -1)
        return b

    def _encode(self, b):
        words, c, h = super()._encode(b)
        scores = (
            self.question_word(words)
            * self.question_query(b["question_embeddings"].detach())[:, None]
        ).sum(-1) / math.sqrt(64)
        mask = b["word_mask"]
        scores = scores - (scores * mask).sum(-1, keepdim=True) / mask.sum(
            -1, keepdim=True
        )
        self._question_scores = scores * b["question_mask"][:, None]
        return words, c, h

    def _pointer_logits(self, states, words, conditions, history, word_mask):
        logits = super()._pointer_logits(states, words, conditions, history, word_mask)
        bias = self.question_gain * self._question_scores[:, None, :]
        return torch.cat((logits[:, :, :-1] + bias, logits[:, :, -1:]), -1)

    def sample_batch(self, b, seeds, **kwargs):
        if set(b) & {
            "fixation_words",
            "fixation_durations",
            "fixation_lengths",
        }:
            raise ValueError("No gaze prefix at inference")
        return super().sample_batch(self._safe(b), seeds, **kwargs)
