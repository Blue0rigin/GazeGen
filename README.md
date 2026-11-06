# GazeGen

Task-conditioned gaze generation for studying text-based knowledge work.

This Stage 1 review artifact accompanies **Gaze Generation for Studying Text-Based Knowledge Work**. GazeGen takes a word sequence and a reading purpose and generates fixation locations, their order, and durations. One trained model supports:

- **Natural reading (NR):** reading for general comprehension, without a previewed question.
- **Information seeking (IS):** reading with a question supplied before the text.

## Scope of this release

The package contains the **trained GazeGen weights, the code needed for inference, and an interactive demo**, with the small set of saved examples and encoder features needed to use them. It uses the original OneStop-only checkpoint described in the proposal. The weights and generation algorithm are unchanged.

The full training pipeline, training/validation datasets, sampling schedules, downstream prediction models, evaluation scripts, and experimental result archives are **not included**. The accompanying proposal reports the preliminary findings; the technical supplement explains the model architecture and training objective. We plan to further develop and document the training pipeline and release the full codebase publicly upon publication. This release supports inspecting the prototype and running inference, rather than reproducing the complete training and evaluation pipeline.

## 1. Explore the demo without installation

Open the **[interactive demo](https://blue0rigin.github.io/GazeGen/)**. It runs in your browser without installation. The [GitHub repository](https://github.com/Blue0rigin/GazeGen) contains the weights and inference code.

For offline use, open **`demo/index.html`** in a browser. Saved examples work without Python or a model download.

1. Choose a passage and a reading purpose. For IS, choose one of its recorded questions.
2. Compare a generated trajectory with a recorded human trajectory.
3. Use **Play**, **Next fixation**, or the timeline to see fixation order and duration.
4. Select the paired Advanced/Elementary view to compare harder and easier versions of related text.

The gallery contains 12 paired passages and 96 text/task/question conditions. Its examples and generated draws were selected for closer agreement with human trajectories. They illustrate successful predictions and **do not represent average model performance**. The selection procedure and chosen records are documented in `demo/selection_manifest.json`. All matched human trajectories for these conditions are retained.

Custom questions require the optional local inference service below. The static demo does not generate trajectories for unseen questions.

## 2. Run inference on the saved examples

From this package's root, create a Python environment. Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first two inputs use the same passage: index 0 is natural reading; index 1 is information seeking with a recorded question.

```bash
python -m gazegen.cli infer --index 0 --samples 3 --seed 42 --output runs/natural_reading.json
python -m gazegen.cli infer --index 1 --samples 3 --seed 42 --output runs/information_seeking.json
```

These commands use the bundled checkpoint and cached word representations. They do not download an encoder or use a GPU. `--device cuda` is optional when a compatible GPU is available. Numerical results need not be identical across hardware or PyTorch versions.

## 3. Generate gaze for new text or a new question

Provide a JSON object with the exact displayed `words`, `task` (`NR` or `IS`), and a `question` for IS. Word indices in the output are zero-based. For NR, omit the question or use an empty string; a question supplied with NR is ignored.

```json
{
  "words": ["Employees", "must", "retain", "receipts", "for", "travel", "expenses."],
  "task": "IS",
  "question": "What must employees retain?"
}
```

A copy is provided as `examples/new_input.json`. New inputs require the frozen **XLM-R-base** encoder, whose large weights are not included. Use a local copy:

```bash
python -m gazegen.cli infer --inputs examples/new_input.json --cache cache \
  --encoder /path/to/xlm-roberta-base --samples 3 --output runs/new_text.json
```

Alternatively, explicitly supply `--encoder FacebookAI/xlm-roberta-base` to obtain the pinned encoder revision from Hugging Face. This requires internet access on first use. The revision is `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`; its identity is recorded in `weights/encoder.json`. Cached representations must come from this encoder and the exact same word sequence.

Inputs do not require a correct answer, evidence annotations, text-version labels, human gaze, reader identity, or workload ratings. An English word sequence and reading purpose are sufficient for NR; IS additionally requires the question.

### Output

The output JSON contains the input and a list of sampled paths:

- `words`: ordered zero-based indices into the supplied word sequence.
- `durations_ms`: corresponding fixation durations in milliseconds.
- `seed` and `length`: the sampling seed and number of events.
- `terminated_by_stop`, `numerical_valid`, `censored`, and `primary_eligible`: completion and validity information.
- `failure_reason`: why a path is incomplete or unsuitable, when applicable.

Sampling can skip or revisit words. It stops when the model selects STOP, with a safety limit of 2,048 events. Incomplete samples are flagged, not silently replaced. CLI output retains native durations; the demo omits events below 50 ms for its display convention.

## 4. Enable custom questions in the browser

For saved examples at a local web address:

```bash
python -m gazegen.serve --port 8769
```

For live inference on a custom question:

```bash
python -m gazegen.serve --port 8769 --encoder /path/to/xlm-roberta-base \
  --cache cache --device cpu
```

Open **http://127.0.0.1:8769/**. The service accepts local connections only. Under IS, enter a question and request a new trajectory. It runs the same fixed GazeGen model; there is no fitting or adaptation. A custom question has **no matched human observation** in the demo. The CLI can also generate NR trajectories for entirely new texts.

## Package contents

| Path | Purpose |
| --- | --- |
| `weights/gazegen.pt` | Selected GazeGen state dictionary |
| `weights/model.json` | Checkpoint identity and supported inputs/outputs |
| `weights/encoder.json` | Frozen encoder identity and revision |
| `gazegen/` | Model definition, encoding, sampling, CLI, and local demo server |
| `examples/inputs.json`, `examples/cache/` | Saved demo inputs and exact-word features for offline inference |
| `examples/new_input.json` | Input-format example for new text |
| `demo/` | Interactive gallery, selected generated/human paths, and data attribution |
| `DATA_AND_TERMS.md` | Data sources and release scope |
| `MANIFEST.sha256` | File integrity checksums |

The checkpoint SHA-256 is `69b8df45be83044390e7abd7a942b1b56598d4cc0e56c9bef2e1af4bd0821cd6`. The filename is shortened for this release; its bytes are unchanged. From the package root, `sha256sum -c MANIFEST.sha256` verifies the distributed files.

## Interpretation and limitations

The checkpoint was trained on first readings in the OneStop English news corpus. It generates population-level predictions conditioned on text and task, not a particular reader's behavior. Repeated samples are model predictions, not additional human observations. Editorial Advanced/Elementary versions are not subjective workload ratings. Workplace transfer and applications to perceived task difficulty, mental effort, or cognitive workload require independent human validation.

The demo maps word-index events onto text reflowed by the browser; connecting lines do not reconstruct the original screen coordinates. Question-conditioned outputs can change with the question, but a visually plausible path alone does not demonstrate valid recovery of human question effects.

## Attribution

OneStop: Berzak et al. (2025), *OneStop: A 360-Participant English Eye Tracking Dataset with Different Reading Regimes*, Scientific Data, 12, 1995.

XLM-R: Conneau et al. (2020), *Unsupervised Cross-lingual Representation Learning at Scale*, Proceedings of ACL, 8440–8451.

See `DATA_AND_TERMS.md` for source links and applicable terms. This public inference prototype accompanies work in progress; it does not imply peer-reviewed publication or completed submission.
