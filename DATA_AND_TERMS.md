# Data and release scope

This artifact contains the GazeGen checkpoint and inference source, together with selected OneStop examples needed for the demo. It does not include the full training corpus or downstream experiment archives.

## OneStop examples

Source and acquisition instructions: https://lacclab.github.io/OneStop-Eye-Movements/

Reference: Berzak et al. (2025), *OneStop: A 360-Participant English Eye Tracking Dataset with Different Reading Regimes*, Scientific Data, 12, 1995.

The downloaded dataset license record identifies CC BY 4.0 and is preserved in `demo/OneStop_license.json`. The bundled examples preserve the original stimulus words and selected, de-identified fixation records. The browser reflows the text and plots word indices; it does not reconstruct the original screen layout. Generated trajectories and human observations are identified separately. `demo/selection_manifest.json` documents the post-hoc selection of illustrative examples. Dataset and underlying third-party text rights continue to apply.

## Frozen language encoder

XLM-R-base: https://huggingface.co/FacebookAI/xlm-roberta-base

Reference: Conneau et al. (2020), *Unsupervised Cross-lingual Representation Learning at Scale*, Proceedings of ACL.

The encoder weights are not bundled. Its pinned revision and file hashes are recorded in `weights/encoder.json`. Small cached word representations are provided for the saved examples. The encoder and other dependencies retain their original terms.

## GazeGen release scope

This public inference prototype provides source and checkpoint files for peer-review inspection and research evaluation. This package does not grant a new blanket redistribution license for third-party materials. Public release and licensing of the full training and evaluation codebase are planned upon publication, after further development and documentation.

The generator's trained parameters and sampling behavior are preserved. This release contains only the inference path, not the training loss, optimizer, training loop, or downstream experiments. Its README explains how to generate new trajectories and use the prototype. Full scientific evaluation remains documented in the accompanying manuscript materials.
