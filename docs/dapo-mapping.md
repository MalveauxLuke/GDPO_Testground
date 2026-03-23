# Algorithm Mapping

This repo keeps GRPO, DAPO, and GDPO inside one harness so the algorithm deltas are visible in code.

## Variant Comparison

| Surface | GRPO | DAPO | GDPO | Local file |
| --- | --- | --- | --- | --- |
| Grouped outcome advantage | Yes | Yes | No, uses decoupled multi-reward group normalization | `algorithms/variants/*.py` |
| Symmetric PPO clipping | Yes | Optional baseline behavior | Yes by default | `algorithms/losses.py` |
| Asymmetric clip-higher | No | Yes | No in v1 | `algorithms/losses.py` |
| Dual-clip lower bound | Optional but not emphasized | Supported | Supported but symmetric by default | `algorithms/losses.py` |
| Dynamic sampling / prompt-group filtering | No | Yes | No in v1 | `algorithms/filtering.py` |
| Overlong shaping | No | Yes | No in v1 | `algorithms/overlong.py` |
| Raw multi-reward preservation | Optional | Optional | Required | `rewards/composition.py` |
| Shared trainer loop | Yes | Yes | Yes | `trainer/loop.py` |

## Where The Logic Sits

- GRPO
  - `src/dapo_lab/algorithms/variants/grpo.py`
- DAPO
  - `src/dapo_lab/algorithms/variants/dapo.py`
- GDPO
  - `src/dapo_lab/algorithms/variants/gdpo.py`

## Why GRPO Is Included

GRPO is included as the clean comparison point. In practice this means:

- same config model
- same reward system
- same trainer loop
- same diagnostics surface
- fewer active features

## Why GDPO Is Included

GDPO is included as the multi-reward comparison point. In this harness it differs from GRPO mainly by:

- preserving raw reward components
- normalizing each reward component separately within prompt groups
- recombining those component advantages with weights
- batch-whitening the combined scalar before token broadcast

That makes it easy to answer “what changed between GRPO, DAPO, and GDPO?” without diffing separate frameworks.
