"""BERTScore-compatible constants/utilities.

We intentionally do NOT depend on the external `bert_score` package at runtime.
Instead, we vendor the *baseline constants* (for rescale-with-baseline) from
`bert-score==0.3.13` and replicate the key conventions needed for evaluation.

References:
- bert-score repo: https://github.com/Tiiiger/bert_score
- baseline files are under `bert_score/rescale_baseline/<lang>/<model_type>.tsv`

Notes:
- "num_layers" in bert-score corresponds to the row index in the baseline TSV.
  Row 0 is the embeddings output; row k is the representation after k encoder layers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# Subset of bert_score.utils.model2layers (bert-score==0.3.13).
# We keep this minimal: just what we need for DeBERTa (and MNLI variants).
MODEL2LAYERS: dict[str, int] = {
    "microsoft/deberta-base": 9,
    "microsoft/deberta-base-mnli": 9,
    "microsoft/deberta-large": 16,
    "microsoft/deberta-large-mnli": 18,
    "microsoft/deberta-xlarge": 18,
    "microsoft/deberta-xlarge-mnli": 40,
}


@dataclass(frozen=True)
class BaselineVals:
    p: float
    r: float
    f: float


_EMBEDDED_BASELINES: dict[tuple[str, str, int], BaselineVals] = {
    # Vendored from bert-score==0.3.13:
    # bert_score/rescale_baseline/en/microsoft/deberta-xlarge-mnli.tsv (row 40)
    ("en", "microsoft/deberta-xlarge-mnli", 40): BaselineVals(
        p=0.5169066,
        r=0.5170288,
        f=0.5150192,
    ),
}


def get_default_num_layers(model_type: str) -> Optional[int]:
    """Return bert-score's recommended num_layers for `model_type` (or None if unknown)."""
    if not model_type:
        return None
    return MODEL2LAYERS.get(str(model_type))


def _baseline_file_path(*, lang: str, model_type: str) -> str:
    base = os.path.join(os.path.dirname(__file__), "bertscore_rescale_baseline")
    # model_type may include "/" (HF repo name). We keep the same directory structure as bert-score.
    return os.path.join(base, str(lang), f"{model_type}.tsv")


def load_baseline_vals(*, lang: str, model_type: str, num_layers: int) -> BaselineVals:
    """Load (P,R,F) baseline constants for a given model/language/num_layers.

    These baselines are used by bert-score's `rescale_with_baseline` option:
        score_rescaled = (score - baseline) / (1 - baseline)
    """
    # First try embedded baselines (robust under Ray packaging / editable installs).
    key = (str(lang), str(model_type), int(num_layers))
    embedded = _EMBEDDED_BASELINES.get(key)
    if embedded is not None:
        return embedded

    path = _baseline_file_path(lang=lang, model_type=model_type)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"BERTScore baseline not found for {key}. "
            f"Tried embedded baselines and file '{path}'. "
            "If you changed reward_pretrain/model_type, vendor the corresponding bert-score TSV "
            "or add an embedded baseline."
        )

    want = int(num_layers)
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    # Expected header: LAYER,P,R,F
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < 4:
            continue
        try:
            layer = int(parts[0])
        except Exception:
            continue
        if layer != want:
            continue
        try:
            return BaselineVals(p=float(parts[1]), r=float(parts[2]), f=float(parts[3]))
        except Exception as e:
            raise ValueError(f"Failed parsing baseline row '{ln}' in '{path}': {e}") from e

    raise ValueError(f"No baseline row for num_layers={want} in '{path}'.")


