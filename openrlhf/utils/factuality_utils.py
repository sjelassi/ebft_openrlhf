from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")


def extract_source_from_prompt(prompt: str) -> str:
    """Best-effort extraction of source text from a summarization prompt.

    For XSum in this repo, prompts have the format:
        "Text: <document content>\nSummary:"

    This function strips the "Text:" prefix and "Summary:" suffix to extract
    just the document content for NLI premise scoring.
    """
    if prompt is None:
        return ""
    s = str(prompt).replace("\r", "")
    s = s.strip()
    if not s:
        return ""

    # ------------------------------------------------------------------
    # Robust extraction for common prompt templates.
    #
    # Many datasets (e.g., XSum) use:
    #   Text: <doc>\nSummary:
    #
    # Under chat templates / instruction wrappers, the "Text:" line may not be
    # at the beginning of the string. Prefer extracting the span between:
    #   (Text|Document|Article):  ...  (Summary|TL;DR):
    # using line-anchored markers when possible.
    # ------------------------------------------------------------------
    start_re = re.compile(r"(?im)^(text|document|article)\s*:\s*")
    end_re = re.compile(r"(?im)^(summary|tl;dr|tldr)\s*:\s*")

    starts = list(start_re.finditer(s))
    if starts:
        # Prefer the first span that has a clear end marker after it.
        for m in starts:
            body = s[m.end() :]
            m_end = end_re.search(body)
            if m_end:
                extracted = body[: m_end.start()].strip()
                if extracted:
                    return extracted

        # No end marker found; fall back to the last start marker.
        body = s[starts[-1].end() :].strip()
        if body:
            return body

    # ------------------------------------------------------------------
    # Backward-compatible heuristics (prefix/suffix at whole-string edges).
    # ------------------------------------------------------------------
    for prefix in ("Text:", "Text :", "Document:", "Article:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix) :].strip()
            break
    for suffix in ("Summary:", "Summary :", "TL;DR:", "TLDR:"):
        if s.lower().endswith(suffix.lower()):
            s = s[: -len(suffix)].strip()
            break
    return s


def split_sentences(text: str, max_sentences: Optional[int] = None) -> List[str]:
    """Simple sentence splitter suitable for fast factuality evaluation.

    This intentionally avoids heavy dependencies (e.g., spaCy, nltk).
    """
    if text is None:
        return []
    s = str(text).replace("\r", "").strip()
    if not s:
        return []

    # Normalize whitespace/newlines.
    s = re.sub(r"\s+", " ", s).strip()

    parts = [p.strip() for p in _SENT_SPLIT_RE.split(s) if p.strip()]
    if not parts:
        return []

    if max_sentences is not None:
        try:
            ms = int(max_sentences)
        except Exception:
            ms = None
        if ms is not None and ms > 0:
            parts = parts[:ms]

    return parts


def infer_entailment_label_id(model) -> Optional[int]:
    """Infer the entailment label index for an NLI classifier."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None

    # Prefer label2id if present.
    label2id = getattr(cfg, "label2id", None) or {}
    for k, v in label2id.items():
        if "entail" in str(k).lower():
            try:
                return int(v)
            except Exception:
                continue

    # Fallback to id2label.
    id2label = getattr(cfg, "id2label", None) or {}
    for k, v in id2label.items():
        if "entail" in str(v).lower():
            try:
                return int(k)
            except Exception:
                continue

    return None


@torch.no_grad()
def _infer_entailment_label_id_by_sanity_check(
    *,
    tokenizer,
    model,
    device: torch.device,
    max_length: int = 512,
    truncation: str = "only_first",
) -> Optional[int]:
    """Infer entailment label index by probing the model on trivial NLI pairs.

    Some custom models (notably `liuyanyi/AlignScore-large-hf`) ship without `id2label/label2id`,
    but expose 3-way NLI logits (e.g., `tri_label_logits`). In that case, assuming the label order
    can silently break scoring.

    This helper runs a tiny sanity check and returns the index that behaves like entailment:
    identical premise/hypothesis should strongly predict entailment.
    """
    if tokenizer is None or model is None or device is None:
        return None

    # Only attempt this for small max_length; keep it lightweight.
    try:
        max_length = int(max_length)
    except Exception:
        max_length = 512
    max_length = max(8, max_length)

    # Construct very simple pairs.
    premise = "A cat sits on a mat."
    hyp_entail = "A cat sits on a mat."
    hyp_contra = "No cat sits on a mat."

    def _tri_probs(prem: str, hyp: str) -> Optional[torch.Tensor]:
        toks = tokenizer(
            prem,
            hyp,
            padding=True,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = model(**toks)
        tri = getattr(out, "tri_label_logits", None)
        if tri is None and isinstance(out, dict):
            tri = out.get("tri_label_logits", None)
        if tri is None:
            return None
        if not isinstance(tri, torch.Tensor):
            try:
                tri = torch.as_tensor(tri)
            except Exception:
                return None
        if tri.dim() == 1:
            tri = tri.unsqueeze(0)
        if tri.size(-1) != 3:
            return None
        return torch.softmax(tri.float(), dim=-1).squeeze(0)

    try:
        p_ent = _tri_probs(premise, hyp_entail)
        p_con = _tri_probs(premise, hyp_contra)
        if p_ent is None:
            return None
        ent_idx = int(torch.argmax(p_ent).item())
        # If contradiction test is available and yields the same index, fall back to None.
        if p_con is not None:
            con_idx = int(torch.argmax(p_con).item())
            if con_idx == ent_idx:
                return None
        return ent_idx
    except Exception:
        return None


def _mean_finite(values: Sequence[float]) -> float:
    finite: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isnan(fv) or math.isinf(fv):
            continue
        finite.append(fv)
    return sum(finite) / len(finite) if finite else float("nan")


def chunk_text_by_tokens(
    text: str,
    *,
    tokenizer,
    chunk_size: int = 350,
    chunk_stride: Optional[int] = None,
    max_chunks: Optional[int] = None,
) -> List[str]:
    """Chunk text into tokenizer windows and decode back to text.

    This is a lightweight helper (no spaCy/nltk) intended for AlignScore-style
    factuality scoring where we compare summary sentences to multiple source chunks.

    Notes:
    - Chunking is performed in tokenizer space (input_ids), not characters.
    - Decoding back to text is best-effort; minor whitespace differences are expected.
    """
    if text is None:
        return []
    s = str(text).replace("\r", "").strip()
    if not s:
        return []

    try:
        chunk_size = int(chunk_size)
    except Exception:
        chunk_size = 350
    chunk_size = max(8, chunk_size)

    if chunk_stride is None:
        chunk_stride = chunk_size
    try:
        chunk_stride = int(chunk_stride)
    except Exception:
        chunk_stride = chunk_size
    chunk_stride = max(1, chunk_stride)

    if max_chunks is not None:
        try:
            max_chunks = int(max_chunks)
        except Exception:
            max_chunks = None
        if max_chunks is not None and max_chunks <= 0:
            max_chunks = None

    # Tokenize without special tokens; we will build pair inputs later.
    try:
        input_ids = tokenizer.encode(s, add_special_tokens=False)
    except Exception:
        input_ids = []

    # If tokenization fails, fall back to a single chunk of raw text.
    if not input_ids:
        return [s]

    chunks: List[str] = []
    start = 0
    while start < len(input_ids):
        end = min(len(input_ids), start + chunk_size)
        piece_ids = input_ids[start:end]
        if not piece_ids:
            break
        try:
            chunk = tokenizer.decode(piece_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception:
            chunk = ""
        chunk = (chunk or "").strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(input_ids):
            break
        start += chunk_stride
        if max_chunks is not None and len(chunks) >= max_chunks:
            break

    return chunks


@torch.no_grad()
def score_factuality_nli(
    sources: List[str],
    summaries: List[str],
    *,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 512,
    entailment_label_id: Optional[int] = None,
    entailment_threshold: float = 0.5,
    max_sentences: Optional[int] = None,
    truncation: str = "only_first",
) -> Dict[str, float]:
    # DEPRECATED: NLI entailment proxy.
    # We keep this implementation for reference, but the repo's summarization eval
    # is intended to use AlignScore-style scoring instead.
    """Compute a lightweight factuality proxy via NLI entailment.

    We split each summary into sentences and score each sentence as a hypothesis against
    the source document as premise. We then aggregate per-summary and report dataset means.

    Returns a dict of scalar metrics suitable for logging.
    """
    n = min(len(sources), len(summaries))
    if n <= 0:
        return {
            "factuality_nli_entail_prob": float("nan"),
            "factuality_nli_sent_entail_frac": float("nan"),
        }

    # Normalize args
    try:
        batch_size = int(batch_size)
    except Exception:
        batch_size = 16
    batch_size = max(1, batch_size)

    try:
        max_length = int(max_length)
    except Exception:
        max_length = 512
    max_length = max(8, max_length)

    try:
        entailment_threshold = float(entailment_threshold)
    except Exception:
        entailment_threshold = 0.5

    if entailment_label_id is None:
        entailment_label_id = infer_entailment_label_id(model)

    # Build (premise, hypothesis) pairs for sentence-level scoring.
    pairs: List[Tuple[str, str, int]] = []
    sent_counts: List[int] = [0] * n

    for i in range(n):
        premise = extract_source_from_prompt(sources[i])
        hyp = "" if summaries[i] is None else str(summaries[i])
        sents = split_sentences(hyp, max_sentences=max_sentences)
        if not sents and hyp.strip():
            sents = [hyp.strip()]
        sent_counts[i] = len(sents)
        for sent in sents:
            pairs.append((premise, sent, i))

    if not pairs:
        return {
            "factuality_nli_entail_prob": float("nan"),
            "factuality_nli_sent_entail_frac": float("nan"),
        }

    # Run NLI in batches.
    probs_by_summary: List[List[float]] = [[] for _ in range(n)]

    model.eval()
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        premises = [p for (p, _h, _i) in chunk]
        hypotheses = [h for (_p, h, _i) in chunk]
        idxs = [i for (_p, _h, i) in chunk]

        toks = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = model(**toks)
        logits = out.logits
        if logits is None:
            continue

        # Default for MNLI-style models is entailment at index 2.
        if entailment_label_id is None:
            entailment_label_id = 2 if logits.shape[-1] >= 3 else int(logits.shape[-1] - 1)

        prob = F.softmax(logits, dim=-1)[:, int(entailment_label_id)]
        prob_list = prob.detach().float().cpu().tolist()

        for p_ent, idx in zip(prob_list, idxs):
            if 0 <= idx < n:
                probs_by_summary[idx].append(float(p_ent))

    # Aggregate per summary, then macro-average.
    mean_ent_per_summary: List[float] = []
    frac_ent_per_summary: List[float] = []
    for probs in probs_by_summary:
        if not probs:
            mean_ent_per_summary.append(float("nan"))
            frac_ent_per_summary.append(float("nan"))
            continue
        mean_ent_per_summary.append(sum(probs) / len(probs))
        frac_ent_per_summary.append(sum(1.0 for p in probs if p >= entailment_threshold) / len(probs))

    return {
        "factuality_nli_entail_prob": _mean_finite(mean_ent_per_summary),
        "factuality_nli_sent_entail_frac": _mean_finite(frac_ent_per_summary),
        "factuality_nli_avg_num_sents": _mean_finite(sent_counts),
    }


@torch.no_grad()
def score_factuality_alignscore(
    sources: List[str],
    summaries: List[str],
    *,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 512,
    entailment_label_id: Optional[int] = None,
    entailment_threshold: float = 0.5,
    max_sentences: Optional[int] = None,
    truncation: str = "only_first",
    chunk_size: int = 350,
    chunk_stride: Optional[int] = None,
    max_chunks: Optional[int] = None,
) -> Dict[str, float]:
    """Compute an AlignScore-style factuality proxy (chunked evidence, sentence-level claims).

    Algorithm (mirrors AlignScore's evaluation intuition):
    - Split source document into chunks (~350 tokens).
    - Split summary into sentences (claims).
    - Score each (source_chunk, claim_sentence) pair with an entailment/alignment model.
    - For each claim sentence, keep the max score across chunks.
    - Aggregate max scores across sentences into a per-summary score (mean).
    """
    n = min(len(sources), len(summaries))
    if n <= 0:
        # Return a sentinel (W&B may drop NaNs).
        return {"factuality_alignscore": -1.0}

    # Normalize args
    try:
        batch_size = int(batch_size)
    except Exception:
        batch_size = 16
    batch_size = max(1, batch_size)

    try:
        max_length = int(max_length)
    except Exception:
        max_length = 512
    max_length = max(8, max_length)

    try:
        entailment_threshold = float(entailment_threshold)
    except Exception:
        entailment_threshold = 0.5

    if entailment_label_id is None:
        entailment_label_id = infer_entailment_label_id(model)
    # If the model provides no label mapping (common for custom AlignScore HF ports),
    # try a tiny sanity-check inference and cache it on the model instance.
    if entailment_label_id is None:
        cached = getattr(model, "_openrlhf_entailment_label_id", None)
        if cached is not None:
            try:
                entailment_label_id = int(cached)
            except Exception:
                entailment_label_id = None
        else:
            model_type = getattr(getattr(model, "config", None), "model_type", None)
            if str(model_type) == "alignscore":
                inferred = _infer_entailment_label_id_by_sanity_check(
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=max_length,
                    truncation=truncation,
                )
                if inferred is not None:
                    entailment_label_id = int(inferred)
                    try:
                        setattr(model, "_openrlhf_entailment_label_id", int(inferred))
                    except Exception:
                        pass

    # Build (chunk, sentence) pairs for scoring.
    pairs: List[Tuple[str, str, int, int]] = []
    sent_counts: List[int] = [0] * n
    chunk_counts: List[int] = [0] * n

    for i in range(n):
        premise_full = extract_source_from_prompt(sources[i])
        chunks = chunk_text_by_tokens(
            premise_full,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_stride=chunk_stride,
            max_chunks=max_chunks,
        )
        if not chunks:
            chunks = [premise_full.strip()] if str(premise_full).strip() else [""]
        chunk_counts[i] = len(chunks)

        hyp = "" if summaries[i] is None else str(summaries[i])
        sents = split_sentences(hyp, max_sentences=max_sentences)
        if not sents and hyp.strip():
            sents = [hyp.strip()]
        sent_counts[i] = len(sents)

        # Debug logging for first 2 examples
        if i < 2:
            _trunc = lambda s, m=200: (s[:m] + "...") if len(s) > m else s
            # NOTE: keep these at DEBUG; emitting raw texts at INFO can overwhelm Ray log streaming.
            logger.debug(f"[ALIGNSCORE DEBUG i={i}] raw_source={_trunc(str(sources[i]))}")
            logger.debug(f"[ALIGNSCORE DEBUG i={i}] extracted_source={_trunc(premise_full)}")
            logger.debug(f"[ALIGNSCORE DEBUG i={i}] num_chunks={len(chunks)}, chunk0={_trunc(chunks[0]) if chunks else 'EMPTY'}")
            logger.debug(f"[ALIGNSCORE DEBUG i={i}] summary={_trunc(hyp)}")
            logger.debug(f"[ALIGNSCORE DEBUG i={i}] num_sents={len(sents)}, sents={[_trunc(s, 80) for s in sents]}")

        for j, sent in enumerate(sents):
            for ch in chunks:
                pairs.append((ch, sent, i, j))

    if not pairs:
        # Return a sentinel (W&B may drop NaNs).
        return {"factuality_alignscore": -1.0}

    model.eval()
    best_by_sent: Dict[Tuple[int, int], float] = {}

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        premises = [p for (p, _h, _i, _j) in chunk]
        hypotheses = [h for (_p, h, _i, _j) in chunk]
        keys = [(i, j) for (_p, _h, i, j) in chunk]

        toks = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = model(**toks)
        # Robustly extract logits across HF / custom-model output types.
        logits = getattr(out, "logits", None)
        if logits is None:
            # Some custom ModelOutput types expose `score` / `scores` instead of `logits`.
            # AlignScore-large-hf uses `tri_label_logits` for 3-class NLI output.
            for attr in (
                "tri_label_logits",  # AlignScore 3-class NLI (entailment/neutral/contradiction)
                "seq_relationship_logits",  # AlignScore binary relation
                "reg_label_logits",  # AlignScore regression output
                "score", "scores", "logit", "prediction", "pred", "prob", "probs",
            ):
                if hasattr(out, attr):
                    logits = getattr(out, attr)
                    break

        if logits is None:
            if isinstance(out, dict):
                logits = out.get("logits", None)
                if logits is None:
                    # Common alternative keys in some custom heads.
                    for k in (
                        "tri_label_logits", "seq_relationship_logits", "reg_label_logits",
                        "score", "scores", "logit", "prediction",
                    ):
                        if k in out:
                            logits = out[k]
                            break
            elif isinstance(out, (tuple, list)) and len(out) > 0:
                logits = out[0]
            elif hasattr(out, "to_tuple"):
                # HF's ModelOutput often supports `to_tuple()` even when it's not a dict/tuple.
                try:
                    t = out.to_tuple()
                    if isinstance(t, (tuple, list)) and len(t) > 0:
                        logits = t[0]
                except Exception:
                    pass

        if logits is None:
            raise RuntimeError(
                f"AlignScore model forward returned no logits (type={type(out)}). "
                "If this is a custom model output, update logits extraction in score_factuality_alignscore."
            )

        if not isinstance(logits, torch.Tensor):
            try:
                logits = torch.as_tensor(logits)
            except Exception as e:
                raise RuntimeError(f"AlignScore logits is not a Tensor and cannot be converted: {type(logits)}: {e}")

        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)

        # Compute probabilities in fp32 for numerical stability.
        logits_f = logits.float()
        if not torch.isfinite(logits_f).all():
            # In strict eval, non-finite logits indicate a real failure (bad dtype / model instability).
            raise RuntimeError("AlignScore model produced non-finite logits.")

        if logits.size(-1) == 1:
            # If we couldn't infer an entailment label id, treat the model as producing a
            # direct alignment score in [0,1] (AlignScore-large-hf behaves like this).
            # If values fall outside [0,1], fall back to sigmoid(logit).
            prob_raw = logits_f.squeeze(-1)
            if entailment_label_id is None and torch.isfinite(prob_raw).all():
                min_v = float(prob_raw.min().item()) if prob_raw.numel() else 0.0
                max_v = float(prob_raw.max().item()) if prob_raw.numel() else 1.0
                if (min_v >= -1e-4) and (max_v <= 1.0 + 1e-4):
                    prob = prob_raw.clamp(0.0, 1.0)
                else:
                    prob = torch.sigmoid(prob_raw)
            else:
                prob = torch.sigmoid(prob_raw)
        else:
            if entailment_label_id is None:
                entailment_label_id = 2 if logits.size(-1) >= 3 else int(logits.size(-1) - 1)
            prob = F.softmax(logits_f, dim=-1)[:, int(entailment_label_id)]

        prob_list = prob.detach().float().cpu().tolist()
        for p_ent, key in zip(prob_list, keys):
            prev = best_by_sent.get(key)
            pe = float(p_ent)
            if prev is None or pe > prev:
                best_by_sent[key] = pe

    if not best_by_sent:
        raise RuntimeError(
            "AlignScore produced no sentence scores (best_by_sent is empty). "
            "This likely means logits extraction failed or all batches were skipped."
        )

    # Aggregate per example (mean over sentence maxima; frac of aligned sentences).
    mean_score_per_example: List[float] = []
    frac_aligned_per_example: List[float] = []
    for i in range(n):
        m = sent_counts[i]
        if m <= 0:
            mean_score_per_example.append(float("nan"))
            frac_aligned_per_example.append(float("nan"))
            continue

        sent_scores = [best_by_sent.get((i, j), float("nan")) for j in range(m)]
        finite_scores: List[float] = []
        aligned = 0
        for s in sent_scores:
            try:
                fs = float(s)
            except Exception:
                continue
            if math.isnan(fs) or math.isinf(fs):
                continue
            finite_scores.append(fs)
            if fs >= entailment_threshold:
                aligned += 1

        if not finite_scores:
            mean_score_per_example.append(float("nan"))
            frac_aligned_per_example.append(float("nan"))
        else:
            mean_score_per_example.append(sum(finite_scores) / len(finite_scores))
            frac_aligned_per_example.append(aligned / len(finite_scores))

    # Debug: log first few per-example scores
    final_score = _mean_finite(mean_score_per_example)
    logger.debug(f"[ALIGNSCORE DEBUG] n={n}, final_score={final_score:.4f}, first_5_scores={mean_score_per_example[:5]}")

    # Minimal logging: expose only the single central score to avoid confusion.
    return {"factuality_alignscore": final_score}



