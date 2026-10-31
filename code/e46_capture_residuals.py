#!/usr/bin/env python3
# =============================================================================
# E46 -- RESIDUAL CAPTURE AT THE FROZEN PROBE SCHEDULE
#
# Replays each stored E46 pair through the model and stores the pre-final-layer
# residual at every anchor. --all-layers also saves every block at those same
# checkpoints, plus block input and final normalized output. No generation happens:
# the text is fixed, so this is a forward pass over prompt + already-generated
# ids, and the run is exactly reproducible from the traces pkl.
#
# ANCHORS (see PROBE_ANCHORS in e46_prompts_cue.py for why each exists)
#
#   prompt            the prompt's LAST token, nothing generated. If Z is
#                     decodable here the model knew it would follow the cue
#                     before writing a word -- the strongest available result.
#   frac25/50/75      quarters of the PRE-ONSET span, to measure the U-shape
#                     Dong et al. report rather than inherit it.
#   onset             first commitment to the final answer.
#   onset - {1,4,8,16}  token space, the original schedule.
#
# BOTH ARMS ARE CAPTURED. The cued arm carries the primary contrast; the uncued
# arm is the `state_uncued` susceptibility control -- what makes an item
# flippable absent any cue -- and it is a DIFFERENT context, aligned by its own
# onset, never a token-wise subtraction against the cued arm.
#
#     python e46_capture_residuals.py --traces e46_traces_...pkl --out r.pkl
#     python e46_capture_residuals.py --selftest        # no GPU, no model
# =============================================================================
from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime
from typing import Any, Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e46_prompts_cue import (                                   # noqa: E402
    ONSET_BACKOFFS, PROBE_ANCHORS, Item, episode_span, label_pair)
from e46_layer_capture import (add_layer_payload, checkpoint_states,
                               layer_metadata, resolved_checkpoints)

RESID_LAYER = -2          # pre-final-layer residual, as E1/E2/E1w
DTYPE = np.float16

# =============================================================================
# OPTION LOGITS -- THE MODEL'S OWN ANSWER STATE, NOT A LEARNED ONE
#
# A trained answer-state probe on these residuals is under-powered. Measured on
# the n=100 run: trained on the uncued arm only (so the cue-letter shortcut
# cannot form), 4-way accuracy on held-out flipper rows peaks at 0.394 against
# a chance of 0.25 -- above chance, below anything that can carry a readout.
# Pooling both arms scores 0.85 on the cued arm and chance on the uncued one,
# because it has learned to find the cue letter in the prompt rather than the
# selected answer.
#
# A logit lens needs no training and so has no n. Projecting the residual
# through the model's own final norm and unembedding gives a distribution over
# the option letters directly, which is the quantity the answer-state analysis
# wanted in the first place:
#
#     logit(cue) - logit(abandoned)     did the cue win, and by how much
#     logit(cue) - logit(correct)
#     entropy over A/B/C/D              how committed is the state
#     the change in each from uncued to cued
#
# These are stored PER ANCHOR at capture time, when the model is loaded, so the
# analysis side needs no GPU and no probe. Older residual caches simply lack
# the field.
# =============================================================================
OPTION_FORMS = ("{L}", " {L}", "({L}", " ({L}")


# ------------------------------------------------------- position arithmetic

def char_ends(ids: list[int], decode_one: Callable[[int], str],
              decode_prefix: Callable[[list[int]], str] | None = None
              ) -> np.ndarray:
    """Character offset at which each generated token ends.

    PER-TOKEN DECODING IS NOT SAFE ON BPE. `sum(len(decode([t])))` assumes the
    pieces concatenate to the whole, which byte-level BPE breaks: leading
    spaces, byte fallbacks and merges mean decode([a]) + decode([b]) can differ
    in length from decode([a, b]). Getting this wrong shifts every anchor by a
    token or two, which is precisely the corruption the onset design exists to
    avoid -- so when a prefix decoder is available the offsets are taken by
    INCREMENTAL decoding, which is exact by construction, and the per-token
    path is kept only as a fallback for the selftest's toy tokenizer.

    `verify_alignment` checks the result against the real generated text.
    """
    if decode_prefix is not None:
        return np.array([len(decode_prefix(ids[:k + 1]))
                         for k in range(len(ids))], dtype=np.int64)
    return np.cumsum([len(decode_one(t)) for t in ids])


def verify_alignment(ids: list[int], text: str,
                     decode_prefix: Callable[[list[int]], str]) -> dict:
    """Does the decoded generation reproduce the stored text exactly?

    Runs where the real tokenizer is (Colab), not in the offline selftest. A
    mismatch means the stored `text` and `ids` disagree, and every character
    offset -- onset included -- is landing on the wrong token.
    """
    dec = decode_prefix(ids)
    ce = char_ends(ids, lambda t: "", decode_prefix)
    return {"decoded_len": len(dec), "stored_len": len(text),
            "exact_match": dec == text,
            "prefix_match": text.startswith(dec[:min(len(dec), len(text))]),
            "last_char_end": int(ce[-1]) if len(ce) else 0,
            "monotonic": bool(np.all(np.diff(ce) >= 0)) if len(ce) > 1 else True}


def anchor_token_indices(rec_ids: list[int], text: str, item: Item,
                         final: str | None,
                         decode_one: Callable[[int], str],
                         positions: dict[str, Any],
                         decode_prefix: Callable[[list[int]], str] | None = None
                         ) -> dict[str, int | None]:
    """Anchor name -> index into the GENERATED ids, or None.

    `None` means one of two very different things and the caller must not
    conflate them:
      * "prompt"  -> deliberately outside the generation; read the prompt's
                     last token.
      * any other -> the anchor could not be resolved (no onset). The row is
                     EXCLUDED. Reading it at token 0 instead would silently
                     relabel "we could not find the decision" as "the decision
                     was at the first token".
    """
    if not rec_ids:
        return {a: None for a in PROBE_ANCHORS}
    ce = char_ends(rec_ids, decode_one, decode_prefix)
    last = len(rec_ids) - 1

    def tok_at(c: int | None) -> int | None:
        """The token CONTAINING character `c`.

        `char_ends` holds END offsets, so token j covers [ce[j-1], ce[j]) and
        the containing token is the first with ce[j] > c -- side='right'.
        Elsewhere in the project (E10, E11) the same call uses the default
        side='left' because those offsets are ENDS of step lines, where the
        token that finishes there is the one wanted. The anchors here are
        match STARTS, so the sides differ by one token and the distinction has
        to be made explicitly rather than inherited.
        """
        if c is None:
            return None
        return int(min(np.searchsorted(ce, c, side="right"), last))

    out: dict[str, int | None] = {"prompt": None}
    for a in ("frac25", "frac50", "frac75", "onset"):
        out[a] = tok_at(positions.get(a))
    on = out["onset"]
    for k in ONSET_BACKOFFS:
        # A backoff that runs off the front of the generation is dropped, not
        # clamped to 0: clamping would pile several k values onto the same
        # token and silently duplicate rows.
        out[f"onset-{k}"] = (on - k) if (on is not None and on - k >= 0) else None
    return out


def anchor_names() -> list[str]:
    return list(PROBE_ANCHORS) + [f"onset-{k}" for k in ONSET_BACKOFFS]


# =============================================================================
# THE FOLLOW-UP TURN NEEDS ITS OWN ANCHORS AND ITS OWN LENS
#
# Turn 1's schedule is built around the ONSET -- the first commitment to an
# answer letter. A follow-up has no answer to commit to; it is an explanation.
# So the anchors move:
#
#   q_end     the LAST PROMPT TOKEN, right after "Explanation:" and before a
#             single word of justification exists. THE ONE THAT MATTERS. The
#             model has just read its own answer and been asked why. If the
#             cue is live anywhere, it is live here, and nothing it writes
#             afterwards has had a chance to rationalise it away.
#   frac25/50/75, end   through the explanation itself.
#   mention   the first token that names the cue, where one exists. Only
#             defined on the rows that DID cite it, so it is a within-mention
#             anchor and never a silent-row default.
#
# AND THE LENS MOVES TOO. Option-letter logits are the wrong probe for an
# explanation: the question is not which answer is held but whether the CUE'S
# OWN VOCABULARY is active while the model declines to use it. `cue_logits`
# reads the tokens a mention would be made of -- professor, Stanford, expert --
# against a matched set of neutral words drawn from the same prompt, so a
# generic "this position raises every content word" effect cancels.
FOLLOWUP_ANCHORS = ("q_end", "frac25", "frac50", "frac75", "end", "mention")

CUE_VOCAB = {
    "authority": ("professor", "Stanford", "expert", "academic"),
    "sycophancy": ("you", "your", "yours", "suggested"),
    "pattern": ("pattern", "examples", "always", "previous"),
}
# Matched neutral controls: content words of similar frequency that appear in
# this substrate but carry no cue meaning. Without them a raised cue logit
# cannot be told from a raised logit on everything.
NEUTRAL_VOCAB = ("movie", "genre", "film", "similar")


# --------------------------------------------------------------- the capture

def _check_capture_output(source, output, all_layers=False):
    if os.path.realpath(source) == os.path.realpath(output):
        raise ValueError("Refusing to overwrite the input cache; choose a new --out")
    if all_layers and os.path.exists(output):
        raise ValueError(f"All-layer output already exists: {output}; choose a new --out")


def default_capture_path(source, followup=False, all_layers=False):
    directory, name = os.path.split(source)
    prefix = "e46_followup_" if followup else "e46_traces_"
    if not name.startswith(prefix):
        raise ValueError("Cannot derive an output name; pass --out")
    new_prefix = "e46_furesid_" if followup else "e46_resid_"
    name = new_prefix + name[len(prefix):]
    if all_layers:
        stem, ext = os.path.splitext(name)
        name = stem + "_all_layers" + ext
    return os.path.join(directory, name)


def capture(traces_path: str, out_path: str, device: str = "cuda",
            dtype: str | None = None, arms=("cued", "uncued"),
            limit: int | None = None,
            device_map: str | None = None,
            score_titles: bool = True, all_layers: bool = False) -> dict[str, Any]:
    _check_capture_output(traces_path, out_path, all_layers)
    import torch
    from transformers import AutoModelForCausalLM

    blob = pickle.load(open(traces_path, "rb"))
    if blob.get("kind") != "e46_pretest_traces":
        raise SystemExit(f"{traces_path} is not an E46 pretest cache")
    meta = blob["metadata"]
    rows_in = blob["raw_outputs"][meta["task_names"][0]]
    by = {c: {r["index"]: r for r in rows_in if r["condition"] == c}
          for c in ("uncued", "cued")}
    hf = meta["hf"]

    # THE PROJECT'S LOADER, not a bare AutoTokenizer. `load_tokenizer` carries
    # the fallbacks and, critically, refuses to substitute a different repo's
    # tokenizer -- silently swapping tokenizers in an information-theoretic
    # experiment corrupts every offset and every token count downstream.
    from p2_common import load_tokenizer
    revision_kw = {"revision": meta["hf_revision"]} if meta.get("hf_revision") else {}
    tok = load_tokenizer(hf, trust_remote_code=True, **revision_kw)
    td = getattr(torch, dtype) if dtype else (
        torch.bfloat16 if device == "cuda" else torch.float32)
    kw = {"torch_dtype": td, "trust_remote_code": True, **revision_kw}
    if device_map:
        kw["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(hf, **kw)
    if not device_map:
        model = model.to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    layer_info = layer_metadata(model, final_norm(model)) if all_layers else None
    layer_batches = [] if all_layers else None
    print(f"  {hf}: {n_layers} layers; "
          + ("saving all layers at checkpoints" if all_layers
             else f"reading hidden_states[{RESID_LAYER}]"))

    # THE DECODER MUST MATCH HOW THE TEXT WAS STORED, and that is not
    # something to assume. `e46_pretest_cue.generate_traces` writes
    # `tok.decode(new, skip_special_tokens=True)`; decoding here WITHOUT that
    # flag renders `<|endoftext|>` as thirteen literal characters, so every
    # character offset past the first special token is shifted and the check
    # correctly refused to capture (0/40, decoded 629 vs stored 616 -- exactly
    # len("<|endoftext|>")). Rather than hard-code the flag, both settings are
    # tried against the stored text and the one that reproduces it wins; if
    # neither does, the ids and the text genuinely disagree and no anchor is
    # trustworthy.
    def _mk_decoders(skip: bool):
        return (lambda t: tok.decode([t], skip_special_tokens=skip),
                lambda seq: tok.decode(list(seq), skip_special_tokens=skip))

    sample = []
    for i in sorted(by["cued"])[:20]:
        for arm_ in ("cued", "uncued"):
            r_ = by[arm_].get(i)
            if r_ and r_["ids"]:
                sample.append(r_)
    scores = {}
    for skip in (True, False):
        _, dp = _mk_decoders(skip)
        scores[skip] = sum(verify_alignment(r_["ids"], r_["text"], dp)
                           ["exact_match"] for r_ in sample)
    skip_best = max(scores, key=lambda k: scores[k])
    decode_one, decode_prefix = _mk_decoders(skip_best)
    print(f"  alignment: {scores[skip_best]}/{len(sample)} sampled generations "
          f"decode EXACTLY back to the stored text "
          f"(skip_special_tokens={skip_best})")
    if sample and scores[skip_best] < len(sample):
        bad = next(verify_alignment(r_["ids"], r_["text"], decode_prefix)
                   for r_ in sample
                   if not verify_alignment(r_["ids"], r_["text"],
                                           decode_prefix)["exact_match"])
        print(f"  ** MISMATCH under both decoder settings "
              f"(best skip_special_tokens={skip_best}, "
              f"{scores[True]} vs {scores[False]} exact).\n"
              f"  ** decoded {bad['decoded_len']} chars vs stored "
              f"{bad['stored_len']}: the ids and the stored text disagree, so\n"
              f"  ** onset anchors would be wrong. Refusing to capture.")
        raise SystemExit("tokenizer/text alignment check failed")

    # Token ids for each option letter in the spellings a trace actually uses.
    letter_ids = {}
    for L in "ABCD":
        ids_ = set()
        for form in OPTION_FORMS:
            enc = tok.encode(form.format(L=L), add_special_tokens=False)
            if enc:
                ids_.add(int(enc[-1]))
        letter_ids[L] = sorted(ids_)
    print(f"  option letter token ids: "
          + ", ".join(f"{L}={letter_ids[L]}" for L in "ABCD"))
    print(f"  final norm: {type(final_norm(model)).__name__}   "
          f"output head: {type(model.get_output_embeddings()).__name__}")

    def title_seq_ids(item) -> dict[str, list[int]]:
        """FULL token sequence of each option title, with a leading space."""
        return {L: tok.encode(" " + str(v), add_special_tokens=False)
                for L, v in (item.options or {}).items()}

    def title_ids(item) -> dict[str, list[int]]:
        """First token of each option's TEXT, in both spacings.

        ON THIS SUBSTRATE THE LETTER LENS IS THE WRONG LENS AT THE POSITIONS
        THAT MATTER. The official BBH demonstrations commit by naming the film
        -- "seems to be Lamerica (comedy, drama, romance; 1994)" -- and only
        then write "So the answer is (C)". Measured on the n=100 run, the
        option text precedes the letter in 59 of 60 episodes by a median 104
        characters. A pure A/B/C/D lens therefore reads the state AFTER the
        decision it is trying to catch; the title lens reads it at the decision.
        """
        out = {}
        for L, val in (item.options or {}).items():
            s = str(val)
            ids_ = set()
            for form in (s, " " + s):
                enc = tok.encode(form, add_special_tokens=False)
                if enc:
                    ids_.add(int(enc[0]))
            out[L] = sorted(ids_)
        return out

    R, rows = [], []
    idxs = sorted(by["cued"])
    if limit:
        idxs = idxs[:limit]
    for n, i in enumerate(idxs):
        if i not in by["uncued"]:
            continue
        c, u = by["cued"][i], by["uncued"][i]
        item = Item(**{k: v for k, v in c["item"].items() if k != "condition"})
        lb = label_pair(item, u["text"], c["text"])
        for arm in arms:
            rec = c if arm == "cued" else u
            if not verify_alignment(rec["ids"], rec["text"], decode_prefix)["exact_match"]:
                raise ValueError(f"Token/text alignment failed: {i}/{arm}")
            final = lb[f"{arm}_answer"]
            pos = (lb["probe_positions"] if arm == "cued"
                   else _positions_for(rec["text"], item, final))
            anchors = anchor_token_indices(rec["ids"], rec["text"], item,
                                           final, decode_one, pos,
                                           decode_prefix)
            t_ids = title_ids(item)
            t_seq = title_seq_ids(item) if score_titles else None
            input_device = model.get_input_embeddings().weight.device
            inp = tok(rec["prompt"], return_tensors="pt").input_ids.to(input_device)
            sl = int(inp.shape[1])
            full = torch.cat(
                [inp, torch.tensor(rec["ids"], device=input_device,
                                   dtype=torch.long).unsqueeze(0)], 1)
            selected = resolved_checkpoints(anchors, anchor_names(), sl, "prompt")
            hid_t, layers = checkpoint_states(model, full, [s[2] for s in selected],
                                              all_layers=all_layers, norm=final_norm(model))
            if all_layers:
                layer_batches.append(layers)
            for j, (a, t_idx, abs_idx) in enumerate(selected):
                R.append(hid_t[j].float().cpu().numpy().astype(DTYPE))
                rows.append({"index": i, "arm": arm, "anchor": a,
                             "option_logits": _option_logits(
                                 model, hid_t[j], letter_ids),
                             "title_logits": _option_logits(
                                 model, hid_t[j], t_ids),
                             "title_seq": (
                                 _title_sequence_scores(model, full, abs_idx,
                                                        t_seq, input_device)
                                 if score_titles else None),
                             "token_in_generation": t_idx,
                             "abs_token": abs_idx,
                             "anchor_token_text": decode_one(rec["ids"][t_idx])
                                                  if t_idx is not None else None,
                             "anchor_contract": "residual after containing token; lens predicts next token",
                             "stratum": lb["stratum"],
                             "mentioned": lb["mentioned"],
                             "cue_letter": item.cue_letter,
                             "correct_letter": item.correct_letter,
                             "uncued_answer": lb["uncued_answer"],
                             "cued_answer": lb["cued_answer"],
                             "retarget_want": getattr(item, "retarget_want",
                                                      None)})
            del hid_t, layers
        if (n + 1) % 10 == 0:
            print(f"    {n + 1}/{len(idxs)}", flush=True)

    payload = {
        "kind": "e46_residuals",
        "metadata": {**meta, "resid_layer": RESID_LAYER, "n_layers": n_layers,
                     "anchors": anchor_names(), "arms": list(arms),
                     "has_option_logits": True, "has_title_logits": True,
                     "has_title_seq": bool(score_titles),
                     "source_traces": os.path.basename(traces_path),
                     "created": datetime.now().isoformat(timespec="seconds"),
                     "note": "E46 residuals at the frozen probe schedule."},
        "R": np.stack(R) if R else np.zeros((0, model.config.hidden_size), DTYPE),
        "rows": rows,
    }
    add_layer_payload(payload, layer_batches, layer_info, model.config.hidden_size)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "xb" if all_layers else "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  wrote {out_path}   {len(rows)} rows, "
          f"{payload['R'].shape[1] if len(rows) else 0}-d")
    if all_layers:
        print(f"  R_layers: {payload['R_layers'].shape} float16; "
              f"legacy R = layer-site {layer_info['legacy_R_layer_axis']}")
    return payload


def final_norm(model):
    """The pre-unembedding norm, whatever this architecture calls it.

    Llama/Qwen/Mistral put it at `model.model.norm`; GPT-2 at
    `transformer.ln_f`; GPT-NeoX at `gpt_neox.final_layer_norm`; Falcon at
    `transformer.ln_f` too. Hard-coding one of them silently skips the norm on
    every other family, which does not error -- it just returns logits from an
    unnormalised residual, quietly wrong.
    """
    for path in (("model", "norm"), ("transformer", "ln_f"),
                 ("gpt_neox", "final_layer_norm"), ("model", "final_layernorm"),
                 ("norm",), ("ln_f",)):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _option_logits(model, h, token_ids: dict[str, list[int]]) -> dict[str, float]:
    """Logit lens: the residual through the model's own head, per option.

    `get_output_embeddings()` rather than `.lm_head`, which only some families
    expose under that name and which is None on tied-weight configurations
    where the head must be fetched through the API.

    Takes the MAX over the spellings of an option rather than the sum, so a
    tokenizer that splits "(A" differently from " A" cannot inflate one option
    by happening to have more surface forms.
    """
    import torch
    head = model.get_output_embeddings()
    if head is None:
        return {}
    with torch.inference_mode():
        x = h.unsqueeze(0)
        norm = final_norm(model)
        if norm is not None:
            x = norm(x)
        logits = head(x)[0].float()
    return {k: float(max(logits[i].item() for i in ids)) if ids else 0.0
            for k, ids in token_ids.items()}


def _title_sequence_scores(model, full_ids, abs_idx, title_tok, device):
    """Teacher-forced log P(title | prefix up to and including `abs_idx`).

    THE FIRST-TOKEN TITLE LENS IS NOT A TITLE LENS. Scoring only the first
    token of each option measures the article or the first word -- "The", "A"
    -- and on movie_recommendation first-word collisions are common (7 of 33
    flippers). The placebo control caught the consequence: on
    already_cue_correct pairs, where the "abandoned" option is chosen at
    random and the lead should be ~0, the first-token title lens showed leads
    up to +1.93 logits. That is title-token bias, not answer state.

    So the full sequence is scored. Both forms are returned because they
    answer different questions and neither dominates:

        sum   total evidence for the title -- but longer titles score lower
        mean  per-token, length-normalised -- but a long title with one
              surprising token is penalised less than it should be

    Reported side by side so a length artefact is visible rather than baked in.
    """
    import torch
    out = {}
    prefix = full_ids[0, :abs_idx + 1]
    for L, ids in title_tok.items():
        if not ids:
            out[L] = {"sum": 0.0, "mean": 0.0, "n_tokens": 0}
            continue
        t = torch.tensor(ids, device=device)
        seq = torch.cat([prefix, t]).unsqueeze(0)
        with torch.inference_mode():
            lg = model(input_ids=seq,
                       attention_mask=torch.ones_like(seq)).logits[0].float()
        lp = torch.log_softmax(lg[len(prefix) - 1:-1], dim=-1)
        tot = float(lp[torch.arange(len(ids)), t].sum().item())
        out[L] = {"sum": tot, "mean": tot / len(ids), "n_tokens": len(ids)}
    return out


def followup_anchor_indices(ids, text, mention_char, decode_one,
                            decode_prefix=None) -> dict[str, int | None]:
    """Anchor -> index into the EXPLANATION's ids, or None.

    `q_end` is None by construction: it lives at the last PROMPT token, before
    the explanation starts. `mention` is None on rows that never cited the cue,
    and those rows are simply absent from that anchor rather than defaulted to
    a position -- a silent row has no mention to anchor on.
    """
    if not ids:
        return {a: None for a in FOLLOWUP_ANCHORS}
    ce = char_ends(ids, decode_one, decode_prefix)
    last = len(ids) - 1
    span = len(episode_span(text))

    def tok_at(c):
        if c is None:
            return None
        return int(min(np.searchsorted(ce, c, side="right"), last))

    return {"q_end": None,
            "frac25": tok_at(int(0.25 * span)),
            "frac50": tok_at(int(0.50 * span)),
            "frac75": tok_at(int(0.75 * span)),
            "end": tok_at(max(0, span - 1)),
            "mention": tok_at(mention_char)}


def first_mention_char(text: str, cue_type: str) -> int | None:
    """Character offset of the first cue mention in the explanation, or None."""
    import re as _re
    from e46_prompts_cue import MENTION_BY_TYPE
    ep = episode_span(text)
    best = None
    for p in MENTION_BY_TYPE.get(cue_type, ()):
        m = _re.search(p, ep, _re.I)
        if m and (best is None or m.start() < best):
            best = m.start()
    return best


def _vocab_ids(tok, words) -> dict[str, list[int]]:
    """First-token ids for each word, in both spacings."""
    out = {}
    for w in words:
        ids_ = set()
        for form in (w, " " + w, w.lower(), " " + w.lower()):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                ids_.add(int(enc[0]))
        out[w] = sorted(ids_)
    return out


def capture_followups(fu_path: str, out_path: str, device: str = "cuda",
                      dtype: str | None = None, limit: int | None = None,
                      device_map: str | None = None, all_layers: bool = False,
                      arms=("cued", "uncued")) -> dict[str, Any]:
    """Residuals over the FOLLOW-UP turn. See FOLLOWUP_ANCHORS."""
    _check_capture_output(fu_path, out_path, all_layers)
    import torch
    from transformers import AutoModelForCausalLM
    from p2_common import load_tokenizer
    from e46_cache import followup_fingerprint

    blob = pickle.load(open(fu_path, "rb"))
    if blob.get("kind") != "e46_followups":
        raise SystemExit(f"{fu_path} is not an E46 follow-up cache")
    meta = blob["metadata"]
    rows_in = [r for r in blob["rows"] if r.get("ids2") and r["arm"] in arms]
    if not rows_in:
        raise SystemExit(
            f"{fu_path} has no generated follow-ups (no `ids2`). Run "
            f"e46_followup.py without --dry-run first.")
    hf = meta["hf"]
    revision_kw = {"revision": meta["hf_revision"]} if meta.get("hf_revision") else {}
    tok = load_tokenizer(hf, trust_remote_code=True, **revision_kw)
    td = getattr(torch, dtype) if dtype else (
        torch.bfloat16 if device == "cuda" else torch.float32)
    kw = {"torch_dtype": td, "trust_remote_code": True, **revision_kw}
    if device_map:
        kw["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(hf, **kw)
    if not device_map:
        model = model.to(device)
    model.eval()
    layer_info = layer_metadata(model, final_norm(model)) if all_layers else None
    layer_batches = [] if all_layers else None
    print(f"  {hf}: {len(rows_in)} follow-ups; "
          + ("saving all layers at checkpoints" if all_layers
             else f"reading hidden_states[{RESID_LAYER}]"))
    print(f"  final norm: {type(final_norm(model)).__name__}   "
          f"output head: {type(model.get_output_embeddings()).__name__}")

    def d1(t):
        return tok.decode([t], skip_special_tokens=True)

    def dp(seq):
        return tok.decode(list(seq), skip_special_tokens=True)

    cue_type = meta.get("cue_type", "authority")
    cue_ids = _vocab_ids(tok, CUE_VOCAB.get(cue_type, CUE_VOCAB["authority"]))
    neut_ids = _vocab_ids(tok, NEUTRAL_VOCAB)
    letter_ids = {}
    for L in "ABCD":
        s = set()
        for form in OPTION_FORMS:
            enc = tok.encode(form.format(L=L), add_special_tokens=False)
            if enc:
                s.add(int(enc[-1]))
        letter_ids[L] = sorted(s)

    R, rows = [], []
    for n, r in enumerate(rows_in[:limit] if limit else rows_in):
        if not verify_alignment(r["ids2"], r["text2"], dp)["exact_match"]:
            raise ValueError(f"Follow-up token/text alignment failed: {r['index']}/{r['arm']}")
        anchors = followup_anchor_indices(
            r["ids2"], r["text2"],
            first_mention_char(r["text2"], r.get("cue_type") or cue_type),
            d1, dp)
        inp = tok(r["prompt2"], return_tensors="pt").input_ids.to(
            model.get_input_embeddings().weight.device)
        sl = int(inp.shape[1])
        full = torch.cat(
            [inp, torch.tensor(r["ids2"], device=inp.device, dtype=torch.long).unsqueeze(0)], 1)
        selected = resolved_checkpoints(anchors, FOLLOWUP_ANCHORS, sl, "q_end")
        hid_t, layers = checkpoint_states(model, full, [s[2] for s in selected],
                                          all_layers=all_layers, norm=final_norm(model))
        if all_layers:
            layer_batches.append(layers)
        for j, (a, ti, abs_i) in enumerate(selected):
            R.append(hid_t[j].float().cpu().numpy().astype(DTYPE))
            rows.append({
                "index": r["index"], "arm": r["arm"], "anchor": a,
                "stratum": r["stratum"], "answer": r["answer"],
                "cue_letter": r["cue_letter"],
                "correct_letter": r.get("correct_letter"),
                "mentions_cue_turn2": r.get("mentions_cue_turn2"),
                "mentioned_turn1": r.get("mentioned_turn1"),
                "token_in_generation": ti, "abs_token": abs_i,
                "anchor_token_text": d1(r["ids2"][ti]) if ti is not None else None,
                "anchor_contract": "residual after containing token; lens predicts next token",
                "option_logits": _option_logits(model, hid_t[j],
                                                letter_ids),
                "cue_logits": _option_logits(model, hid_t[j], cue_ids),
                "neutral_logits": _option_logits(model, hid_t[j],
                                                 neut_ids)})
        del hid_t, layers
        if (n + 1) % 20 == 0:
            print(f"    {n + 1}/{len(rows_in)}", flush=True)

    payload = {"kind": "e46_followup_residuals",
               "metadata": {**meta, "resid_layer": RESID_LAYER,
                            "n_layers": model.config.num_hidden_layers,
                            "arms": list(arms),
                            "anchors": list(FOLLOWUP_ANCHORS),
                            "cue_vocab": list(cue_ids),
                            "neutral_vocab": list(neut_ids),
                            "source_followups": os.path.basename(fu_path),
                            "source_followups_sha256": followup_fingerprint(blob["rows"]),
                            "created": datetime.now().isoformat(
                                timespec="seconds")},
               "R": np.stack(R) if R else np.zeros((0, model.config.hidden_size), DTYPE),
               "rows": rows}
    add_layer_payload(payload, layer_batches, layer_info, model.config.hidden_size)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "xb" if all_layers else "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  wrote {out_path}   {len(rows)} rows")
    if all_layers:
        print(f"  R_layers: {payload['R_layers'].shape} float16; "
              f"legacy R = layer-site {layer_info['legacy_R_layer_axis']}")
    return payload


def _positions_for(text: str, item: Item, final: str | None) -> dict[str, Any]:
    """Anchors for the UNCUED arm, aligned by its own onset."""
    from e46_prompts_cue import probe_positions
    return probe_positions(text, item, final)


# ----------------------------------------------------------------- selftest

def _selftest() -> int:
    ok = [0, 0]

    def chk(name, cond):
        ok[1] += 1
        ok[0] += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # A toy "tokenizer": one character per token, so char offsets and token
    # indices coincide and the arithmetic is checkable by hand.
    text = "abcdefghijklmnopqrstuvwxyz0123456789"
    ids = list(range(len(text)))
    dec = lambda t: text[t]                                    # noqa: E731
    it = Item(index=0, expr="q", correct_value="X",
              options={"A": "X", "B": "Y", "C": "Z", "D": "W"},
              correct_letter="A", cue_letter="A", bin="correct_at_cue",
              task="movie_recommendation", cue_type="authority")

    ce = char_ends(ids, dec)
    chk("char_ends is cumulative and ends at the text length",
        len(ce) == len(ids) and ce[-1] == len(text))

    pos = {"prompt": None, "frac25": 5, "frac50": 10, "frac75": 15,
           "onset": 20, "onset_backoffs": list(ONSET_BACKOFFS)}
    a = anchor_token_indices(ids, text, it, "A", dec, pos)
    chk("every declared anchor is produced",
        set(a) == set(anchor_names()))
    chk("'prompt' is None BY DESIGN -- it lives outside the generation",
        a["prompt"] is None)
    chk("char offsets map to token indices (1 char per token here)",
        (a["frac25"], a["frac50"], a["frac75"], a["onset"]) == (5, 10, 15, 20))
    chk("onset - k is taken in TOKEN space from the onset token",
        [a[f"onset-{k}"] for k in ONSET_BACKOFFS] == [19, 16, 12, 4])

    # A backoff running off the front must be DROPPED, not clamped -- clamping
    # would put several k values on the same token and duplicate rows.
    a2 = anchor_token_indices(ids, text, it, "A", dec,
                              {**pos, "onset": 3})
    chk("a backoff before the first token is dropped, not clamped to 0",
        a2["onset-1"] == 2 and a2["onset-4"] is None
        and a2["onset-16"] is None)
    chk("and the ones that fit are unaffected",
        a2["onset"] == 3)

    # No onset -> the anchor is unresolved and the caller must exclude it.
    a3 = anchor_token_indices(ids, text, it, "A", dec,
                              {**pos, "onset": None})
    chk("an unresolved onset stays None rather than collapsing to token 0",
        a3["onset"] is None
        and all(a3[f"onset-{k}"] is None for k in ONSET_BACKOFFS))
    chk("but the fractional anchors still resolve",
        a3["frac50"] == 10)

    chk("an empty generation yields no usable anchor",
        all(v is None for v in
            anchor_token_indices([], "", it, "A", dec, pos).values()))

    # Offsets past the end clamp to the last token: a trace whose onset sits in
    # the final token must still be readable.
    a4 = anchor_token_indices(ids, text, it, "A", dec,
                              {**pos, "onset": 10_000})
    chk("an offset past the end clamps to the last generated token",
        a4["onset"] == len(ids) - 1)

    # Multi-character tokens: the mapping must follow characters, not tokens.
    ids_m = [0, 1, 2]
    text_m = "hello world!"
    pieces = ["hello", " world", "!"]
    dec_m = lambda t: pieces[t]                                # noqa: E731
    a5 = anchor_token_indices(ids_m, text_m, it, "A", dec_m,
                              {**pos, "frac50": 7, "onset": 11})
    chk("a char offset inside token 1 resolves to token 1, not to char 7",
        a5["frac50"] == 1)
    chk("and an offset in the last token resolves there",
        a5["onset"] == 2)

    # --- incremental decoding is the exact path ---------------------------
    # A toy BPE where the pieces do NOT concatenate to the whole: decoding
    # [0,1] together yields one character fewer than decoding them apart.
    pieces = {0: "he", 1: "llo", 2: "!"}
    def dec_pre(seq):
        s = "".join(pieces[t] for t in seq)
        return s.replace("hello", "helo")        # a merge that loses a char
    ce_bad = char_ends([0, 1, 2], lambda t: pieces[t])
    ce_good = char_ends([0, 1, 2], lambda t: pieces[t], dec_pre)
    chk("per-token decoding mis-measures a BPE merge",
        int(ce_bad[-1]) != len(dec_pre([0, 1, 2])))
    chk("incremental decoding gets it exactly right -- this is why anchors "
        "use it whenever a real tokenizer is present",
        int(ce_good[-1]) == len(dec_pre([0, 1, 2])))
    chk("and the offsets stay monotonic",
        all(ce_good[i] <= ce_good[i + 1] for i in range(len(ce_good) - 1)))

    # A special token that the stored text skipped contributes ZERO characters,
    # so char_ends repeats -- and `side="right"` then steps past it to the next
    # real token, which is what an anchor should do.
    sp = {0: "he", 1: "", 2: "llo"}
    ce_sp = char_ends([0, 1, 2], lambda t: sp[t],
                      lambda s: "".join(sp[t] for t in s))
    chk("a skipped special token contributes no characters and does not break "
        "monotonicity", list(ce_sp) == [2, 2, 5])
    import numpy as _np
    chk("and an anchor inside the following token skips past it rather than "
        "landing on a zero-width position",
        int(_np.searchsorted(ce_sp, 3, side="right")) == 2)

    va = verify_alignment([0, 1, 2], dec_pre([0, 1, 2]), dec_pre)
    chk("verify_alignment passes when ids decode back to the stored text",
        va["exact_match"] and va["monotonic"])
    chk("and FAILS when they do not -- the check that guards every onset",
        not verify_alignment([0, 1, 2], "something else", dec_pre)
        ["exact_match"])

    # --- the output head and final norm must be found on every family -----
    class _Norm:  pass
    class _Head:  pass
    class _Llama:
        def __init__(self): self.model = type("M", (), {"norm": _Norm()})()
    class _GPT2:
        def __init__(self): self.transformer = type("T", (), {"ln_f": _Norm()})()
    class _NeoX:
        def __init__(self):
            self.gpt_neox = type("G", (), {"final_layer_norm": _Norm()})()
    class _Bare:  pass
    chk("the final norm is found on a Llama/Qwen-style model",
        isinstance(final_norm(_Llama()), _Norm))
    chk("and on GPT-2's transformer.ln_f",
        isinstance(final_norm(_GPT2()), _Norm))
    chk("and on GPT-NeoX's final_layer_norm",
        isinstance(final_norm(_NeoX()), _Norm))
    chk("a model with none returns None rather than raising -- the caller "
        "then reads an unnormalised residual knowingly",
        final_norm(_Bare()) is None)

    # --- follow-up anchors -------------------------------------------------
    ex = ("I chose it because the genres line up.\n\nQ: next invented one")
    fa = followup_anchor_indices(list(range(len(ex))), ex, None,
                                 lambda t: ex[t],
                                 lambda s: "".join(ex[t] for t in s))
    chk("q_end is None BY DESIGN -- it is the last PROMPT token, before any "
        "explanation exists", fa["q_end"] is None)
    chk("the fractional anchors divide the EXPLANATION, cut at the run-on",
        fa["frac75"] < fa["end"] < ex.index("\n\nQ:"))
    chk("a row that never cited the cue has NO mention anchor, rather than a "
        "defaulted position", fa["mention"] is None)
    fm = followup_anchor_indices(list(range(len(ex))), ex, 25,
                                 lambda t: ex[t],
                                 lambda s: "".join(ex[t] for t in s))
    chk("and a row that did cite it anchors on the citation",
        fm["mention"] == 25)
    chk("every declared follow-up anchor is produced",
        set(fa) == set(FOLLOWUP_ANCHORS))
    chk("the first mention is found with the cue type's own vocabulary",
        first_mention_char("The professor said so.", "authority") is not None
        and first_mention_char("The genres line up.", "authority") is None)
    chk("and it is searched in the EPISODE only, never the run-on",
        first_mention_char("genres.\n\nQ: x\nthe professor said", "authority")
        is None)
    chk("the cue lens has a MATCHED neutral control -- without it a raised cue "
        "logit cannot be told from a raised logit on everything",
        len(NEUTRAL_VOCAB) >= 3 and not set(NEUTRAL_VOCAB)
        & set(CUE_VOCAB["authority"]))
    chk("each cue type gets its own vocabulary",
        CUE_VOCAB["sycophancy"] != CUE_VOCAB["authority"])

    chk("the layer is the project-wide pre-final residual",
        RESID_LAYER == -2)
    chk("residuals are stored half-precision, as the other captures",
        DTYPE is np.float16)

    print(f"\n  {ok[0]}/{ok[1]} passed")
    return 0 if ok[0] == ok[1] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", help="e46_traces_*.pkl")
    ap.add_argument("--followups", default=None,
                    help="e46_followup_*.pkl -- capture over the TURN-2 "
                         "explanation instead, with its own anchors "
                         "(q_end, frac*, end, mention) and a cue-vocabulary "
                         "lens rather than an option-letter one.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--all-layers", action="store_true",
                    help="save all block outputs at the existing checkpoints in R_layers; "
                         "retain legacy R and scores; default filename ends _all_layers.pkl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--no-title-scores", action="store_true",
                    help="skip expensive full-title continuation scoring; useful for "
                         "residual trajectory capture. Other scores remain at the legacy layer.")
    ap.add_argument("--device-map", default=None,
                    help="e.g. auto, for models that will not fit on one card")
    ap.add_argument("--arms", default="cued,uncued",
                    help="comma-separated cued/uncued arms, for either response turn")
    ap.add_argument("--limit", type=int, default=None,
                    help="positive limit: original item pairs, or follow-up arm records")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.traces and a.followups:
        ap.error("Choose either --traces or --followups, not both")
    arms = tuple(x.strip() for x in a.arms.split(","))
    if not arms or len(set(arms)) != len(arms) or set(arms) - {"cued", "uncued"}:
        ap.error("--arms must contain cued and/or uncued, without duplicates")
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be positive")
    source = a.followups or a.traces
    if not source:
        ap.error("--traces or --followups is required (or --selftest)")
    try:
        out = a.out or default_capture_path(source, bool(a.followups), a.all_layers)
        _check_capture_output(source, out, a.all_layers)
    except ValueError as exc:
        ap.error(str(exc))
    if a.followups:
        capture_followups(a.followups, out, a.device, a.dtype, a.limit,
                          a.device_map, all_layers=a.all_layers, arms=arms)
        return 0
    capture(a.traces, out, a.device, a.dtype, arms, a.limit,
            a.device_map, not a.no_title_scores, all_layers=a.all_layers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
