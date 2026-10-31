#!/usr/bin/env python3
"""E46 text-channel ceiling: stronger text readers, with a capacity gate.

WHY THIS EXISTS, AND WHAT IT IS NOT

The frozen E46 forecast compares a state reader (PCA + logistic regression on
the q_end residual) against a text reader (TF-IDF + logistic regression on the
follow-up prompt). On the qwen_fresh_01 run, as saved, the text reader scores
AUC 0.488 and balanced accuracy 0.500 -- exactly chance. A gap measured against
a reader that did not fit is not a verbalization gap; it is a measurement of
that reader's failure. This is the reader-capacity floor the paper already
names.

That saved figure is environment-specific, which is its own finding. The
forecast was produced under sklearn 1.6.1 / numpy 2.1.3 / scipy 1.16.3. Under
sklearn 1.7.2 / numpy 2.2.6 / scipy 1.15.3, with byte-identical inputs and
byte-identical analysis code, the same deterministic pipeline returns text AUC
0.610 and a state-vs-text gap of 0.060 bits rather than 0.171. Neither stack is
privileged. A headline that moves by a factor of three across a minor version
bump is fragile regardless of which number is "right" -- so this script refuses
to mix stacks (see `environment_check`) rather than quietly producing a fourth
incomparable number.

The obvious fix -- "use the E6 order-aware Transformer, the paper's strongest
text-reader family" -- does not follow. E6's reader was fitted on hundreds of
examples per cell. Here n=96 with 22 positives and ~2900-character documents
whose variation is dominated by the rotating demonstrations, which carry no
signal by construction. A 2-layer Transformer on 96 examples will land at
chance too, and it will do so while *looking* more rigorous. A
stronger-in-principle reader that underfits INFLATES the apparent gap. That is
a worse failure than the current one, because it is harder to see.

So this script does three things, in this order:

  1. Runs a LADDER of text readers, not one: the frozen TF-IDF rungs, trimmed
     variants that strip known-nuisance text, an LSA rung that gives the text
     channel the same dimensionality reduction the state channel gets, and --
     when torch and an embedding matrix are supplied -- an E6-style order-aware
     Transformer over the embedding sequence, with the subsumption head that
     guarantees it cannot score below its own pooled features.

  2. Reports the gap against the CEILING of that ladder, never against one
     rung. Picking a single text reader and calling it "the" text baseline is
     how a capacity artifact becomes a headline.

  3. Gates the whole thing on a PLANTED-SIGNAL CAPACITY CONTROL. A token
     correlated with the target at a known strength is spliced into the text;
     if no rung in the ladder can recover it at this n, the text channel cannot
     support an absence claim and the script reports
     `text_channel_underpowered` instead of a gap. The control runs FIRST and
     its verdict is recorded whatever the gap turns out to be.

This forecasts later cue citation from the pre-explanation residual. It does
not measure suppression, causal self-knowledge, or the completed-trace
verbalization gap. Read-only with respect to every frozen artifact: it writes
to its own prefix and never touches the primary forecast.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from e46_readers import VERSION, loss_rows, matrix_reader, nested_oof

PINNED = ("numpy", "scikit-learn", "scipy")

CONTROL_RATE = 0.85     # P(marker present | y=1); absent when y=0 at 1-rate
CONTROL_MIN_AUC = 0.70  # a rung must beat this on planted data to count as able
SEQ_KEEP = 256          # trailing tokens fed to the order-aware rung


# ---------------------------------------------------------------- text views

def episode_only(row):
    """The model's own turn-1 episode: where a cue mention would have to live."""
    return row["episode1"] or ""


def question_only(row):
    """The posed question and options, without demonstrations."""
    prompt = row["prompt1"]
    return prompt.rsplit("\nQ:", 1)[-1] if "\nQ:" in prompt else prompt


def full_followup(row):
    return row["prompt2"]


def episode_plus_question(row):
    return question_only(row) + "\n" + episode_only(row)


TEXT_VIEWS = {
    "full": full_followup,
    "episode": episode_only,
    "question": question_only,
    "question_episode": episode_plus_question,
}


# ------------------------------------------------------------ reader ladder

def lr(c):
    return LogisticRegression(C=c, max_iter=1000, random_state=46)


def tfidf(kind, c):
    return make_pipeline(
        TfidfVectorizer(analyzer="char" if kind == "char" else "word",
                        ngram_range=(3, 5) if kind == "char" else (1, 2),
                        max_features=20000),
        lr(c))


def lsa(kind, dim, c):
    """The text channel's analogue of the state channel's PCA.

    The state reader is handed 8 or 16 dense components. The frozen text reader
    is handed 20000 sparse TF-IDF features for 96 rows. Part of the asymmetry
    in the frozen comparison is dimensionality, not modality, so the ladder
    includes a rung that closes it.
    """
    return make_pipeline(
        TfidfVectorizer(analyzer="char" if kind == "char" else "word",
                        ngram_range=(3, 5) if kind == "char" else (1, 2),
                        max_features=20000),
        TruncatedSVD(n_components=dim, random_state=46),
        lr(c))


def text_candidates():
    """(name, factory) pairs for one text view. Mirrors the frozen grid, plus LSA."""
    out = [(f"tfidf_{kind}_C{c}", lambda kind=kind, c=c: tfidf(kind, c))
           for kind in ("word", "char") for c in (0.1, 1.0, 10.0)]
    out += [(f"lsa_{kind}{d}_C{c}", lambda kind=kind, d=d, c=c: lsa(kind, d, c))
            for kind in ("word",) for d in (8, 16, 32) for c in (0.1, 1.0, 10.0)]
    return out


# ------------------------------------------------- E6-style order-aware rung

class OrderAwareTextReader:
    """Transformer encoder over the ordered embedding sequence, E6-style.

    Ported in spirit from E3_order_aware_decoder.OrderAwareTransformer: sinusoidal
    positions, attention pooling, and -- critically -- the SUBSUMPTION head. The
    mean-pooled sequence is concatenated into the readout, so the order-aware
    reader can in the worst case ignore order and reproduce its own pooled
    baseline. Without that, a negative order effect is uninterpretable: the two
    readers would simply be solving different problems.

    Scaled down hard from E6's settings (d_model 128 -> 32, 2 layers -> 1),
    because 96 rows cannot fit E6's reader. That scaling is itself an admission:
    see the capacity control, which is what decides whether any of this counts.
    """

    def __init__(self, dim=32, layers=1, heads=2, epochs=40, lr_=1e-3,
                 wd=1e-2, dropout=0.2, seed=46):
        self.dim, self.layers, self.heads = dim, layers, heads
        self.epochs, self.lr_, self.wd, self.dropout, self.seed = epochs, lr_, wd, dropout, seed

    def _build(self, d_in):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self, d_in, d, layers, heads, dropout):
                super().__init__()
                self.proj = nn.Linear(d_in, d)
                enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout,
                                                 batch_first=True)
                self.enc = nn.TransformerEncoder(enc, layers)
                self.attn = nn.Linear(d, 1)
                # subsumption: pooled features enter the head alongside the
                # attention readout, so order can only help.
                self.head = nn.Linear(2 * d, 2)

            def forward(self, x, mask):
                h = self.enc(self.proj(x), src_key_padding_mask=~mask)
                w = self.attn(h).masked_fill(~mask.unsqueeze(-1), -1e9)
                w = torch.softmax(w, dim=1)
                attended = (w * h).sum(1)
                denom = mask.sum(1, keepdim=True).clamp(min=1)
                pooled = (h * mask.unsqueeze(-1)).sum(1) / denom
                return self.head(torch.cat([attended, pooled], dim=-1))

        torch.manual_seed(self.seed)
        return Net(d_in, self.dim, self.layers, self.heads, self.dropout)

    @staticmethod
    def _pack(seqs, torch):
        keep = min(SEQ_KEEP, max(len(s) for s in seqs))
        d = seqs[0].shape[1]
        x = np.zeros((len(seqs), keep, d), dtype=np.float32)
        m = np.zeros((len(seqs), keep), dtype=bool)
        for i, s in enumerate(seqs):
            s = s[-keep:]
            x[i, :len(s)] = s
            m[i, :len(s)] = True
        return torch.from_numpy(x), torch.from_numpy(m)

    def fit(self, X, y):
        import torch
        self.classes_ = np.unique(y)
        net = self._build(X[0].shape[1])
        xb, mb = self._pack(X, torch)
        yb = torch.from_numpy(np.asarray(y, dtype=np.int64))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr_, weight_decay=self.wd)
        # class weights: 22/96 positives, and an unweighted fit collapses to
        # the majority class long before it learns anything.
        counts = np.bincount(np.asarray(y), minlength=2).astype(float)
        w = torch.from_numpy((counts.sum() / np.maximum(counts, 1)).astype(np.float32))
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            lossf(net(xb, mb), yb).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        self.net_ = net
        return self

    def predict_proba(self, X):
        import torch
        self.net_.eval()
        xb, mb = self._pack(X, torch)
        with torch.no_grad():
            return torch.softmax(self.net_(xb, mb), dim=-1).numpy()


def embedding_sequences(texts, hf, embeddings, dim=32):
    """Tokenize, look up embeddings, project to `dim` with a label-free PCA.

    The PCA is fitted on the embedding MATRIX, not on this run's rows, so it
    cannot leak the target -- the same argument E6 makes for EMB_PCA_DIM.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf)
    emb = np.load(embeddings).astype(np.float32)
    centre = emb.mean(axis=0)
    _, _, vt = np.linalg.svd(emb[:20000] - centre, full_matrices=False)
    basis = vt[:dim].T.astype(np.float32)
    out = []
    for text in texts:
        ids = tok(text, add_special_tokens=False)["input_ids"][-SEQ_KEEP:]
        out.append((emb[ids] - centre) @ basis)
    return out


# ------------------------------------------------------------------ scoring

def runtime_versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": sys.version}
    for package in ("numpy", "scikit-learn", "scipy", "torch", "transformers",
                    "tokenizers"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    return out


def environment_check(reference):
    """Refuse to compare numbers produced under a different library stack.

    nested_oof is deterministic given code, inputs and seed -- verified. It is
    NOT deterministic across scikit-learn versions. On this run the frozen
    forecast was produced under sklearn 1.6.1 / numpy 2.1.3 and re-running it
    under 1.7.2 / 2.2.6 moved the text channel from AUC 0.488 to 0.610 and the
    state-vs-text gap from 0.171 to 0.060 bits.

    A ladder result computed here and compared against a ceiling computed there
    is not a comparison. So the mismatch is surfaced as a first-class field
    rather than a footnote, and `--allow-environment-drift` is required to
    proceed -- which marks the output as non-comparable.
    """
    current = runtime_versions()
    drift = {p: {"frozen": reference.get(p), "current": current.get(p)}
             for p in PINNED
             if reference.get(p) is not None and reference.get(p) != current.get(p)}
    return current, drift


def bits(P, y):
    return float(loss_rows(np.asarray(y, int), P).mean())


def evaluate_rung(X, y, groups, candidates, seed):
    fit = nested_oof(X, y, groups, 2, candidates, seed=seed)
    ce = bits(fit["P"], y)
    try:
        auc = float(roc_auc_score(y, fit["P"][:, 1]))
    except ValueError:
        auc = float("nan")
    return {"ce_bits": ce, "auc": auc,
            "selected": [d.get("candidate") for d in fit["details"]]}, fit


def capacity_control(views, y, groups, seed, rng):
    """Can ANY rung recover a signal we know is present in the text?

    Splices a marker token into each document with probability CONTROL_RATE
    when y=1 and 1-CONTROL_RATE when y=0. This is a deliberately easy,
    lexically obvious signal. A text family that cannot find it at this n
    cannot be used to argue that the text lacks the real signal.
    """
    present = np.where(np.asarray(y) == 1,
                       rng.random(len(y)) < CONTROL_RATE,
                       rng.random(len(y)) >= CONTROL_RATE)
    results = {}
    for name, docs in views.items():
        planted = [d + ("\nzqxmarker present." if p else "\nzqxmarker absent.")
                   for d, p in zip(docs, present)]
        score, _ = evaluate_rung(planted, y, groups, text_candidates(), seed)
        results[name] = score
    best = max(results.values(), key=lambda r: r["auc"])
    return {"planted_rate": CONTROL_RATE, "threshold_auc": CONTROL_MIN_AUC,
            "per_view": results, "best_auc": best["auc"],
            "text_channel_able": bool(best["auc"] >= CONTROL_MIN_AUC)}


# --------------------------------------------------------------------- main

def run(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from e46_followup_predict import collect

    selected, y, groups, X, meta = collect(a.furesid, a.followups)
    import pickle
    with open(a.followups, "rb") as f:
        rows = {(r["index"], r["arm"]): r for r in pickle.load(f)["rows"]}
    source = [rows[(r["index"], "cued")] for r in selected]

    views = {name: [fn(r) for r in source] for name, fn in TEXT_VIEWS.items()}
    rng = np.random.default_rng(a.seed)

    reference = {}
    receipt = Path(a.furesid).parent / "capture-followup_receipt.json"
    if receipt.exists():
        reference = json.loads(receipt.read_text()).get("runtime_versions", {})
    current, drift = environment_check(reference)
    if drift and not a.allow_environment_drift:
        raise ValueError(
            "Library versions differ from the sealed run: "
            + "; ".join(f"{p} {v['frozen']} -> {v['current']}" for p, v in drift.items())
            + ". Numbers computed here cannot be compared with the frozen "
              "forecast. Re-run in the recorded environment, or pass "
              "--allow-environment-drift to record an explicitly non-comparable result.")

    report = {"evaluation_version": VERSION, "analysis": "e46 text-channel ceiling",
              "runtime_versions": current, "frozen_runtime_versions": reference,
              "environment_drift": drift or None,
              "comparable_to_frozen_forecast": not drift,
              "scope": ("Forecasting later cue citation from the pre-explanation "
                        "residual. Not suppression, not causal self-knowledge, "
                        "not the completed-trace verbalization gap."),
              "n": int(len(y)), "n_positive": int(np.sum(y)),
              "n_source_groups": int(len(np.unique(groups))), "seed": a.seed,
              "design_id": meta.get("replication_design_id")}

    print(f"n={len(y)}  positives={int(y.sum())}  groups={len(np.unique(groups))}\n")

    print("== capacity control (planted signal) ==")
    control = capacity_control(views, y, groups, a.seed, rng)
    report["capacity_control"] = control
    for name, res in control["per_view"].items():
        print(f"   {name:18s} planted AUC {res['auc']:.3f}")
    print(f"   -> text channel able to learn at this n: {control['text_channel_able']}\n")

    print("== text ladder (real data) ==")
    ladder = {}
    for name, docs in views.items():
        score, _ = evaluate_rung(docs, y, groups, text_candidates(), a.seed)
        ladder[f"tfidf_lsa::{name}"] = score
        print(f"   {name:18s} ce={score['ce_bits']:.4f}  auc={score['auc']:.3f}")

    if a.embeddings:
        print("\n== order-aware rung (E6-style) ==")
        for name in a.order_aware_views:
            seqs = embedding_sequences(views[name], a.hf, a.embeddings)
            cands = [("order_aware", lambda: OrderAwareTextReader(seed=a.seed))]
            score, _ = evaluate_rung(seqs, y, groups, cands, a.seed)
            ladder[f"order_aware::{name}"] = score
            print(f"   {name:18s} ce={score['ce_bits']:.4f}  auc={score['auc']:.3f}")
    else:
        report["order_aware"] = "skipped: --embeddings not supplied"
        print("\n(order-aware rung skipped: pass --embeddings and --hf to enable)")

    report["text_ladder"] = ladder
    best_name = min(ladder, key=lambda k: ladder[k]["ce_bits"])
    report["text_ceiling"] = {"rung": best_name, **ladder[best_name]}

    state_score, _ = evaluate_rung(X["state"], y, groups,
                                   [(f"pca{d}_C{c}", lambda d=d, c=c:
                                     matrix_reader(lambda: lr(c), d))
                                    for d in (8, 16) for c in (0.01, 0.1, 1.0)],
                                   a.seed)
    report["state"] = state_score
    gap = report["text_ceiling"]["ce_bits"] - state_score["ce_bits"]

    if not control["text_channel_able"]:
        report["status"] = "text_channel_underpowered"
        report["gap_bits"] = None
        report["verdict"] = (
            "No text rung recovered a planted signal at this n. The text channel "
            "cannot support a gap claim: a state-minus-text difference here "
            "measures text-reader capacity, not information absent from the text.")
    else:
        report["status"] = "estimated"
        report["gap_bits"] = float(gap)
        report["verdict"] = (
            "Gap is state versus the CEILING of the text ladder. It remains a "
            "forecasting result on an exploratory run with discovery-item reuse."
            + ("" if not drift else
               " COMPUTED UNDER A DIFFERENT LIBRARY STACK than the frozen "
               "forecast; not comparable with it."))

    print(f"\ntext ceiling : {best_name}  ce={report['text_ceiling']['ce_bits']:.4f}")
    print(f"state        : ce={state_score['ce_bits']:.4f}  auc={state_score['auc']:.3f}")
    print(f"status       : {report['status']}")
    if report["gap_bits"] is not None:
        print(f"gap (bits)   : {report['gap_bits']:+.4f}")
    print(f"\n{report['verdict']}")

    out = Path(a.out_prefix + ".json")
    if out.exists():
        raise ValueError(f"Output exists: {out}; choose a new prefix")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"\nWrote {out}")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--furesid", required=True)
    p.add_argument("--followups", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--embeddings", help=".npy of the subject model's input embedding "
                                        "matrix; enables the order-aware rung")
    p.add_argument("--hf", help="repo id for the tokenizer (with --embeddings)")
    p.add_argument("--order-aware-views", nargs="+", default=["question_episode"],
                   choices=sorted(TEXT_VIEWS))
    p.add_argument("--seed", type=int, default=46)
    p.add_argument("--allow-environment-drift", action="store_true",
                   help="proceed despite a library-version mismatch with the "
                        "sealed run; the output is marked non-comparable")
    a = p.parse_args(argv)
    if bool(a.embeddings) != bool(a.hf):
        p.error("--embeddings and --hf must be given together")
    try:
        run(a)
    except (ValueError, FileNotFoundError) as exc:
        p.exit(2, f"E46 order-aware: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
