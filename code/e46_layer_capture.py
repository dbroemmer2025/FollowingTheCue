"""E46 checkpoint-only layer capture, with an unchanged legacy readout site."""
from __future__ import annotations

import numpy as np


def transformer_blocks(model):
    # Same architecture paths as E4; keep this helper independent of its runner.
    for path in ("model.layers", "transformer.h", "gpt_neox.layers",
                 "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if len(obj):
            return obj
    raise ValueError("All-layer capture cannot locate transformer blocks on this architecture")


def layer_metadata(model, norm):
    import torch
    import transformers

    blocks = transformer_blocks(model)
    if norm is None:
        raise ValueError("All-layer capture requires an identifiable final normalization module")
    n = len(blocks)
    sites = [{"axis_index": 0, "name": "block_input", "block_number": 0,
              "stage": "before_first_block"}]
    sites += [{"axis_index": i, "name": f"block_{i}_output", "block_number": i,
               "stage": "after_block_before_final_norm"} for i in range(1, n + 1)]
    sites.append({"axis_index": n + 1, "name": "final_norm_output", "block_number": n,
                  "stage": "after_final_norm"})
    return {"layer_capture_version": "e46_checkpoints_all_layers_v1",
            "capture_torch_version": str(torch.__version__),
            "capture_transformers_version": transformers.__version__,
            "layer_capture": "all", "n_layers": n, "layer_sites": sites,
            "R_layers_axes": ["row", "layer_site", "hidden_feature"],
            "legacy_R_layer_axis": n - 1,
            "layer_capture_note": "Only existing checkpoint positions are saved; rows are not multiplied by layers."}


def checkpoint_states(model, full_ids, positions, *, all_layers=False, norm=None):
    """One replay -> legacy native-dtype vectors and optional float16 CPU layers.

    positions preserves row order, including distinct anchors on the same token.
    Hooks are removed before any subsequent title-scoring forward pass.
    """
    import torch

    if model.training:
        raise ValueError("Checkpoint replay requires model.eval()")
    if full_ids.ndim != 2 or full_ids.shape[0] != 1:
        raise ValueError("Checkpoint capture expects one sequence per replay")
    if not positions or any(p < 0 or p >= full_ids.shape[1] for p in positions):
        raise ValueError("Checkpoint positions must be nonempty and within the replay sequence")

    def selected(hidden):
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
        if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("Unexpected transformer activation shape at a capture site")
        index = torch.tensor(positions, device=hidden.device, dtype=torch.long)
        return hidden[0].index_select(0, index).detach()

    if not all_layers:
        with torch.inference_mode():
            out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                        output_hidden_states=True, use_cache=False)
        return selected(out.hidden_states[-2]), None

    info = layer_metadata(model, norm)
    blocks = transformer_blocks(model)
    buffers = [None] * len(info["layer_sites"])
    legacy = None
    handles = []

    def save(site, hidden):
        nonlocal legacy
        if buffers[site] is not None:
            raise ValueError(f"Capture site {site} executed twice during one replay")
        values = selected(hidden)
        if site == info["legacy_R_layer_axis"]:
            legacy = values.clone()  # Native precision for the existing lens scores.
        buffers[site] = values.to(device="cpu", dtype=torch.float16).numpy().copy()

    def first_input(module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        save(0, hidden)

    try:
        handles.append(blocks[0].register_forward_pre_hook(first_input, with_kwargs=True))
        for i, block in enumerate(blocks, 1):
            handles.append(block.register_forward_hook(
                lambda module, args, output, site=i: save(site, output)))
        handles.append(norm.register_forward_hook(
            lambda module, args, output: save(len(blocks) + 1, output)))
        with torch.inference_mode():
            model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                  output_hidden_states=False, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = [i for i, value in enumerate(buffers) if value is None]
    if missing or legacy is None:
        raise ValueError(f"Missing layer capture sites: {missing}")
    if len({value.shape for value in buffers}) != 1:
        raise ValueError("Layer widths differ; cannot form a shared residual trajectory")
    return legacy, np.stack(buffers, axis=1)


def resolved_checkpoints(anchors, names, prompt_length, prompt_anchor):
    """Keep the established anchor schedule; omit unresolved generated positions."""
    if prompt_length < 1:
        raise ValueError("Cannot anchor a replay with an empty prompt")
    return [(name, anchors.get(name), prompt_length - 1 if name == prompt_anchor
             else prompt_length + anchors[name])
            for name in names if name == prompt_anchor or anchors.get(name) is not None]


def add_layer_payload(payload, batches, metadata, hidden_size):
    """Keep one metadata row per checkpoint; never flatten the layer axis."""
    if batches is None:
        payload["metadata"]["layer_capture"] = "single"
        return
    n_sites = len(metadata["layer_sites"])
    layers = (np.concatenate(batches, axis=0) if batches else
              np.empty((0, n_sites, hidden_size), dtype=np.float16))
    if layers.shape[0] != len(payload["rows"]):
        raise ValueError("Layer vectors do not match checkpoint row metadata")
    if not np.array_equal(payload["R"], layers[:, metadata["legacy_R_layer_axis"], :]):
        raise ValueError("Legacy R does not match its all-layer slice")
    payload["R_layers"] = layers
    payload["metadata"].update(metadata)
    payload["metadata"]["lens_layer_axis"] = metadata["legacy_R_layer_axis"]
