#!/usr/bin/env python

import os
import re
import random
import numpy as np
import pandas as pd
import torch


def init_mowl(jvm_memory="16g"):
    """
    Start JVM and return PathDataset class.

    Important:
    Import PathDataset after mowl.init_jvm(), because mOWL binds Java/OWLAPI
    through JPype.
    """
    import mowl
    mowl.init_jvm(jvm_memory)

    try:
        from mowl.datasets import PathDataset
    except Exception:
        from mowl.datasets.base import PathDataset

    return PathDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_model_compat(model_cls, **kwargs):
    """
    mOWL APIs changed slightly across versions. This helper tries a few
    constructor variants.

    For example:
    - ELEmbeddings usually accepts reg_norm.
    - ELBoxEmbeddings / ELBE may or may not accept reg_norm depending on version.
    """
    candidates = []

    # Full argument set.
    candidates.append(dict(kwargs))

    # Remove reg_norm.
    kw = dict(kwargs)
    kw.pop("reg_norm", None)
    candidates.append(kw)

    # Remove margin too.
    kw = dict(kwargs)
    kw.pop("reg_norm", None)
    kw.pop("margin", None)
    candidates.append(kw)

    # Minimal commonly accepted arguments.
    minimal_keys = [
        "dataset",
        "embed_dim",
        "learning_rate",
        "batch_size",
        "model_filepath",
        "device",
    ]
    candidates.append({k: kwargs[k] for k in minimal_keys if k in kwargs})

    last_error = None

    for kw in candidates:
        kw = {k: v for k, v in kw.items() if v is not None}
        try:
            return model_cls(**kw)
        except TypeError as e:
            last_error = e

    raise last_error


def train_without_validation(model, epochs):
    """
    Train without validation/test.

    In recent mOWL versions, train() can accept validate_every.
    Setting validate_every > epochs avoids validation during training.
    If the installed version does not expose validate_every, fall back to train(epochs=...).

    Also: do not set evaluator.
    """
    try:
        model.train(epochs=epochs, validate_every=epochs + 1)
    except TypeError:
        model.train(epochs=epochs)


def get_embedding_dict(model, attr_name):
    """
    Extract model.class_embeddings or model.object_property_embeddings.

    mOWL normally exposes these as dictionaries:
        entity_name -> numpy vector
    """
    if not hasattr(model, attr_name):
        raise AttributeError(f"Model has no attribute: {attr_name}")

    value = getattr(model, attr_name)

    if callable(value):
        value = value()

    if value is None:
        raise RuntimeError(f"{attr_name} is None. Was the model trained?")

    if not isinstance(value, dict):
        raise TypeError(
            f"{attr_name} should be a dict, got {type(value)}. "
            "Check the installed mOWL version."
        )

    return value


def to_numpy_vector(x):
    """
    Convert embedding object to a flat numpy vector.

    For box embeddings, some implementations may expose a tuple/list of tensors.
    In that case we concatenate flattened parts.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32).reshape(-1)

    if isinstance(x, np.ndarray):
        return x.astype(np.float32).reshape(-1)

    if isinstance(x, (list, tuple)):
        parts = []
        for item in x:
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            else:
                item = np.asarray(item)
            parts.append(item.astype(np.float32).reshape(-1))
        return np.concatenate(parts, axis=0)

    return np.asarray(x, dtype=np.float32).reshape(-1)


def simplify_entity_name(x):
    """
    Convert OWL entity string to a compact ID when possible.

    Examples:
        <http://purl.obolibrary.org/obo/GO_0008150> -> GO_0008150
        http://purl.obolibrary.org/obo/RO_0002211 -> RO_0002211
    """
    s = str(x).strip()

    # Common OWLAPI-style strings can contain <...>.
    s = s.strip("<>").strip()

    # Extract OBO-style identifiers.
    m = re.search(r"([A-Za-z]+_[0-9]+)", s)
    if m:
        return m.group(1)

    # owl:Thing / owl:Nothing etc.
    if "owl#Thing" in s or s.endswith("owl:Thing"):
        return "owl:Thing"
    if "owl#Nothing" in s or s.endswith("owl:Nothing"):
        return "owl:Nothing"

    return s


def should_keep_class(name, go_only=False):
    if not go_only:
        return True
    return "GO_" in name


def save_embedding_dict(
    emb_dict,
    out_file,
    key_col,
    shorten_iris=True,
    go_only=False,
    elem_embed_dim=None,
):
    """
    Save embeddings into pandas pkl.

    Output columns:
        key_col      e.g. classes / relations
        embeddings   flat numpy vector

    For ELEmbeddings, if vector length == elem_embed_dim + 1, also save:
        center
        radius
    """
    rows = []

    for entity, vec in emb_dict.items():
        name = simplify_entity_name(entity) if shorten_iris else str(entity)

        if key_col == "classes" and not should_keep_class(name, go_only=go_only):
            continue

        arr = to_numpy_vector(vec)

        row = {
            key_col: name,
            "embeddings": arr,
        }

        # ELEmbeddings often uses center + radius.
        if elem_embed_dim is not None and arr.shape[0] == elem_embed_dim + 1:
            row["center"] = arr[:-1]
            row["radius"] = float(abs(arr[-1]))

        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    df.to_pickle(out_file)

    print(f"Saved {len(df)} rows -> {out_file}")