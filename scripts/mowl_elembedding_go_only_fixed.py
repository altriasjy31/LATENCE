#!/usr/bin/env python
"""
Train mOWL ELEmbeddings on a GO OWL ontology and export complete class balls.

mOWL's generic ``model.class_embeddings`` property exposes only class centers.
An ELEm class requires both ``class_embed`` and ``class_rad``.  This script
exports each class as:

    [center_1, ..., center_d, abs(radius)]

For ``--embedding-size 512``, each saved class vector therefore has width 513.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import click as ck
import numpy as np

from mowl_common import (
    init_mowl,
    set_seed,
    normalize_device,
    build_model_compat,
    train_without_validation,
    get_embedding_dict,
    save_embedding_dict,
)


def _as_numpy_weight(layer, name: str) -> np.ndarray:
    if layer is None:
        raise AttributeError(f"mOWL module does not expose '{name}'.")
    weight = getattr(layer, "weight", None)
    if weight is None:
        raise AttributeError(f"mOWL attribute '{name}' has no '.weight' tensor.")
    array = weight.detach().cpu().numpy().astype(np.float32, copy=False)
    if array.ndim != 2:
        raise ValueError(f"Expected {name}.weight to be 2-D, found shape {array.shape}.")
    return array


def _ordered_index_items(index_dict: Mapping) -> Iterable[Tuple[str, int]]:
    items = [(str(name), int(index)) for name, index in index_dict.items()]
    items.sort(key=lambda item: item[1])
    return items


def extract_elem_class_embeddings(model, expected_dim: int) -> Dict[str, np.ndarray]:
    module = getattr(model, "module", None)
    if module is None:
        raise AttributeError("The mOWL model has no initialized '.module'.")

    centers = _as_numpy_weight(getattr(module, "class_embed", None), "class_embed")
    radii = _as_numpy_weight(getattr(module, "class_rad", None), "class_rad")
    radii = np.abs(radii)

    if centers.shape[1] != expected_dim:
        raise ValueError(
            f"Requested embedding size is {expected_dim}, but class centers have width "
            f"{centers.shape[1]}."
        )
    if radii.shape[0] != centers.shape[0]:
        raise ValueError(
            f"ELEm center/radius row mismatch: center={centers.shape}, radius={radii.shape}."
        )
    if radii.shape[1] != 1:
        raise ValueError(
            f"Expected one radius per ELEm class, found class_rad shape {radii.shape}."
        )

    index_dict = getattr(model, "class_index_dict", None)
    if index_dict is None:
        raise AttributeError("The mOWL model does not expose 'class_index_dict'.")
    if len(index_dict) != centers.shape[0]:
        raise ValueError(
            "Class index size does not match ELEm tensors: "
            f"index={len(index_dict)}, tensors={centers.shape[0]}."
        )

    result: Dict[str, np.ndarray] = {}
    for name, row in _ordered_index_items(index_dict):
        result[name] = np.concatenate((centers[row], radii[row]), axis=0)
    return result


@ck.command()
@ck.option("--ontology-file", "-i", required=True, type=ck.Path(exists=True),
           help="Input OWL ontology file, e.g. go.owl. Do NOT pass normalized txt.")
@ck.option("--out-classes-file", "-ocf", default="go_elem_classes.pkl", show_default=True)
@ck.option("--out-relations-file", "-orf", default="go_elem_relations.pkl", show_default=True)
@ck.option("--embedding-size", "-es", default=50, type=ck.IntRange(min=1), show_default=True,
           help="Dimension d of class centers and relation vectors.")
@ck.option("--batch-size", "-bs", default=32768, type=ck.IntRange(min=1), show_default=True)
@ck.option("--epochs", "-e", default=1000, type=ck.IntRange(min=0), show_default=True)
@ck.option("--learning-rate", "-lr", default=1e-3, type=ck.FloatRange(min=0.0, min_open=True),
           show_default=True)
@ck.option("--margin", "-m", default=0.0, type=float, show_default=True)
@ck.option("--reg-norm", "-rn", default=1.0, type=float, show_default=True)
@ck.option("--device", "-d", default="auto", show_default=True)
@ck.option("--jvm-memory", default="16g", show_default=True)
@ck.option("--model-filepath", default=None)
@ck.option("--seed", default=100, type=int, show_default=True)
@ck.option("--shorten-iris/--keep-full-iris", default=True, show_default=True)
@ck.option("--go-only/--all-classes", default=False, show_default=True)
def main(
    ontology_file,
    out_classes_file,
    out_relations_file,
    embedding_size,
    batch_size,
    epochs,
    learning_rate,
    margin,
    reg_norm,
    device,
    jvm_memory,
    model_filepath,
    seed,
    shorten_iris,
    go_only,
):
    set_seed(seed)

    PathDataset = init_mowl(jvm_memory)
    from mowl.models import ELEmbeddings

    device = normalize_device(device)

    print("Loading ontology:", ontology_file)
    dataset = PathDataset(ontology_file)

    print("Building ELEmbeddings model")
    model = build_model_compat(
        ELEmbeddings,
        dataset=dataset,
        embed_dim=embedding_size,
        margin=margin,
        reg_norm=reg_norm,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_filepath=model_filepath,
        device=device,
    )

    print("Training without validation/test")
    train_without_validation(model, epochs)

    print("Extracting complete ELEm balls: class_embed + abs(class_rad)")
    class_embeddings = extract_elem_class_embeddings(model, embedding_size)
    relation_embeddings = get_embedding_dict(model, "object_property_embeddings")

    first_vector = next(iter(class_embeddings.values()), None)
    if first_vector is None:
        raise RuntimeError("No ELEm class embeddings were extracted.")
    print(
        "Class ball export:",
        f"classes={len(class_embeddings)},",
        f"center_dim={embedding_size},",
        "radius_dim=1,",
        f"saved_width={first_vector.shape[0]}",
    )

    save_embedding_dict(
        class_embeddings,
        out_classes_file,
        key_col="classes",
        shorten_iris=shorten_iris,
        go_only=go_only,
        elem_embed_dim=embedding_size,
    )
    save_embedding_dict(
        relation_embeddings,
        out_relations_file,
        key_col="relations",
        shorten_iris=shorten_iris,
        go_only=False,
        elem_embed_dim=None,
    )

    print("Saved complete ELEm class balls:", out_classes_file)


if __name__ == "__main__":
    main()
