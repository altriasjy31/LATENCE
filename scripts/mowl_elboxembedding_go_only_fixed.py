#!/usr/bin/env python
"""
Train an mOWL ELBE / ELBoxEmbeddings model on a GO OWL ontology and export
complete class-box representations.

Important
---------
mOWL's generic ``model.class_embeddings`` property exposes only
``model.module.class_embed``.  ELBE classes are boxes and require both:

    center = model.module.class_embed
    offset = abs(model.module.class_offset)

This script exports each class embedding as the concatenated vector:

    [center_1, ..., center_d, offset_1, ..., offset_d]

Therefore, for ``--embedding-size 512``, the saved ``embeddings`` vectors have
width 1024 and are compatible with:

    evaluate_go_embedding.py --geometry box --layout center_offset \
        --embedding-dim 512
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


def import_elbox_model():
    """Import the EL box model across mOWL versions."""
    try:
        from mowl.models import ELBE

        return ELBE, "ELBE"
    except Exception:
        from mowl.models import ELBoxEmbeddings

        return ELBoxEmbeddings, "ELBoxEmbeddings"


def _as_numpy_weight(layer, name: str) -> np.ndarray:
    """Return a torch embedding/module weight as a 2-D float32 array."""
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
    """Yield (entity_name, row_index) in model row order."""
    items = [(str(name), int(index)) for name, index in index_dict.items()]
    items.sort(key=lambda item: item[1])
    return items


def extract_elbox_class_embeddings(model, expected_dim: int) -> Dict[str, np.ndarray]:
    """
    Extract complete ELBE class boxes as concatenated center/offset vectors.

    Returns
    -------
    dict
        Mapping from ontology class IRI to a vector of width ``2 * expected_dim``.
    """
    module = getattr(model, "module", None)
    if module is None:
        raise AttributeError("The mOWL model has no initialized '.module'.")

    centers = _as_numpy_weight(getattr(module, "class_embed", None), "class_embed")
    offsets = _as_numpy_weight(getattr(module, "class_offset", None), "class_offset")
    offsets = np.abs(offsets)

    if centers.shape != offsets.shape:
        raise ValueError(
            "ELBE center and offset tensors have different shapes: "
            f"center={centers.shape}, offset={offsets.shape}."
        )
    if centers.shape[1] != expected_dim:
        raise ValueError(
            f"Requested embedding size is {expected_dim}, but the trained ELBE "
            f"center tensor has width {centers.shape[1]}."
        )

    index_dict = getattr(model, "class_index_dict", None)
    if index_dict is None:
        raise AttributeError("The mOWL model does not expose 'class_index_dict'.")
    if len(index_dict) != centers.shape[0]:
        raise ValueError(
            "Class index size does not match ELBE tensors: "
            f"index={len(index_dict)}, tensors={centers.shape[0]}."
        )

    result: Dict[str, np.ndarray] = {}
    for name, row in _ordered_index_items(index_dict):
        if row < 0 or row >= centers.shape[0]:
            raise IndexError(f"Class '{name}' has invalid row index {row}.")
        result[name] = np.concatenate((centers[row], offsets[row]), axis=0)

    return result


@ck.command()
@ck.option(
    "--ontology-file",
    "-i",
    required=True,
    type=ck.Path(exists=True),
    help="Input OWL ontology file, e.g. go.owl. Do NOT pass normalized txt.",
)
@ck.option(
    "--out-classes-file",
    "-ocf",
    default="go_elbox_classes.pkl",
    show_default=True,
    help="Output pickle file for complete class box embeddings.",
)
@ck.option(
    "--out-relations-file",
    "-orf",
    default="go_elbox_relations.pkl",
    show_default=True,
    help="Output pickle file for object property embeddings.",
)
@ck.option("--embedding-size", "-es", default=50, type=ck.IntRange(min=1), show_default=True,
           help="Dimension d of both the box center and box offset.")
@ck.option("--batch-size", "-bs", default=32768, type=ck.IntRange(min=1), show_default=True)
@ck.option("--epochs", "-e", default=1000, type=ck.IntRange(min=0), show_default=True)
@ck.option("--learning-rate", "-lr", default=1e-3, type=ck.FloatRange(min=0.0, min_open=True),
           show_default=True)
@ck.option("--margin", "-m", default=0.0, type=float, show_default=True,
           help="Margin accepted by the installed ELBE/ELBox implementation.")
@ck.option("--reg-norm", "-rn", default=1.0, type=float, show_default=True,
           help="Regularization norm if accepted by the installed mOWL version.")
@ck.option("--device", "-d", default="auto", show_default=True,
           help="PyTorch device: auto, cpu, cuda, cuda:0, etc.")
@ck.option("--jvm-memory", default="16g", show_default=True,
           help="JVM memory for OWLAPI/mOWL, e.g. 8g, 16g, 32g.")
@ck.option("--model-filepath", default=None,
           help="Optional mOWL model checkpoint path.")
@ck.option("--seed", default=100, type=int, show_default=True)
@ck.option("--shorten-iris/--keep-full-iris", default=True, show_default=True,
           help="Save GO_0008150 instead of full IRI when possible.")
@ck.option("--go-only/--all-classes", default=False, show_default=True,
           help="If enabled, save only GO_* class embeddings.")
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
    ELBoxModel, model_name = import_elbox_model()
    device = normalize_device(device)

    print("Loading ontology:", ontology_file)
    dataset = PathDataset(ontology_file)

    print(f"Building {model_name} model")
    model = build_model_compat(
        ELBoxModel,
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

    print("Extracting complete ELBE boxes: class_embed + abs(class_offset)")
    class_embeddings = extract_elbox_class_embeddings(model, embedding_size)
    relation_embeddings = get_embedding_dict(model, "object_property_embeddings")

    first_vector = next(iter(class_embeddings.values()), None)
    if first_vector is None:
        raise RuntimeError("No class box embeddings were extracted.")
    print(
        "Class box export:",
        f"classes={len(class_embeddings)},",
        f"center_dim={embedding_size},",
        f"offset_dim={embedding_size},",
        f"saved_width={first_vector.shape[0]}",
    )

    save_embedding_dict(
        class_embeddings,
        out_classes_file,
        key_col="classes",
        shorten_iris=shorten_iris,
        go_only=go_only,
        elem_embed_dim=None,
    )

    save_embedding_dict(
        relation_embeddings,
        out_relations_file,
        key_col="relations",
        shorten_iris=shorten_iris,
        go_only=False,
        elem_embed_dim=None,
    )

    print("Saved complete ELBE class boxes:", out_classes_file)
    print("Evaluate with:")
    print(
        "  python evaluate_go_embedding.py "
        f"--embedding-file {out_classes_file} --go-file <go.obo> "
        "--geometry box --layout center_offset "
        f"--embedding-dim {embedding_size} --edge-types is_a --output-dir <output_dir>"
    )


if __name__ == "__main__":
    main()
