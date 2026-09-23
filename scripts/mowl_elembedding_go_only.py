#!/usr/bin/env python

import click as ck

from mowl_common import (
    init_mowl,
    set_seed,
    normalize_device,
    build_model_compat,
    train_without_validation,
    get_embedding_dict,
    save_embedding_dict,
)


@ck.command()
@ck.option("--ontology-file", "-i", required=True, type=ck.Path(exists=True),
           help="Input OWL ontology file, e.g. go.owl. Do NOT pass normalized txt.")
@ck.option("--out-classes-file", "-ocf", default="go_elem_classes.pkl",
           help="Output pickle file for class embeddings.")
@ck.option("--out-relations-file", "-orf", default="go_elem_relations.pkl",
           help="Output pickle file for object property embeddings.")
@ck.option("--embedding-size", "-es", default=50, type=int,
           help="Embedding dimension for class centers / relation vectors.")
@ck.option("--batch-size", "-bs", default=32768, type=int,
           help="Batch size.")
@ck.option("--epochs", "-e", default=1000, type=int,
           help="Training epochs.")
@ck.option("--learning-rate", "-lr", default=1e-3, type=float,
           help="Learning rate.")
@ck.option("--margin", "-m", default=0.0, type=float,
           help="ELEmbeddings margin. mOWL default is often 0.")
@ck.option("--reg-norm", "-rn", default=1.0, type=float,
           help="ELEmbeddings norm regularization target.")
@ck.option("--device", "-d", default="auto",
           help="PyTorch device: auto, cpu, cuda, cuda:0, etc.")
@ck.option("--jvm-memory", default="16g",
           help="JVM memory for OWLAPI/mOWL, e.g. 8g, 16g, 32g.")
@ck.option("--model-filepath", default=None,
           help="Optional mOWL model checkpoint path.")
@ck.option("--seed", default=100, type=int,
           help="Random seed.")
@ck.option("--shorten-iris/--keep-full-iris", default=True,
           help="Save GO_0008150 instead of full IRI when possible.")
@ck.option("--go-only/--all-classes", default=False,
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

    print("Extracting embeddings")
    class_embeddings = get_embedding_dict(model, "class_embeddings")
    relation_embeddings = get_embedding_dict(model, "object_property_embeddings")

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


if __name__ == "__main__":
    main()