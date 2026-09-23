#!/usr/bin/env python
"""
Ontology-only EL embedding for GO normalized axioms.

PyTorch 2.4+ rewrite of elembedding_go_only.py.

It trains class/relation embeddings from normalized EL axioms only, e.g.

    SubClassOf(<C> <D>)
    SubClassOf(<C> ObjectSomeValuesFrom(<R> <D>))
    SubClassOf(ObjectSomeValuesFrom(<R> <C>) <D>)
    SubClassOf(ObjectIntersectionOf(<C> <D>) <E>)

It also accepts simplified normalized forms:

    C SubClassOf D
    C SubClassOf R some D
    R some C SubClassOf D
    C and D SubClassOf E

Outputs:
    classes.pkl      columns: classes, embeddings
    relations.pkl    columns: relations, embeddings

Note:
    class embeddings have embedding_size + 1 dimensions.
    The last dimension is interpreted as class radius.

    center = embeddings[:, :-1]
    radius = abs(embeddings[:, -1])
"""

import csv
import math
import os
import random
import re
import time
import logging

import click as ck
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


logging.basicConfig(level=logging.INFO)


@ck.command()
@ck.option("--data-file", "-df", required=True,
           help="Normalized ontology file generated from GO OWL.")
@ck.option("--out-classes-file", "-ocf", default="go_cls_embeddings.pkl",
           help="Output pickle file for class/GO embeddings.")
@ck.option("--out-relations-file", "-orf", default="go_rel_embeddings.pkl",
           help="Output pickle file for relation embeddings.")
@ck.option("--batch-size", "-bs", default=512, type=int,
           help="Batch size.")
@ck.option("--epochs", "-e", default=1000, type=int,
           help="Training epochs.")
@ck.option("--device", "-d", default="auto",
           help="PyTorch device: auto, cuda, cuda:0, cpu. Also accepts TF-style gpu:0 or cpu:0.")
@ck.option("--embedding-size", "-es", default=50, type=int,
           help="Embedding center size. Class embedding size is embedding_size + 1 because of radius.")
@ck.option("--reg-norm", "-rn", default=1.0, type=float,
           help="Norm regularization target for embedding centers.")
@ck.option("--margin", "-m", default=-0.1, type=float,
           help="EL loss margin. Original elembedding.py often used -0.1.")
@ck.option("--learning-rate", "-lr", default=0.01, type=float,
           help="Learning rate.")
@ck.option("--loss-history-file", "-lhf", default="go_el_loss_history.csv",
           help="CSV file for loss history.")
@ck.option("--save-every", default=50, type=int,
           help="Save embeddings every N epochs. Use 0 to save only final embeddings.")
@ck.option("--use-negatives/--no-negatives", default=False,
           help="Whether to add corrupted negative nf3 triples from ontology relations. Default: disabled.")
@ck.option("--seed", default=100, type=int,
           help="Random seed.")
@ck.option("--clip-norm", default=1.0, type=float,
           help="Gradient clipping max norm. Use 0 or negative value to disable.")
@ck.option("--empty-mode", type=ck.Choice(["tf", "skip"]), default="tf",
           help="How to handle empty axiom groups. 'tf' reproduces the old zero-batch behavior; 'skip' ignores empty groups.")
@ck.option("--add-center-radius/--no-add-center-radius", default=False,
           help="If enabled, also save center and radius columns for class embeddings.")
@ck.option("--save-state-file", default=None,
           help="Optional torch checkpoint path.")
@ck.option("--log-every", default=1, type=int,
           help="Print loss every N epochs.")
@ck.option("--deterministic/--no-deterministic", default=False,
           help="Use deterministic algorithms when possible. May reduce speed.")
@ck.option("--torch-compile/--no-torch-compile", default=False,
           help="Use torch.compile for the training forward pass. Default disabled.")
@ck.option("--num-threads", default=0, type=int,
           help="Set torch CPU thread count. 0 means do not change.")
def main(
    data_file,
    out_classes_file,
    out_relations_file,
    batch_size,
    epochs,
    device,
    embedding_size,
    reg_norm,
    margin,
    learning_rate,
    loss_history_file,
    save_every,
    use_negatives,
    seed,
    clip_norm,
    empty_mode,
    add_center_radius,
    save_state_file,
    log_every,
    deterministic,
    torch_compile,
    num_threads,
):
    if num_threads and num_threads > 0:
        torch.set_num_threads(num_threads)

    set_seed(seed, deterministic=deterministic)
    device = resolve_device(device)

    print("PyTorch:", torch.__version__)
    print("Device:", device)
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(device))

    train_data, classes, relations = load_data(
        data_file,
        use_negatives=use_negatives,
        seed=seed,
    )

    nb_classes = len(classes)
    nb_relations = len(relations)

    print("Classes:", nb_classes)
    print("Relations:", nb_relations)
    for key in ["nf1", "nf2", "nf3", "nf4", "disjoint", "top", "nf3_neg"]:
        print(f"{key}:", train_data[key].shape)

    nb_train_data = max(len(v) for v in train_data.values() if len(v) > 0)
    train_steps = int(math.ceil(nb_train_data / float(batch_size)))
    train_steps = max(1, train_steps)

    print("Batch size:", batch_size)
    print("Steps per epoch:", train_steps)
    print("Empty mode:", empty_mode)

    generator = BatchGenerator(
        train_data,
        batch_size=batch_size,
        steps=train_steps,
        empty_mode=empty_mode,
    )

    cls_list = invert_index(classes)
    rel_list = invert_index(relations)

    model = ELModel(
        nb_classes=nb_classes,
        nb_relations=nb_relations,
        embedding_size=embedding_size,
        batch_size=batch_size,
        margin=margin,
        reg_norm=reg_norm,
        use_negatives=use_negatives,
    ).to(device)

    if torch_compile:
        if empty_mode == "skip":
            print("Warning: torch.compile with --empty-mode skip may trigger recompilation if shapes change.")
        train_model = torch.compile(model)
    else:
        train_model = model

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # Save initial embeddings.
    save_embeddings(
        model,
        cls_list,
        rel_list,
        suffix_path(out_classes_file, "_init"),
        suffix_path(out_relations_file, "_init"),
        add_center_radius=add_center_radius,
    )

    if loss_history_file:
        init_history_file(loss_history_file)

    best_loss = float("inf")
    nan_detected = False

    for epoch in range(1, epochs + 1):
        train_model.train()
        t0 = time.time()

        running = {
            "loss": 0.0,
            "nf1": 0.0,
            "nf2": 0.0,
            "nf3": 0.0,
            "nf4": 0.0,
            "disjoint": 0.0,
            "top": 0.0,
            "nf3_neg": 0.0,
        }

        for step in range(train_steps):
            batch_np = generator.next_batch()
            batch = move_batch_to_device(batch_np, device)

            optimizer.zero_grad(set_to_none=True)

            loss, components = train_model(batch)

            if not torch.isfinite(loss):
                print(f"Non-finite loss detected at epoch {epoch}, step {step + 1}. Stopping.")
                nan_detected = True
                break

            loss.backward()

            if clip_norm is not None and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            optimizer.step()

            running["loss"] += float(loss.detach().cpu())
            for key in components:
                running[key] += float(components[key].detach().cpu())

        if nan_detected:
            break

        logs = {k: v / train_steps for k, v in running.items()}
        logs["epoch"] = epoch
        logs["time_sec"] = time.time() - t0
        logs["lr"] = optimizer.param_groups[0]["lr"]

        if loss_history_file:
            append_history(loss_history_file, logs)

        current_loss = logs["loss"]

        if log_every and log_every > 0 and epoch % log_every == 0:
            print(
                f"Epoch {epoch:05d}/{epochs} | "
                f"loss={logs['loss']:.6f} | "
                f"nf1={logs['nf1']:.6f} | "
                f"nf2={logs['nf2']:.6f} | "
                f"nf3={logs['nf3']:.6f} | "
                f"nf4={logs['nf4']:.6f} | "
                f"dis={logs['disjoint']:.6f} | "
                f"top={logs['top']:.6f} | "
                f"neg={logs['nf3_neg']:.6f} | "
                f"time={logs['time_sec']:.2f}s"
            )

        # Same spirit as the TF callback: save current best to the main output path.
        if current_loss < best_loss:
            best_loss = current_loss
            save_embeddings(
                model,
                cls_list,
                rel_list,
                out_classes_file,
                out_relations_file,
                add_center_radius=add_center_radius,
            )

        if save_every and save_every > 0 and epoch % save_every == 0:
            save_embeddings(
                model,
                cls_list,
                rel_list,
                suffix_path(out_classes_file, f"_epoch{epoch}"),
                suffix_path(out_relations_file, f"_epoch{epoch}"),
                add_center_radius=add_center_radius,
            )
            print(f"Saved epoch {epoch} embeddings. Current loss: {current_loss:.6f}")

    if not nan_detected:
        # Match the old script's behavior: final embeddings overwrite the main output.
        save_embeddings(
            model,
            cls_list,
            rel_list,
            out_classes_file,
            out_relations_file,
            add_center_radius=add_center_radius,
        )
        print("Saved final embeddings:", out_classes_file, out_relations_file)
    else:
        print("Training stopped due to non-finite loss.")
        print("The last finite best embeddings, if any, were saved to:")
        print(" ", out_classes_file)
        print(" ", out_relations_file)

    if save_state_file:
        os.makedirs(os.path.dirname(save_state_file) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "classes": classes,
                "relations": relations,
                "embedding_size": embedding_size,
                "margin": margin,
                "reg_norm": reg_norm,
                "use_negatives": use_negatives,
                "best_loss": best_loss,
            },
            save_state_file,
        )
        print("Saved torch checkpoint:", save_state_file)


class ELModel(nn.Module):
    """
    PyTorch version of the TensorFlow ELModel.

    Class embedding:
        shape = [nb_classes, embedding_size + 1]
        last dimension = radius parameter

    Relation embedding:
        shape = [max(nb_relations, 1), embedding_size]
    """

    def __init__(
        self,
        nb_classes,
        nb_relations,
        embedding_size,
        batch_size,
        margin=-0.1,
        reg_norm=1.0,
        use_negatives=False,
    ):
        super().__init__()

        self.nb_classes = nb_classes
        self.nb_relations = nb_relations
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.margin = float(margin)
        self.reg_norm = float(reg_norm)
        self.use_negatives = use_negatives
        self.inf = 100.0

        cls_weights = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(nb_classes, embedding_size + 1),
        ).astype(np.float32)
        cls_norm = np.linalg.norm(cls_weights, axis=1, keepdims=True)
        cls_weights = cls_weights / np.maximum(cls_norm, 1e-12)

        rel_weights = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(max(nb_relations, 1), embedding_size),
        ).astype(np.float32)
        rel_norm = np.linalg.norm(rel_weights, axis=1, keepdims=True)
        rel_weights = rel_weights / np.maximum(rel_norm, 1e-12)

        self.cls_embeddings = nn.Embedding(nb_classes, embedding_size + 1)
        self.rel_embeddings = nn.Embedding(max(nb_relations, 1), embedding_size)

        with torch.no_grad():
            self.cls_embeddings.weight.copy_(torch.from_numpy(cls_weights))
            self.rel_embeddings.weight.copy_(torch.from_numpy(rel_weights))

    def forward(self, batch):
        """
        Returns:
            total_loss: scalar tensor
            components: dict of scalar tensors
        """
        components = {}

        components["nf1"] = self.mean_or_zero(self.nf1_loss(batch["nf1"]))
        components["nf2"] = self.mean_or_zero(self.nf2_loss(batch["nf2"]))
        components["nf3"] = self.mean_or_zero(self.nf3_loss(batch["nf3"]))
        components["nf4"] = self.mean_or_zero(self.nf4_loss(batch["nf4"]))
        components["disjoint"] = self.mean_or_zero(self.dis_loss(batch["disjoint"]))
        components["top"] = self.mean_or_zero(self.top_loss(batch["top"]))

        if self.use_negatives:
            components["nf3_neg"] = self.mean_or_zero(self.nf3_neg_loss(batch["nf3_neg"]))
        else:
            components["nf3_neg"] = self.zero()

        total_loss = (
            components["nf1"]
            + components["nf2"]
            + components["nf3"]
            + components["nf4"]
            + components["disjoint"]
            + components["top"]
            + components["nf3_neg"]
        )

        return total_loss, components

    def nf1_loss(self, input_ids):
        """
        C SubClassOf D

        Loss:
            relu(||c - d|| + r_c - r_d - margin)
            + reg(c)
            + reg(d)
        """
        c = self.cls_embeddings(input_ids[:, 0])
        d = self.cls_embeddings(input_ids[:, 1])

        rc = torch.abs(c[:, -1:])
        rd = torch.abs(d[:, -1:])

        x1 = c[:, :-1]
        x2 = d[:, :-1]

        euc = self.safe_norm(x1 - x2)
        dst = F.relu(euc + rc - rd - self.margin)

        return dst + self.reg(x1) + self.reg(x2)

    def nf2_loss(self, input_ids):
        """
        C and D SubClassOf E

        This follows the original TensorFlow implementation.
        Notice that the radius of E is not used in the old code.
        """
        c = self.cls_embeddings(input_ids[:, 0])
        d = self.cls_embeddings(input_ids[:, 1])
        e = self.cls_embeddings(input_ids[:, 2])

        rc = torch.abs(c[:, -1:])
        rd = torch.abs(d[:, -1:])

        x1 = c[:, :-1]
        x2 = d[:, :-1]
        x3 = e[:, :-1]

        sr = rc + rd

        dst = self.safe_norm(x2 - x1)
        dst2 = self.safe_norm(x3 - x1)
        dst3 = self.safe_norm(x3 - x2)

        dst_loss = (
            F.relu(dst - sr - self.margin)
            + F.relu(dst2 - rc - self.margin)
            + F.relu(dst3 - rd - self.margin)
        )

        return dst_loss + self.reg(x1) + self.reg(x2) + self.reg(x3)

    def nf3_loss(self, input_ids):
        """
        C SubClassOf R some D
        """
        c = self.cls_embeddings(input_ids[:, 0])
        r = self.rel_embeddings(input_ids[:, 1])
        d = self.cls_embeddings(input_ids[:, 2])

        x1 = c[:, :-1]
        x2 = d[:, :-1]

        rc = torch.abs(c[:, -1:])
        rd = torch.abs(d[:, -1:])

        euc = self.safe_norm((x1 + r) - x2)
        dst = F.relu(euc + rc - rd - self.margin)

        return dst + self.reg(x1) + self.reg(x2)

    def nf3_neg_loss(self, input_ids):
        """
        Negative corrupted nf3 triples.

        Original TensorFlow logic:
            dst = -(euc - rc - rd - margin)
            relu(dst)
        """
        c = self.cls_embeddings(input_ids[:, 0])
        r = self.rel_embeddings(input_ids[:, 1])
        d = self.cls_embeddings(input_ids[:, 2])

        x1 = c[:, :-1]
        x2 = d[:, :-1]

        rc = torch.abs(c[:, -1:])
        rd = torch.abs(d[:, -1:])

        euc = self.safe_norm((x1 + r) - x2)
        dst = -(euc - rc - rd - self.margin)

        return F.relu(dst) + self.reg(x1) + self.reg(x2)

    def nf4_loss(self, input_ids):
        """
        R some C SubClassOf D
        """
        r = self.rel_embeddings(input_ids[:, 0])
        c = self.cls_embeddings(input_ids[:, 1])
        d = self.cls_embeddings(input_ids[:, 2])

        rc = torch.abs(c[:, -1:])
        rd = torch.abs(d[:, -1:])

        x1 = c[:, :-1]
        x2 = d[:, :-1]

        dst = self.safe_norm((x1 - r) - x2)
        dst_loss = F.relu(dst - (rc + rd) - self.margin)

        return dst_loss + self.reg(x1) + self.reg(x2)

    def dis_loss(self, input_ids):
        """
        Disjointness-style loss.

        The third column is usually owl:Nothing, but the original implementation
        only used the first two class IDs. This behavior is preserved.
        """
        c = self.cls_embeddings(input_ids[:, 0])
        d = self.cls_embeddings(input_ids[:, 1])

        rc = torch.abs(c[:, -1:])
        rd = torch.abs(d[:, -1:])

        x1 = c[:, :-1]
        x2 = d[:, :-1]

        dst = self.safe_norm(x2 - x1)

        return F.relu((rc + rd) - dst + self.margin) + self.reg(x1) + self.reg(x2)

    def top_loss(self, input_ids):
        """
        Force owl:Thing radius to a large value.
        """
        d = self.cls_embeddings(input_ids[:, 0])
        rd = torch.abs(d[:, -1:])

        return torch.abs(rd - self.inf)

    def safe_norm(self, x, axis=1, eps=1e-7):
        return torch.sqrt(torch.sum(x * x, dim=axis, keepdim=True) + eps)

    def reg(self, x):
        return torch.abs(self.safe_norm(x) - self.reg_norm)

    def zero(self):
        return self.cls_embeddings.weight.new_tensor(0.0)

    def mean_or_zero(self, loss_vec):
        if loss_vec.numel() == 0:
            return self.zero()
        return loss_vec.mean()


class BatchGenerator:
    """
    Replacement for the old tf.keras.utils.Sequence generator.

    It samples each normal form independently with replacement, just like the
    old code.

    empty_mode:
        tf:
            If a normal form has zero axioms, return an all-zero batch.
            This reproduces the old TensorFlow behavior.

        skip:
            If a normal form has zero axioms, return a tensor with shape [0, width].
            The model then contributes zero loss for that normal form.
    """

    def __init__(self, data, batch_size=128, steps=100, empty_mode="tf"):
        self.data = data
        self.batch_size = batch_size
        self.steps = int(steps)
        self.empty_mode = empty_mode

    def __len__(self):
        return self.steps

    def _sample(self, key, width):
        arr = self.data[key]

        if len(arr) == 0:
            if self.empty_mode == "skip":
                return np.zeros((0, width), dtype=np.int64)
            return np.zeros((self.batch_size, width), dtype=np.int64)

        idx = np.random.choice(arr.shape[0], self.batch_size, replace=True)
        return arr[idx].astype(np.int64, copy=False)

    def next_batch(self):
        return {
            "nf1": self._sample("nf1", 2),
            "nf2": self._sample("nf2", 3),
            "nf3": self._sample("nf3", 3),
            "nf4": self._sample("nf4", 3),
            "disjoint": self._sample("disjoint", 3),
            "top": self._sample("top", 1),
            "nf3_neg": self._sample("nf3_neg", 3),
        }


def load_data(filename, use_negatives=False, seed=100):
    classes = {}
    relations = {}
    data = {
        "nf1": [],
        "nf2": [],
        "nf3": [],
        "nf4": [],
        "disjoint": [],
    }

    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("SubObjectPropertyOf"):
                continue

            parsed = parse_axiom(line)

            if parsed is None:
                continue

            form, values = parsed

            if form == "nf1":
                c, d = values
                data["nf1"].append((class_id(classes, c), class_id(classes, d)))

            elif form == "nf2":
                c, d, e = values
                form2 = "disjoint" if e == "owl:Nothing" else "nf2"
                data[form2].append(
                    (
                        class_id(classes, c),
                        class_id(classes, d),
                        class_id(classes, e),
                    )
                )

            elif form == "nf3":
                c, r, d = values
                data["nf3"].append(
                    (
                        class_id(classes, c),
                        relation_id(relations, r),
                        class_id(classes, d),
                    )
                )

            elif form == "nf4":
                r, c, d = values
                data["nf4"].append(
                    (
                        relation_id(relations, r),
                        class_id(classes, c),
                        class_id(classes, d),
                    )
                )

    if "owl:Thing" not in classes:
        classes["owl:Thing"] = len(classes)

    if "owl:Nothing" not in classes:
        classes["owl:Nothing"] = len(classes)

    # Add a minimal disjointness constraint if none exists.
    # This follows your modified TF version.
    if len(data["disjoint"]) == 0 and len(classes) >= 3:
        nothing = classes["owl:Nothing"]
        ids = [
            v
            for k, v in classes.items()
            if k not in {"owl:Thing", "owl:Nothing"}
        ]
        if len(ids) >= 2:
            data["disjoint"].append((ids[0], ids[1], nothing))

    rng = np.random.RandomState(seed)

    data["nf3_neg"] = []

    if use_negatives and len(data["nf3"]) > 0:
        class_ids = np.array(list(classes.values()), dtype=np.int64)

        for c, r, d in data["nf3"]:
            data["nf3_neg"].append((c, r, int(rng.choice(class_ids))))
            data["nf3_neg"].append((int(rng.choice(class_ids)), r, d))

    data["nf1"] = np.asarray(data["nf1"], dtype=np.int64).reshape((-1, 2))
    data["nf2"] = np.asarray(data["nf2"], dtype=np.int64).reshape((-1, 3))
    data["nf3"] = np.asarray(data["nf3"], dtype=np.int64).reshape((-1, 3))
    data["nf4"] = np.asarray(data["nf4"], dtype=np.int64).reshape((-1, 3))
    data["disjoint"] = np.asarray(data["disjoint"], dtype=np.int64).reshape((-1, 3))
    data["top"] = np.asarray([[classes["owl:Thing"]]], dtype=np.int64)
    data["nf3_neg"] = np.asarray(data["nf3_neg"], dtype=np.int64).reshape((-1, 3))

    for key in data:
        if len(data[key]) > 0:
            idx = np.arange(len(data[key]))
            rng.shuffle(idx)
            data[key] = data[key][idx]

    return data, classes, relations


def parse_axiom(line):
    """
    Parse both OWL functional-style and simplified normalized forms.
    """
    line = line.strip()

    # Functional-style: SubClassOf(...)
    if line.startswith("SubClassOf(") and line.endswith(")"):
        inner = line[len("SubClassOf("):-1].strip()

        if inner.startswith("ObjectIntersectionOf("):
            # ObjectIntersectionOf(C D) E
            m = re.match(r"ObjectIntersectionOf$(\S+)\s+(\S+)$\s+(\S+)$", inner)
            if m:
                return "nf2", (m.group(1), m.group(2), m.group(3))

        if inner.startswith("ObjectSomeValuesFrom("):
            # ObjectSomeValuesFrom(R C) D
            m = re.match(r"ObjectSomeValuesFrom$(\S+)\s+(\S+)$\s+(\S+)$", inner)
            if m:
                return "nf4", (m.group(1), m.group(2), m.group(3))

        if "ObjectSomeValuesFrom(" in inner:
            # C ObjectSomeValuesFrom(R D)
            m = re.match(r"(\S+)\s+ObjectSomeValuesFrom$(\S+)\s+(\S+)$$", inner)
            if m:
                return "nf3", (m.group(1), m.group(2), m.group(3))

        toks = inner.split()
        if len(toks) == 2:
            return "nf1", (toks[0], toks[1])

        return None

    # Simplified examples:
    # C SubClassOf D
    # C SubClassOf R some D
    # R some C SubClassOf D
    # C and D SubClassOf E
    toks = line.split()

    if len(toks) == 3 and toks[1] == "SubClassOf":
        return "nf1", (toks[0], toks[2])

    if len(toks) == 5 and toks[1] == "and" and toks[3] == "SubClassOf":
        return "nf2", (toks[0], toks[2], toks[4])

    if len(toks) == 5 and toks[1] == "SubClassOf" and toks[3] == "some":
        return "nf3", (toks[0], toks[2], toks[4])

    if len(toks) == 5 and toks[1] == "some" and toks[3] == "SubClassOf":
        return "nf4", (toks[0], toks[2], toks[4])

    return None


def class_id(classes, name):
    if name not in classes:
        classes[name] = len(classes)
    return classes[name]


def relation_id(relations, name):
    if name not in relations:
        relations[name] = len(relations)
    return relations[name]


def invert_index(mapping):
    inv = {v: k for k, v in mapping.items()}
    return [inv[i] for i in range(len(inv))]


def save_embeddings(
    el_model,
    cls_list,
    rel_list,
    out_classes_file,
    out_relations_file,
    add_center_radius=False,
):
    os.makedirs(os.path.dirname(out_classes_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_relations_file) or ".", exist_ok=True)

    cls_embeddings = (
        el_model.cls_embeddings.weight
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    rel_embeddings = (
        el_model.rel_embeddings.weight
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    cls_data = {
        "classes": cls_list,
        "embeddings": list(cls_embeddings),
    }

    if add_center_radius:
        cls_data["center"] = list(cls_embeddings[:, :-1])
        cls_data["radius"] = list(np.abs(cls_embeddings[:, -1]))

    pd.DataFrame(cls_data).to_pickle(out_classes_file)

    pd.DataFrame({
        "relations": rel_list,
        "embeddings": list(rel_embeddings[:len(rel_list)]),
    }).to_pickle(out_relations_file)


def move_batch_to_device(batch_np, device):
    return {
        key: torch.as_tensor(value, dtype=torch.long, device=device)
        for key, value in batch_np.items()
    }


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(device):
    d = str(device).lower().strip()

    if d == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Accept old TF-style names.
    if d.startswith("/"):
        d = d[1:]

    if d.startswith("gpu:"):
        d = "cuda:" + d.split(":", 1)[1]

    if d.startswith("cpu:"):
        d = "cpu"

    out = torch.device(d)

    if out.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    return out


def suffix_path(path, suffix):
    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}{suffix}{ext}"
    return f"{path}{suffix}"


def init_history_file(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fieldnames = [
        "epoch",
        "loss",
        "nf1",
        "nf2",
        "nf3",
        "nf4",
        "disjoint",
        "top",
        "nf3_neg",
        "time_sec",
        "lr",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_history(path, logs):
    fieldnames = [
        "epoch",
        "loss",
        "nf1",
        "nf2",
        "nf3",
        "nf4",
        "disjoint",
        "top",
        "nf3_neg",
        "time_sec",
        "lr",
    ]

    row = {k: logs.get(k, "") for k in fieldnames}

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


if __name__ == "__main__":
    main()