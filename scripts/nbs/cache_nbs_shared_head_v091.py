#!/usr/bin/env python3
"""Resumable frozen v089-epoch60 cache with global supervision exclusion."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nbs_models/nbs_protein_go")]


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--query-batch", type=int)
    args = p.parse_args(argv)
    os.chdir(ROOT)
    import fcntl
    import torch
    from nbs_pg.shared_head_io_v091 import (CACHE_FILES, SCHEMA, labels_for_ids, load_cache,
        load_source, macro_id_groups, split_cache_macro, sha256, write_json,
        candidate_statistics, update_candidate_statistics, finish_candidate_statistics)
    cfg = json.loads(args.config.read_text())
    cache = cfg["cache"]
    size = cache["query_batch"] if args.query_batch is None else args.query_batch
    if type(size) is not int or size < 1 or cfg.get("source_epoch") != 60:
        raise ValueError("requires a positive query microbatch and source epoch60")
    spec = {**cache, "query_batch":size, "sampling_view":"fixed_dynamic_context_per_macro; frozen_eval; global_seed_exclusion; not_DDP_replay"}
    implementation = {Path(__file__).name:sha256(__file__),
                      "shared_head_io_v091.py":sha256(ROOT / "nbs_models/nbs_protein_go/nbs_pg/shared_head_io_v091.py")}
    if cache["sampler"] == "fast":
        from nbs_pg.full_task_sampler_fast import implementation_sha256
        implementation["full_task_sampler_fast.py"] = implementation_sha256()
    directory = Path(cfg["cache_dir"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = directory / "cache_manifest.json"
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if previous is not None:
            if (previous.get("schema") != SCHEMA or previous["source_checkpoint_sha256"] != sha256(cfg["source_checkpoint"])
                or previous["source_config"] != json.loads(Path(cfg["source_config"]).read_text())
                or previous["cache_spec"] != spec or previous["cache_builder_implementation"] != implementation):
                raise ValueError("existing cache has different source/config/view/implementation; use a new cache_dir")
            if previous.get("complete") is True:
                load_cache(directory)
                print(f"[cache] verified and reused: {directory}", flush=True)
                return
        elif any(x.name != ".cache.lock" for x in directory.iterdir()):
            raise FileExistsError("unrecognized partial cache: use a new cache_dir")
        model, data, ontology, metadata = load_source(cfg["source_config"], cfg["source_checkpoint"], args.device,
            sampler=cache["sampler"], expected_epoch=60)
        ft, world = metadata["source_config"]["full_task"], metadata["source_world_size"]
        if any(cache["global_"+k] != ft[k]*world for k in ("weak_batch","core_batch")):
            raise ValueError("cache global weak/core counts must match source experiment")
        if np.intersect1d(np.r_[data.weak_ids,data.core_ids], data.validation_ids).size:
            raise ValueError("holdout IDs entered training population")
        groups = list(macro_id_groups(data.weak_ids, data.core_ids, cache["global_weak_batch"],cache["global_core_batch"],cache["seed"]))
        ids = np.concatenate([g[0] for g in groups]); weak = np.concatenate([g[1] for g in groups])
        label_rows = labels_for_ids(data,ids)
        mapping = model.task_to_ontology.cpu().numpy().astype(np.int64)
        geometry = np.concatenate((ontology["center"].cpu().numpy()[mapping],
                   np.log(np.abs(ontology["offset"].cpu().numpy()[mapping]).clip(1e-8))),axis=1).astype(np.float32)
        values = {"protein_ids.i64.npy":ids, "is_weak.bool.npy":weak,
            "label_indptr.i64.npy":np.r_[0,np.cumsum([len(x) for x in label_rows],dtype=np.int64)],
            "label_go_idx.i32.npy":np.concatenate(label_rows).astype(np.int32),
            "geometry.f32.npy":geometry, "task_to_ontology.i64.npy":mapping,
            "original_weight.f32.npy":model.classifier.weight.detach().cpu().numpy(),
            "original_bias.f32.npy":model.classifier.bias.detach().cpu().numpy()}
        # Skip the large frozen classifier matmul: the source forward now
        # returns exactly the classifier input used by the original model.
        width = model.classifier.in_features
        if geometry.shape != (data.num_task_go,1024) or cfg["head"]["hidden_dim"] != width:
            raise ValueError("shared head dimensions differ from source hidden/geometry")
        model.classifier = torch.nn.Identity()
        if previous is None:
            for name,value in values.items():
                np.save(directory/name,value,allow_pickle=False)
            hidden = np.lib.format.open_memmap(directory/"hidden.f32.npy",mode="w+",dtype=np.float32,shape=(len(ids),width))
            manifest = dict(schema=SCHEMA,complete=False,**metadata,cache_spec=spec,
                num_proteins=len(ids),num_task_go=data.num_task_go,completed_macros=0,completed_rows=0,
                committed_hidden=[],candidate_stats={r:candidate_statistics() for r in ("core","weak")},
                cache_builder_implementation=implementation)
            write_json(manifest_path,manifest)
        else:
            manifest = previous
            for name,value in values.items():
                actual = np.load(directory/name,mmap_mode="r",allow_pickle=False)
                if actual.dtype != value.dtype or not np.array_equal(actual,value):
                    raise ValueError("partial cache static data differs: "+name)
            hidden = np.lib.format.open_memmap(directory/"hidden.f32.npy",mode="r+")
            if hidden.shape != (len(ids),width) or hidden.dtype != np.float32:
                raise ValueError("partial hidden cache dimensions differ")
            for row in manifest["committed_hidden"]:
                if array_sha(hidden[row["start"]:row["end"]]) != row["sha256"]:
                    raise ValueError("committed cache segment corrupted")
            if (len(manifest["committed_hidden"]) != manifest["completed_macros"] or
                sum(len(g[0]) for g in groups[:manifest["completed_macros"]]) != manifest["completed_rows"]):
                raise ValueError("partial cache progress counters differ")
            print(f"[cache] resuming after macro {manifest['completed_macros']}",flush=True)
        del values,label_rows
        cursor=manifest["completed_rows"]
        started=time.monotonic()
        with torch.no_grad():
            go=model.encode_go()
            for index,(global_ids,global_weak) in enumerate(groups,1):
                if index <= manifest["completed_macros"]:
                    continue
                begin=cursor
                data.set_sampling_context(index,0,True)
                # Reorder rows only within compute; write back in immutable ID order.
                positions={int(protein):begin+i for i,protein in enumerate(global_ids)}
                for local in split_cache_macro(global_ids,global_weak,size):
                    batch=data.batch(local,args.device,supervision_seed_ids=global_ids)
                    h=model(batch,go_encoding=go).float().cpu().numpy()
                    if h.shape != (len(local),width) or not np.isfinite(h).all():
                        raise ValueError("invalid source hidden representation")
                    loc=np.asarray([positions[int(protein)] for protein in local])
                    hidden[loc]=h
                    candidates=batch["sampled_candidate_go"][batch["sampled_seed_index"]].cpu().numpy()
                    confidence=batch["sampled_candidate_attr"][batch["sampled_seed_index"]].cpu().numpy()[...,0]
                    targets=batch["positive_mask"].cpu().numpy()
                    roles=batch["is_weak"].cpu().numpy()
                    for role,select in (("weak",roles),("core",~roles)):
                        update_candidate_statistics(manifest["candidate_stats"][role],candidates[select],confidence[select],
                            [np.flatnonzero(row) for row in targets[select]])
                    del batch
                cursor += len(global_ids)
                hidden.flush()
                manifest["completed_macros"]=index; manifest["completed_rows"]=cursor
                manifest["committed_hidden"].append(dict(start=begin,end=cursor,sha256=array_sha(hidden[begin:cursor])))
                write_json(manifest_path,manifest)
                print(f"[cache] macro {index}/{len(groups)} rows {cursor}/{len(ids)} elapsed={time.monotonic()-started:.1f}s",flush=True)
        hidden.flush(); del hidden
        records={}
        for name in CACHE_FILES:
            value=np.load(directory/name,mmap_mode="r",allow_pickle=False)
            records[name]=dict(sha256=sha256(directory/name),shape=list(value.shape),dtype=str(value.dtype))
        manifest.update(complete=True,files=records)
        write_json(directory/"candidate_quality_train.json",{r:finish_candidate_statistics(v) for r,v in manifest["candidate_stats"].items()})
        write_json(manifest_path,manifest)
        load_cache(directory)
        print(f"[cache] complete: {directory}",flush=True)


if __name__ == "__main__":
    main()
