#!/usr/bin/env python3
"""Stream Stage-1 frozen Backbone / Expert / Modelout full-task predictions.

Calls the installed Stage-1 architecture, dataset, checkpoint loader, and
query-decoder implementation. It does not use test labels in model forward.
For a new dark-weak split the MSA index and (optional) external probability
array MUST be prepared for the exact split; this script does not invent them.

Usage: python scripts/dark_weak/export_stage1_predictions.py --help
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

# The project modules are installed in their original locations.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'msa_models'))
from experiments import eval_weak_ind_test_detr_diagnostics as stage1
from scripts.dark_weak.stage1_alignment import ProteinRowPlacement


def read_ids(path):
    vals = [s.strip() for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not vals or len(vals) != len(set(vals)):
        raise ValueError(f'Protein IDs must be nonempty and unique: {path}')
    return vals


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(8<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def export(args):
    if getattr(args, 'allow_eval_label_boost', False):
        raise ValueError('Evaluation-label boost is forbidden in the Dark–Weak inference protocol.')
    if int(args.num_classes or 0) not in (21312,7038,2903):
        n = stage1.infer_num_classes_from_checkpoint(args.checkpoint)
        if n not in (21312,7038,2903):
            raise ValueError('Set --num_classes to your unchanged task vocabulary width')
        args.num_classes = int(n)
    task = stage1.normalize_task(args.task)
    expected_width = {'bp':21312,'mf':7038,'cc':2903}[task]
    if args.num_classes != expected_width:
        raise ValueError(f'Task={task} expects {expected_width} GO columns, got {args.num_classes}')
    go_ids = [s.strip() for s in Path(args.go_ids).read_text().splitlines() if s.strip()]
    if len(go_ids) != expected_width or len(set(go_ids)) != expected_width:
        raise ValueError('GO identifiers must be unique, complete and in original model column order')
    requested = read_ids(args.expected_protein_ids)
    import pickle
    with Path(args.file_address).open('rb') as handle:
        meta = pickle.load(handle)
    original_target_ids = [str(x) for x in meta[args.mode][{'bp':'biological_process','mf':'molecular_function','cc':'cellular_component'}[task]]['proteins']]
    if original_target_ids != requested:
        raise ValueError('Expected ID file must have the EXACT order of the inference metadata split; '
                         'the external probability wrapper indexes probabilities by this metadata order')
    placement = ProteinRowPlacement(requested)
    if args.device == 'auto':
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device=torch.device(args.device)
    args.distributed=False; args.rank=0; args.world_size=1; args.local_rank=0
    args.task=task; args.batch_size=args.eval_batch_size
    opt = stage1.build_weak_opt_from_config(args)
    opt.mode=args.mode; opt.shuffle=False
    model=stage1.WeakMSAGOWithDETRDecoder(opt,args)
    info=stage1.load_detr_checkpoint(model,args.checkpoint,device)
    if args.export_modelout and (info.get('warning') or info.get('missing_query_decoder')
                                 or info.get('unexpected_query_decoder')):
        raise ValueError('Stage1 Modelout requires a fully loaded, trained query decoder; '
                         f'checkpoint diagnostic: {info}')
    if info.get('missing_backbone'):
        raise ValueError(f'Stage1 backbone load missing state keys: {info["missing_backbone"][:10]}')
    model=model.to(device); model.eval()
    base_dataset=stage1.build_msa_dataset(opt,mode=args.mode,task=task,need_proteins=True)
    extpath=stage1.clean_optional_path(args.external_prob_path)
    if extpath:
        if not args.external_protein_ids:
            raise ValueError('External Expert probabilities require --external-protein-ids; '
                             'a .npy shape match alone cannot prove row identity')
        if read_ids(args.external_protein_ids)!=requested:
            raise ValueError('External Expert probability protein IDs/order differ from requested targets')
        dataset=stage1.PseudoProbDataset(base_dataset=base_dataset,
            metadata_file=args.file_address,mode=args.mode,task=task,
            prob_path=extpath,num_classes=args.num_classes)
    else:
        dataset=base_dataset
    dataset_proteins = [str(v) for v in dataset.proteins]
    if len(dataset_proteins) != len(set(dataset_proteins)):
        raise ValueError('MSA dataset has duplicate protein names')
    absent = set(requested) - set(dataset_proteins)
    unexpected = set(dataset_proteins) - set(requested)
    if absent or unexpected or len(dataset)!=len(requested):
        raise ValueError(f'MSA index coverage differs: missing={len(absent)} {sorted(absent)[:6]}, '
            f'extra={len(unexpected)} {sorted(unexpected)[:6]}; '
            'MSABinaryDataset defaults to missing_action=skip. Build a matching historical MSA index view.')
    if args.export_modelout and args.modelout_contract == 'v088_weak_supervision':
        actual_mode = getattr(args, 'query_decoder_logit_base_mode', None)
        train_args_path = (stage1.auto_train_args_path(args.checkpoint)
             if str(getattr(args,'train_args_json','auto')).lower() == 'auto'
             else Path(args.train_args_json) if args.train_args_json else None)
        saved_train_args = stage1.load_json_if_exists(train_args_path)
        saved_mode = saved_train_args.get('query_decoder_logit_base_mode')
        if saved_mode is not None and saved_mode != 'mix_expert_base_anchor':
            raise ValueError('The checkpoint training args.json identifies a different decoder logit '
                             f'base mode {saved_mode!r}; do not override it merely for Dark–Weak evaluation')
        if args.decoder_prob_source != 'expert' or actual_mode != 'mix_expert_base_anchor':
            raise ValueError('The NBS v088 training weak-target Modelout is precisely '
                             'modelout::mix_expert_base_anchor::decoderprob::expert; '
                             'require --decoder-prob-source expert and trained '
                             '--query_decoder_logit_base_mode mix_expert_base_anchor, '
                             f'got {args.decoder_prob_source!r} and {actual_mode!r}')
    if args.export_modelout and args.decoder_prob_source!='base' and extpath is None:
        raise ValueError('Modelout requires properly aligned external Expert probabilities for this split')
    if args.export_expert and extpath is None:
        raise ValueError('Expert export requested without --external_prob_path')
    original_np=None
    if args.decoder_prob_source=='mix_expert_original':
        if not args.original_prob_path:
            raise ValueError('mix_expert_original requires --original_prob_path')
        if not args.original_protein_ids or read_ids(args.original_protein_ids)!=requested:
            raise ValueError('Original probabilities require aligned --original-protein-ids')
        original_np=np.load(args.original_prob_path,mmap_mode='r')
        if original_np.shape != (len(dataset),expected_width):
            raise ValueError(f'original_prob shape={original_np.shape}, expected {(len(dataset),expected_width)}')
    loader=stage1.make_loader(dataset,batch_size=args.eval_batch_size,shuffle=False,
        num_workers=args.dataloader_num_workers,pin_memory=args.pin_memory,
        drop_last=False,rank=0,world_size=1,seed=args.seed,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers and args.dataloader_num_workers>0)
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Nonempty output directory: {out}; specify --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    channels=['backbone_base']
    if args.export_expert: channels.append('expert')
    if args.export_modelout: channels.append('modelout')
    # v085 exact micro PR-AUC/AP group tied scores. Export float32 when possible
    # to avoid introducing fp16 quantization ties absent from native NBS v088.
    output_dtype = np.dtype(getattr(args, 'output_dtype', 'float16'))
    dtype_tag = 'f32' if output_dtype == np.dtype('float32') else 'f16'
    output_paths = {key: out / f'{key}.prob.{dtype_tag}.npy' for key in channels}
    sink = {key: np.lib.format.open_memmap(output_paths[key], mode='w+',
            dtype=output_dtype, shape=(len(dataset), expected_width))
            for key in channels}
    sample_count=0
    amp=(device.type=='cuda' and not args.no_amp)
    with torch.no_grad():
        for batch in tqdm(loader,desc='Stage1 dark-weak export'):
            if extpath:
                proteins,x,y,ext_prob=stage1.unpack_pseudo_batch(batch)
            else:
                proteins,x,y=stage1.unpack_batch(batch);ext_prob=None
            if proteins is None:
                raise RuntimeError('MSA dataset did not supply protein IDs; cannot verify row alignment')
            if isinstance(proteins,str): proteins=[proteins]
            proteins=[str(s) for s in proteins]
            # make_loader groups MSA batches BY SHARD even with shuffle=False.
            # Do NOT assume batch iteration order equals metadata/probability order.
            positions=placement.locate_batch(proteins)
            stage1.set_model_proteins(model,proteins)
            x=x.to(device,non_blocking=True).long()
            if ext_prob is not None:
                ext_prob=ext_prob.to(device,non_blocking=True).float().clamp(0,1)
            with autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
                base_logits,h=stage1.model_forward_once(model,x,tuple(map(int,args.permute_dims)))
                base_prob=torch.sigmoid(base_logits.float())
                sink['backbone_base'][positions]=base_prob.cpu().numpy().astype(output_dtype)
                if args.export_expert:
                    sink['expert'][positions]=ext_prob.cpu().numpy().astype(output_dtype)
                if args.export_modelout:
                    if args.decoder_prob_source=='expert':
                        source=ext_prob
                    elif args.decoder_prob_source=='base':
                        source=base_prob
                    elif args.decoder_prob_source=='mix_expert_base':
                        a=float(args.decoder_mix_alpha)
                        source=(a*ext_prob + (1-a)*base_prob).clamp(0,1)
                    elif args.decoder_prob_source=='mix_expert_original':
                        original=torch.as_tensor(np.asarray(original_np[positions],dtype=np.float32),device=device)
                        a=float(args.decoder_mix_alpha)
                        source=(a*ext_prob+(1-a)*original).clamp(0,1)
                    else: raise ValueError(args.decoder_prob_source)
                    q=model.query_decoder(h=h,base_logits=base_logits,
                        classifier_weight=model.classifier.weight,external_prob=source,
                        has_external_prob=torch.ones(len(proteins),device=device,dtype=torch.bool),
                        topk_source='external_topk',external_prob_blend_alpha=1.0,
                        y_hint=None)
                    sink['modelout'][positions]=torch.sigmoid(q['logits'].float()).cpu().numpy().astype(output_dtype)
            sample_count += len(proteins)
    placement.assert_complete()
    if sample_count!=len(requested):
        raise ValueError(f'Only {sample_count} / {len(requested)} proteins evaluated')
    for item in sink.values(): item.flush()
    del sink
    (out/'protein_ids.txt').write_text('\n'.join(requested)+'\n')
    (out/'go_ids.txt').write_text('\n'.join(go_ids)+'\n')
    manifest={
        'model_family':'LATENCE Stage1 DETR', 'task':task, 'mode':args.mode,
        'num_proteins':len(requested),'num_go':len(go_ids), 'channels':channels,
        'checkpoint':str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256':hash_file(args.checkpoint),
        'input_metadata':str(Path(args.file_address).resolve()),
        'expected_protein_ids_sha256':hash_file(args.expected_protein_ids),
        'go_ids_sha256':hash_file(args.go_ids),
        'external_prob':str(extpath) if extpath else None,
        'external_prob_sha256':hash_file(extpath) if extpath else None,
        'external_protein_ids_sha256':hash_file(args.external_protein_ids) if extpath else None,
        'modelout_semantics':f"actual_query_decoder_logits source={args.decoder_prob_source} mix_alpha={args.decoder_mix_alpha}",
        'modelout_prediction_key': ('modelout::mix_expert_base_anchor::decoderprob::expert'
              if args.export_modelout and args.modelout_contract=='v088_weak_supervision'
              else None),
        'batch_order_policy':'scatter_by_verified_protein_name_into_metadata_order',
        'output_storage_dtype': str(output_dtype),
        'forward_uses_eval_labels':False,
        'model_loading_info':info,
        'output_files': {key: {'path': str(output_paths[key].resolve()),
                              'sha256': hash_file(output_paths[key]),
                              'storage_dtype': str(output_dtype)} for key in channels}
    }
    (out/'prediction_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(f'[Exported] Stage1 {len(requested)} proteins, {len(go_ids)} GO labels to {out}')
    return manifest


def main(argv=None):
    parser=stage1.build_argparser()
    parser.description=__doc__
    parser.add_argument('--expected-protein-ids',required=True)
    parser.add_argument('--go-ids',required=True)
    parser.add_argument('--export-expert',action='store_true')
    parser.add_argument('--external-protein-ids',type=Path)
    parser.add_argument('--original-protein-ids',type=Path)
    parser.add_argument('--export-modelout',action='store_true')
    parser.add_argument('--decoder-prob-source',choices=('expert','base','mix_expert_base','mix_expert_original'),default='expert')
    parser.add_argument('--modelout-contract',choices=('v088_weak_supervision','generic'),default='v088_weak_supervision')
    parser.add_argument('--decoder-mix-alpha',type=float,default=.8)
    parser.add_argument('--output-dtype',choices=('float16','float32'),default='float16',
                        help='Recommend float32 for original v085 exact micro PR-AUC/AP; '
                             'float16 keeps v0.4.2 paths but creates additional tied scores')
    parser.add_argument('--overwrite',action='store_true')
    keys=stage1.get_cli_provided_keys(list(argv) if argv is not None else sys.argv[1:])
    args=parser.parse_args(argv)
    args=stage1.overlay_train_args(args,cli_keys=keys)
    stage1.validate_args(args)
    if not (0<=args.decoder_mix_alpha<=1): parser.error('alpha must be in [0,1]')
    if args.dataloader_num_workers==0:args.persistent_workers=False
    export(args)

if __name__=='__main__':
    main()
