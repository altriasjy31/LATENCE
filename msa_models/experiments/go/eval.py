from argparse import Namespace
import traceback
import copy
import typing as T
from torch import Tensor
from models import Arch
import sys
import os

from torch.amp.autocast_mode import autocast
amp_flag = True

prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if not prj_dir in sys.path:
    sys.path.append(prj_dir)

# from models import MSAModel
# from models import Coevo
# from experiments.go.GO import MSADataset
import experiments.msa as D

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from functools import reduce
import helper_functions.helper as helper

import tqdm

def evaluation_prog(opt: Namespace, local_rank, world_size):
    """
    Evaluate multiple models and create an ensemble prediction.
    
    Args:
        opt: Command line options
        local_rank: Local rank for distributed training
        world_size: Total number of processes for distributed training
    """
    torch.cuda.set_device(f"cuda:{local_rank}")
    torch.manual_seed(3407)

    batch_size = opt.batch_size
    
    # Setup dataset
    dataset = D.MSADataset(opt.file_address, opt.working_dir, opt.mode, opt.task,
        opt.num_classes, opt.top_k, opt.max_len,
        need_proteins=False,
        msa_max_size=opt.msa_max_size)

    eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=opt.shuffle)
    eval_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=eval_sampler,
                             num_workers=opt.dataloader_num_workers,
                             pin_memory=True)

    # Parse model specifications
    model_dir = opt.model_saving
    model_list = opt.trained_model.split(",")
    no_eval = opt.no_eval
    pred_savingpath = opt.prediction_save
    permute_dims = getattr(opt, "permute_dims", (0, 3, 2, 1))
    
    models = []
    model_names = []
    
    if local_rank == 0:
        print(f"Evaluating {len(model_list)} models: {model_list}")
    
    # Create and load models
    base_model = None
    
    for model_name in model_list:
        model_name = model_name.strip()
        model_names.append(model_name)
        
        # Create new model or clone base model to save initialization time
        if base_model is None:
            model = Arch(opt)
            
            if opt.torch_compile:
                assert hasattr(torch, "compile"), "need pytorch > 2.0"
                model = torch.compile(model)
                
            model = model.to(f"cuda:{local_rank}")
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], 
                                                    output_device=local_rank)
            base_model = model
        else:
            # Instead of clone, create a new model instance to avoid potential issues
            model = Arch(opt)
            
            if opt.torch_compile:
                model = torch.compile(model)
                
            model = model.to(f"cuda:{local_rank}")
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], 
                                                    output_device=local_rank)
        
        # Load state dictionary
        try:
            state_path = os.path.join(model_dir, model_name)
            if not os.path.exists(state_path):
                if local_rank == 0:
                    print(f"Warning: Model file {state_path} not found. Skipping.")
                continue
                
            loaded_state = torch.load(state_path, map_location=f"cuda:{local_rank}")
            model.load_state_dict(state_dict=loaded_state)
            model.eval()
            models.append(model)
            
            if local_rank == 0:
                print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            if local_rank == 0:
                print(f"Error loading model {model_name}: {str(e)}")
    
    if not models:
        raise ValueError("No valid models were loaded for evaluation")
    
    # Evaluate models
    fmax_score_regular, auprc_score_regular, tp_tensor, _ = eval_multi(
        eval_loader, models, no_eval, permute_dims
    )
    
    # Report results
    if local_rank == 0:
        outfmt = "Models evaluated: {}\n" + \
                "Ensemble fmax score: {:.4f}%, " + \
                "Ensemble AuPRC score: {:.4f}%"
        print(outfmt.format(", ".join(model_names), fmax_score_regular, auprc_score_regular))
        
        # Save predictions if requested
        if pred_savingpath is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(pred_savingpath), exist_ok=True)
            np.save(pred_savingpath, tp_tensor.cpu().numpy())
            print(f"Saved predictions to {pred_savingpath}")

def eval_multi(dataloader: DataLoader, model_list: list, no_eval: bool,
               permute_dims: T.Tuple[int, int, int, int] = (0, 3, 2, 1),
               do_weight_averaging: bool = False,
               model_class=None,
               model_opt=None):
    """
    Evaluate multiple models and create ensemble predictions.
    Also evaluates a weight-averaged model if requested.
    
    Args:
        dataloader: DataLoader for evaluation data
        model_list: List of models to evaluate
        no_eval: If True, skip evaluation metrics calculation
        permute_dims: Dimensions to permute in the input data
        do_weight_averaging: Whether to create and evaluate a weight-averaged model
        model_class: Class to use for creating the weight-averaged model
        model_args: Arguments to pass to the model_class constructor
        
    Returns:
        Tuple of (fmax_score, auprc_score, prediction_tensor, weight_avg_metrics)
    """
    print(f"Starting evaluation with {len(model_list)} models")
    Sig = torch.nn.Sigmoid()
    
    # Storage for all model predictions
    all_predictions = []
    all_model_states = []
    weight_avg_metrics = None
    targs = None
    
    # Process each model
    for i, model in enumerate(model_list):
        print(f"Evaluating model {i+1}/{len(model_list)}")
        
        # Store model state dictionary for weight averaging
        if do_weight_averaging:
            all_model_states.append(model.state_dict())
        
        preds_regular = []
        model_targs = []
        
        # Evaluate the model on all batches
        for batch_idx, (X, y) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            if isinstance(X, torch.Tensor):
                X = X.cuda()
            else:
                assert isinstance(X, list)
                proteins, X = X
                assert isinstance(X, torch.Tensor)
                X = X.cuda()
                # Check if the model has this method
                if hasattr(model, 'set_proteins'):
                    model.set_proteins(proteins)
            
            y = y.cuda()
            
            with torch.no_grad():
                with autocast(device_type="cuda"):
                    output_regular = Sig(model(X, permute_dims=permute_dims))
            
            preds_regular.append(output_regular.cpu().detach())
            model_targs.append(y.cpu().detach())
            
            # Clear memory
            model.zero_grad()
        
        # Aggregate predictions for this model
        y_score = torch.cat(preds_regular)
        
        # Store model predictions
        all_predictions.append(y_score)
        
        # Only need to store targets once
        if targs is None:
            targs = torch.cat(model_targs)
    
    # Create ensemble predictions by averaging
    y_true = targs
    y_score = torch.stack(all_predictions).mean(dim=0)
    
    # Calculate metrics for individual models
    individual_metrics = []
    if not no_eval:
        try:
            report = helper.evalperf_torch(
                y_true, y_score, 
                threshold=True,
                auprc=True,
                no_zero_classes=True
            )
            ensemble_fmax = report["fmax"]
            ensemble_auprc = report["auprc"]
            
            # Additional metrics for individual models
            if len(model_list) > 1:
                print("\nIndividual model performance:")
                for i, pred in enumerate(all_predictions):
                    model_report = helper.evalperf_torch(
                        y_true, pred,
                        threshold=True,
                        auprc=True,
                        no_zero_classes=True
                    )
                    individual_metrics.append({
                        'fmax': model_report['fmax'],
                        'auprc': model_report['auprc']
                    })
                    print(f"Model {i+1}: fmax={model_report['fmax']:.4f}, auprc={model_report['auprc']:.4f}")
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            ensemble_fmax, ensemble_auprc = -1, -1
    else:
        ensemble_fmax, ensemble_auprc = -1, -1
    
    # Create and evaluate weight-averaged model if requested
    if do_weight_averaging and len(all_model_states) > 1:
        try:
            print("\nCreating and evaluating weight-averaged model...")
            
            # Create averaged state dictionary
            avg_state_dict = {}
            for key in all_model_states[0].keys():
                # Check if this parameter exists in all models
                if all(key in state for state in all_model_states):
                    # Check if shapes match
                    if all(state[key].shape == all_model_states[0][key].shape for state in all_model_states):
                        avg_state_dict[key] = sum(state[key] for state in all_model_states) / len(all_model_states)
                    else:
                        print(f"Skipping parameter {key} due to shape mismatch")
                else:
                    print(f"Skipping parameter {key} as it's not present in all models")
            
            # Create a new model with averaged weights
            local_rank = model_list[0].device_ids[0]
            if model_class is not None and model_opt is not None:
                weight_avg_model = model_class.module(model_opt)
                weight_avg_model = nn.parallel.DistributedDataParallel(weight_avg_model,
                                                                       device_ids=[local_rank],
                                                                       output_device=local_rank)
                weight_avg_model.load_state_dict(avg_state_dict)
            else:
                # If model_class not provided, create a clone of the first model
                weight_avg_model = copy.deepcopy(model_list[0].module)
                weight_avg_model = nn.parallel.DistributedDataParallel(weight_avg_model,
                                                                       device_ids=[local_rank],
                                                                       output_device=local_rank)
                weight_avg_model.load_state_dict(avg_state_dict)
            
            weight_avg_model.eval()
            
            # Evaluate the weight-averaged model
            preds_weight_avg = []
            
            for batch_idx, (X, y) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
                if isinstance(X, torch.Tensor):
                    X = X.cuda()
                else:
                    assert isinstance(X, list)
                    proteins, X = X
                    assert isinstance(X, torch.Tensor)
                    X = X.cuda()
                    if hasattr(weight_avg_model, 'set_proteins'):
                        weight_avg_model.set_proteins(proteins)
                
                with torch.no_grad():
                    with autocast(device_type="cuda"):
                        output = Sig(weight_avg_model(X, permute_dims=permute_dims))
                
                preds_weight_avg.append(output.cpu().detach())
                weight_avg_model.zero_grad()
            
            weight_avg_preds = torch.cat(preds_weight_avg)
            
            if not no_eval:
                weight_avg_report = helper.evalperf_torch(
                    y_true, weight_avg_preds,
                    threshold=True,
                    auprc=True,
                    no_zero_classes=True
                )
                
                weight_avg_fmax = weight_avg_report["fmax"]
                weight_avg_auprc = weight_avg_report["auprc"]
                
                print(f"Weight-averaged model: fmax={weight_avg_fmax:.4f}, auprc={weight_avg_auprc:.4f}")
                
                weight_avg_metrics = {
                    'fmax': weight_avg_fmax,
                    'auprc': weight_avg_auprc,
                    'predictions': weight_avg_preds
                }
                
                # Determine the best fmax and auprc across all models including weight-averaged
                if weight_avg_fmax > ensemble_fmax:
                    ensemble_fmax = weight_avg_fmax
                    print(f"Weight-averaged model has the best fmax score: {weight_avg_fmax:.4f}")
                
                if weight_avg_auprc > ensemble_auprc:
                    ensemble_auprc = weight_avg_auprc
                    print(f"Weight-averaged model has the best auprc score: {weight_avg_auprc:.4f}")
                
                # Save the weight-averaged model state if needed
                # torch.save(avg_state_dict, "weight_averaged_model.pth")
        
        except Exception as e:
            print(f"Error during weight-averaged model evaluation: {str(e)}")
            traceback.print_exc()
    
    return ensemble_fmax, ensemble_auprc, torch.stack([y_true.cpu(), y_score.cpu()], dim=0), weight_avg_metrics