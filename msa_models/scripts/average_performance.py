import sys
import os

prj_dir = os.path.dirname(os.path.dirname(__file__))
if not prj_dir in sys.path:
    sys.path.append(prj_dir)

import experiments.msa as D
import models.gendis as G
import helper_functions.helper as helper
from models.utils import parsing

import typing as T
import numpy as np
import argparse as P
import tqdm
import torch
import torch as th
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as ud
import torch.amp.autocast_mode as am

def eval(opt: P.Namespace, local_rank, world_size):
    """
    Evaluation function supporting both multi-model ensembles and multi-sampling evaluation.
    
    Features:
    - Can evaluate multiple models specified as comma-separated filenames
    - Supports multiple sampling passes for each model
    - Calculates metrics for individual models and for the ensemble
    - Saves predictions optionally
    """
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    torch.cuda.set_device(f"cuda:{local_rank}")
    torch.manual_seed(3407)
    batch_size = opt.batch_size

    # Load dataset
    dataset = D.MSADataset(opt.file_address, opt.working_dir, opt.mode, opt.task,
        opt.num_classes, opt.top_k, opt.max_len,
        need_proteins=False,
        msa_max_size=opt.msa_max_size)

    # Setup dataloader
    eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=opt.shuffle)
    eval_loader = ud.DataLoader(dataset, batch_size=batch_size,
                           sampler=eval_sampler,
                           num_workers=opt.dataloader_num_workers,
                           pin_memory=True)
    
    # Parse model paths
    model_dir = opt.model_saving
    model_paths = [os.path.join(model_dir, name.strip()) 
                  for name in opt.trained_model.split(",")]
    
    # Prepare variables for predictions
    all_model_predictions = []  # Store predictions from each model
    targs = None  # Will store targets (only need to collect once)
    
    if local_rank == 0:
        print(f"Evaluating {len(model_paths)} models with {opt.num_samplings} sampling(s) each")
    
    Sig = torch.nn.Sigmoid()
    
    # Process each model
    for model_idx, model_path in enumerate(model_paths):
        if local_rank == 0:
            print(f"\nModel {model_idx+1}/{len(model_paths)}: {os.path.basename(model_path)}")
        
        try:
            # Initialize model
            model = G.Arch(opt)
            
            if opt.torch_compile:
                assert hasattr(torch, "compile"), "need pytorch > 2.0"
                model = torch.compile(model)
            
            model = model.to(f"cuda:{local_rank}")
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank)
            
            # Load model weights
            loaded_state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict=loaded_state)
            model.eval()
            
            # Initialize predictions for this model
            model_preds = th.tensor(0)
            
            # Do multiple sampling passes for this model
            for j in range(opt.num_samplings):
                if local_rank == 0:
                    print(f"  Sampling: {j+1}/{opt.num_samplings}")
                
                sample_preds = []
                sample_targs = []
                
                # Process all batches
                for i, (X, y) in tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader)):
                    if isinstance(X, torch.Tensor):
                        X = X.cuda()
                    else:
                        assert isinstance(X, list)
                        proteins, X = X
                        assert isinstance(X, torch.Tensor)
                        X = X.cuda()
                        # Only set proteins if model has this method
                        if hasattr(model, 'set_proteins'):
                            model.set_proteins(proteins)
                    
                    y = y.cuda()
                    
                    # Compute predictions
                    with torch.no_grad():
                        with am.autocast(device_type="cuda"):
                            output_regular = Sig(model(X)).cpu()
                    
                    # Store predictions
                    sample_preds.append(output_regular.detach())
                    sample_targs.append(y.cpu().detach())
                    
                    # Clear memory
                    model.zero_grad()
                    torch.cuda.empty_cache()
                
                # Combine batch predictions for this sampling
                full_sample_preds = torch.cat(sample_preds)

                # On first sampling of first model, collect targets
                if targs is None and j == 0:
                    targs = torch.cat(sample_targs)
              
                # For first sampling, initialize model_preds
                if j == 0:
                    model_preds = full_sample_preds
                else:
                    # Add to accumulated predictions
                    model_preds += full_sample_preds
            
            # Average predictions across samplings
            model_preds = model_preds / opt.num_samplings
            
            # Store this model's predictions
            all_model_predictions.append(model_preds)
            
            # Calculate and report per-model metrics
            if local_rank == 0:
                report = helper.evalperf_torch(targs, model_preds, auprc=True)
                print(f"  Model {model_idx+1} metrics: Fmax={report['fmax']:.4f}, AuPRC={report['auprc']:.4f}")
        
        except Exception as e:
            print(f"Error processing model {model_path}: {str(e)}")
            continue
    
    # Calculate ensemble predictions (average across all models)
    if all_model_predictions:
        ensemble_preds = torch.stack(all_model_predictions).mean(dim=0)
        
        # Calculate ensemble metrics
        ensemble_report = helper.evalperf_torch(targs, ensemble_preds, auprc=True)
        
        if local_rank == 0:
            print("\n===== ENSEMBLE RESULTS =====")
            print(f"Models: {opt.trained_model}")
            print(f"Samplings per model: {opt.num_samplings}")
            print(f"Ensemble Fmax: {ensemble_report['fmax']:.4f}")
            print(f"Ensemble AuPRC: {ensemble_report['auprc']:.4f}")
            
            # Save predictions if requested
            if opt.prediction_save:
                prediction_tensor = torch.stack([targs, ensemble_preds], dim=0)
                os.makedirs(os.path.dirname(opt.prediction_save), exist_ok=True)
                np.save(opt.prediction_save, prediction_tensor.numpy())
                print(f"Saved ensemble predictions to {opt.prediction_save}")
    else:
        print("No models were successfully evaluated.")


def main():
    parser = P.ArgumentParser()
    # positional arguments
    parser.add_argument("file_address", metavar="dataset_path")
    parser.add_argument("working_dir", help="the msa data directory")
    parser.add_argument("model_saving", help="the path for model saving")
    # config filepath
    parser.add_argument("-c", "--config", type=str,
                      help="config filepath")
    parser.add_argument("-n", "--num-samplings", type=int, default=1,
                      help="Number of sampling passes per model")
    parser.add_argument("--trained_model", type=str, default="model-highest.pth",
                      help="Comma-separated list of model filenames to evaluate")
    
    opt = parsing(parser)

    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f"world_size:{world_size}, local_rank:{local_rank}")

    eval(opt, local_rank, world_size)

if __name__ == "__main__":
    main()