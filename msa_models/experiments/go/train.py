import typing as T
from argparse import Namespace
import json
from time import strftime, localtime
import pickle
import sys
import os
import functools
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch.multiprocessing as mp

prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if prj_dir not in sys.path:
    sys.path.append(prj_dir)

import torch.amp as amp
import torch.amp.grad_scaler as grad_scaler
import torch.amp.autocast_mode as autocast_mode

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.utils as nn_utils
import torch.utils.data as utils_data
from torch.utils.data.distributed import DistributedSampler
import torch.nn.parallel as nn_parallel
from torch.optim import lr_scheduler, optimizer
from functools import reduce

from models import Arch
from models.utils import to_np
import helper_functions.helper as helper
from loss_functions.loss import AsymmetricLoss, AsymmetricLossOptimized

from loss_functions.loss import FocalLossV2 as FocalLoss
import optimizers.optim as myoptim

import timm.optim as tiop

# from experiments.go.GO import MSADataset
import experiments.msa as D

import helper_functions.aug as aug

import wandb

Tensor = torch.Tensor

name2loss = dict(
    CE=BCEWithLogitsLoss(),
    Focal=FocalLoss(),
    # ASL=AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    ASL=AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
)

name2optim = {
    'Adam': functools.partial(torch.optim.Adam, eps=1e-4, weight_decay=0),
    'SGD': functools.partial(torch.optim.SGD, momentum=0.9, weight_decay=0),
    'AMSGrad': functools.partial(torch.optim.Adam, eps=1e-4, weight_decay=0, amsgrad=True),
    'NAdam': functools.partial(torch.optim.NAdam, eps=1e-4, weight_decay=0.),
    'NAdamW': functools.partial(tiop.create_optimizer_v2, opt="nadamw", eps=1e-4, weight_decay=0),
    'AdamW': functools.partial(torch.optim.AdamW, eps=1e-3, weight_decay=0, amsgrad=True),
    'CAdamW': functools.partial(tiop.create_optimizer_v2, opt="adamw", eps=1e-4, weight_decay=0.,
                                caution=True),
    'Mars': functools.partial(tiop.create_optimizer_v2, opt="mars", eps=1e-4, weight_decay=0.,),
    'CMars': functools.partial(tiop.create_optimizer_v2, opt="mars", eps=1e-4, weight_decay=0.,
                               caution=True),
    'Lamb': functools.partial(tiop.create_optimizer_v2, opt="Lamb", eps=1e-6, weight_decay=0.,),
    'Lookahead_AMSGrad': functools.partial(tiop.create_optimizer_v2, opt="Lookahead_Adam", eps=1e-4, weight_decay=0, amsgrad=True)
}

def get_scheduler(n: str, 
                  optim,
                  **kwargs):
  match n:
    case "cycle":
      assert kwargs.get("max_lr") is not None
      assert kwargs.get("steps_per_epoch") is not None
      assert kwargs.get("pct_start") is not None
      assert kwargs.get("epochs") is not None
      assert kwargs.get("three_phase") is not None
      assert kwargs.get("final_div_factor") is not None
      return lr_scheduler.OneCycleLR(optim, 
                                     max_lr=kwargs["max_lr"],
                                     steps_per_epoch=kwargs["steps_per_epoch"],
                                     epochs=kwargs["epochs"],
                                     pct_start=kwargs["pct_start"],
                                     three_phase=kwargs["three_phase"],
                                     final_div_factor=kwargs["final_div_factor"])
    case "multi":
      assert kwargs.get("milestones") is not None
      assert kwargs.get("gamma") is not None
      return lr_scheduler.MultiStepLR(optim,
                                      milestones=kwargs["milestones"],
                                      gamma=kwargs["gamma"])
    case _:
      raise NotImplementedError(f"no impelement for {n}")

def validate_multi(val_loader, model, ema_model,
                   no_amp=False,
                   permute_dims: T.Tuple[int, int, int, int] = (0, 3, 2, 1)):
    """
    Validation function that correctly handles distributed evaluation.
    Gathers predictions from all processes before computing metrics.
    """
    # Get distributed training info
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    if local_rank == 0:
        print("Starting validation across", world_size, "processes")
    
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targs = []
    
    # Set validation mode
    model.eval()
    ema_model.eval()
    
    # Set dataloader to validation mode
    val_loader.sampler.set_epoch(0)  # Use consistent sampling for validation
    
    # Process batches
    for _, (X, y) in enumerate(val_loader):
        if isinstance(X, torch.Tensor):
            X = X.cuda()
        else:
            assert isinstance(X, T.List)
            proteins, X = X
            assert isinstance(X, torch.Tensor)
            X = X.cuda()
            # Set proteins only if the model has this method
            if hasattr(model, 'set_proteins'):
                model.set_proteins(proteins)
            if hasattr(ema_model.module, 'set_proteins'):
                ema_model.module.set_proteins(proteins)
        
        y = y.float().cuda()
        
        # compute output
        with torch.no_grad():
            if not no_amp:
                with amp.autocast(device_type="cuda"):
                    output_regular = Sig(model(X, permute_dims=permute_dims))
                    output_ema = Sig(ema_model.module(X, permute_dims=permute_dims))
            else:
                output_regular = Sig(model(X, permute_dims=permute_dims))
                output_ema = Sig(ema_model.module(X, permute_dims=permute_dims))

        # Store predictions and targets
        preds_regular.append(output_regular.detach())
        preds_ema.append(output_ema.detach())
        targs.append(y.detach())
    
    # Concatenate local predictions and targets
    local_Y = torch.cat(targs)
    local_reg = torch.cat(preds_regular)
    local_ema = torch.cat(preds_ema)
    
    # Gather predictions and targets from all processes
    gathered_Y = [torch.zeros_like(local_Y) for _ in range(world_size)]
    gathered_reg = [torch.zeros_like(local_reg) for _ in range(world_size)]
    gathered_ema = [torch.zeros_like(local_ema) for _ in range(world_size)]
    
    # Gather all results
    dist.all_gather(gathered_Y, local_Y)
    dist.all_gather(gathered_reg, local_reg)
    dist.all_gather(gathered_ema, local_ema)
    
    # Compute metrics on concatenated results from all processes (only on rank 0)
    if local_rank == 0:
        # Concatenate results from all processes
        all_Y = torch.cat(gathered_Y).cpu()
        all_reg = torch.cat(gathered_reg).cpu()
        all_ema = torch.cat(gathered_ema).cpu()
        
        # Calculate metrics
        report_reg = helper.evalperf_torch(all_Y, all_reg, threshold=True, auprc=True,
                                           no_zero_classes=True)
        report_ema = helper.evalperf_torch(all_Y, all_ema, threshold=True, auprc=True,
                                           no_zero_classes=True)
        
        f1 = report_reg["fmax"]
        a1 = report_reg["auprc"]
        f2 = report_ema["fmax"]
        a2 = report_ema["auprc"]
        
        print("fmax score regular {:.4f}, fmax score EMA {:.4f}".format(f1, f2))
        print("AuPRC score regular {:.4f}, AuPRC score EMA {:.4f}".format(a1, a2))
        
        # Log to wandb
        wandb.log({"fmax score regular": f1, "fmax score EMA": f2})
        wandb.log({"AuPRC score regular": a1, "AuPRC score EMA": a2})
        
        best_fmax = f2 if f2 > f1 else f1
        best_auprc = a2 if a2 > a1 else a1
    else:
        # Initialize with placeholder values for non-rank-0 processes
        best_fmax = 0.0
        best_auprc = 0.0
    
    # Broadcast results to all processes so they all return the same values
    best_fmax_tensor = torch.tensor([best_fmax], device=f"cuda:{local_rank}")
    best_auprc_tensor = torch.tensor([best_auprc], device=f"cuda:{local_rank}")
    
    dist.broadcast(best_fmax_tensor, src=0)
    dist.broadcast(best_auprc_tensor, src=0)
    
    return best_fmax_tensor.item(), best_auprc_tensor.item()


def training_prog(opt: Namespace, local_rank, world_size, is_optuna_trial=False):
    """
    Setup for multi-node distributed training
    """
    # Initialize distributed training if not already done
    if not is_optuna_trial:
        # Get distributed training details from environment variables
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
    
    # Set device
    torch.cuda.set_device(local_rank)
    torch.manual_seed(3407)
    # Initialize model
    model = Arch(opt)
    first_param = next(model.parameters())
    print(first_param.device)

    if opt.for_retrain is not None:
        trained_state_dict = torch.load(opt.for_retrain)
        unloaded_state_dict = model.state_dict()
        for k, v in trained_state_dict.items():
            if k in unloaded_state_dict:
                unloaded_state_dict[k] = v
        model.load_state_dict(unloaded_state_dict)
        if not opt.full_fine_tuning:
            model.freeze_all_except_fc()

        model.train()
        print("Re-train")

    if local_rank == 0 and not is_optuna_trial:
        timestamp = strftime("%y%m%d%H%M%S", localtime())
        option_path = os.path.join(opt.model_saving,
                                "training_option_{}.pkl".format(timestamp))
        with open(option_path, "wb") as h:
            pickle.dump(opt.__dict__, h)
        
        # save as json
        option_path = option_path.removesuffix(".pkl") + ".json"
        with open(option_path, "w") as h:
            json.dump(opt.__dict__, h)

    if opt.torch_compile:
      assert hasattr(torch, "compile"), "need pytorch > 2.0"
      model = torch.compile(model)
    model = model.to(f"cuda:{local_rank}")
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], 
                                                output_device=local_rank,
                                                )
    # Initialize EMA model with the specified decay
    ema = helper.ModelEma(model, opt.ema_decay)


    # Initialize the dataset and dataloaders
    dataset = D.MSADataset(opt.file_address, opt.working_dir, opt.mode, opt.task,
                           opt.num_classes, opt.top_k, opt.max_len, 
                           need_proteins=False,
                           msa_max_size=opt.msa_max_size)
    # Setup dataloaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=opt.shuffle)
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        num_workers=opt.dataloader_num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if opt.extra_val is not None or opt.validation_split != 0:
        # Handle validation dataset
        if opt.extra_val is not None:
            val_dataset = D.MSADataset(opt.file_address, opt.working_dir, opt.extra_val, opt.task,
                                       opt.num_classes, opt.top_k, opt.max_len,
                                       need_proteins=False)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=opt.batch_size,
                sampler=val_sampler,
                num_workers=opt.dataloader_num_workers,
                pin_memory=True
            )
        else:
            # Split the dataset for validation if no extra_val is provided
            validation_split = opt.validation_split
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))
            if opt.shuffle:
                np.random.seed(42)
                np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]
            
            # Create subset samplers
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)
            
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, shuffle=False)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, shuffle=False)
            
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=opt.batch_size,
                sampler=train_sampler,
                num_workers=opt.dataloader_num_workers,
                pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=opt.batch_size,
                sampler=valid_sampler,
                num_workers=opt.dataloader_num_workers,
                pin_memory=True
            )

    # Set optimizer with the specified weight_decay
    steps_per_epoch = len(train_loader) // opt.accum_steps
    option_dict = opt.__dict__
    loss_func = name2loss[option_dict.get("loss", "ASL")]
    parameters = helper.add_weight_decay(model, opt.weight_decay) \
        if not opt.no_model_weight_decay else model.parameters()
    
    # Initialize optimizer with the specified learning rate
    optimizer = name2optim[opt.optim](parameters, lr=opt.lr, eps=opt.optim_eps,
                                    weight_decay=0 if not opt.no_model_weight_decay else opt.weight_decay)

    smoothing = float(smth) \
        if (smth := getattr(opt, "smoothing", None)) is not None else 0.

    # Configure scheduler with the specified lr_pct_start
    if opt.lr_policy == "cycle":
        scheduler = get_scheduler(opt.lr_policy,
                                optimizer,
                                max_lr=opt.lr,
                                steps_per_epoch=steps_per_epoch,
                                epochs=opt.epochs,
                                pct_start=opt.lr_pct_start,
                                three_phase=opt.lr_cycle_three_phase,
                                final_div_factor=opt.lr_final_div_factor,)
    elif opt.lr_policy == "multi":
        milestones = [n * steps_per_epoch for n in opt.milestones]
        scheduler = get_scheduler(opt.lr_policy,
                                optimizer,
                                milestones=milestones,
                                gamma=0.1)
    else:
        raise NotImplementedError(f"no implement for {opt.lr_policy}")

    highest_fmax, highest_AuPRC = 0, 0
    highest_result = (0,0)
    scaler = amp.GradScaler() if not opt.no_amp else None

    if local_rank == 0 and not is_optuna_trial:
        timestamp = strftime("%y%m%d%H%M%S", localtime())
        saving_info_path = os.path.join(opt.model_saving, f"training_info_{timestamp}.pkl")
        multi_node = f"multi_node{rank}"
        params_str = f"lr{opt.lr}-pct_start{opt.lr_pct_start}"\
            + f"-wd{opt.weight_decay}-ema{opt.ema_decay}"
        wandb_name = f"{multi_node}-{opt.netD}-{params_str}-{opt.task}"
        wandb.init(project=f'{opt.wandb}', name=f'{wandb_name}',
                   group="ddp_training", config=opt,
                   settings=wandb.Settings(init_timeout=1000))
    elif local_rank == 0 and is_optuna_trial:
        # For Optuna trials, we'll log the hyperparameters
        wandb.log({
            "lr": opt.lr,
            "lr_pct_start": opt.lr_pct_start,
            "weight_decay": opt.weight_decay,
            "ema_decay": opt.ema_decay
        })

    # Initialize variables for tracking best models
    k = opt.top_k_models  # Number of top models to save
    top_combined = []
    top_fmax = []
    top_auprc = []
    current_epoch = 1

    accum_steps = opt.accum_steps  # gradient accumulation

    trainInfoList = []
    print_interval = 100
    log_interval = 10
    permute_dims = getattr(opt, "permute_dims", (0,3,2,1))
    start_time = time.time()

    # Initialize at the beginning of epoch
    optimizer.zero_grad()
    
    # Training loop
    for epoch in range(1, opt.epochs+1):
        if epoch > opt.stop_epoch:
            break

        if local_rank == 0:
            print(f"Starting Epoch {epoch}/{opt.epochs}")
        iter_cnt = 0
        epoch_start_time = time.time()
        
        # Set train loader's epoch for distributed sampler
        train_loader.sampler.set_epoch(epoch)
        
        for i, (inputData, target) in enumerate(train_loader):
            if isinstance(inputData, torch.Tensor):
                inputData = inputData.cuda()
            else:
                proteins, inputData = inputData
                inputData = inputData.cuda()
                model.module.set_proteins(proteins)
                ema.module.set_proteins(proteins)
            target = target.float().cuda()

            target = aug.mixup_target(target, num_classes=opt.num_classes,
                                    smoothing=smoothing)

            if not opt.no_amp:
                with amp.autocast(device_type="cuda"):
                    output, target_mix = model(inputData, permute_dims=permute_dims,
                                              aug_params={"y": target,
                                                         "mixup_alpha": opt.mixup_alpha})
                    loss = loss_func(output, target_mix)
            else:
                output, target_mix = model(inputData, permute_dims=permute_dims,    
                                          aug_params={"y": target,
                                                     "mixup_alpha": opt.mixup_alpha})
                loss = loss_func(output, target_mix)

            # Scale loss for gradient accumulation
            loss = loss / accum_steps

            # Handle AMP and gradient accumulation properly
            if not opt.no_amp:
                # With mixed precision
                scaler.scale(loss).backward()
                
                if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Update scheduler after optimizer step
                    scheduler.step()
                    
                    # Update EMA after optimizer step
                    ema.update(model)
            else:
                # Without mixed precision
                loss.backward()
                
                if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Update scheduler after optimizer step
                    scheduler.step()
                    
                    # Update EMA after optimizer step
                    ema.update(model)

            if local_rank == 0:
                # store information
                if i % print_interval == 0:
                    trainInfoList.append([epoch, i, loss.item()])
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.4e}, Loss: {:.8f}'
                        .format(epoch, opt.epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                                scheduler.get_last_lr()[0], \
                                loss.item()))

                if i % log_interval == 0:
                    wandb.log({"Learing Rate": scheduler.get_last_lr()[0],
                            "Loss": to_np(loss)})

            iter_cnt += 1

        epoch_end_time = time.time()
        epoch_total_time = epoch_end_time - epoch_start_time
        avg_step_time = epoch_total_time / iter_cnt 
        if local_rank == 0:
            print("epoch_total_time:", epoch_total_time)
            print("step:", len(train_loader))
            print("iter_cnt:", iter_cnt)
            print(f'Average time per step in Epoch {epoch}: {avg_step_time:.4f} seconds')
        
        # Validation
        if val_loader is not None:
            model.eval()
            fmax_score, AuPRC_score = validate_multi(val_loader, model, ema, 
                                                    no_amp = opt.no_amp, 
                                                    permute_dims=permute_dims)
            model.train()
            
            # For Optuna trials, we focus on updating validation metrics
            if is_optuna_trial:
                if fmax_score > highest_fmax:
                    highest_fmax = fmax_score
                if AuPRC_score > highest_AuPRC:
                    highest_AuPRC = AuPRC_score
            else:
                # Inside training loop for regular (non-Optuna) training
                if local_rank == 0:
                    # Handle combined score (fmax and AuPRC)
                    combined_entry = (fmax_score, AuPRC_score, current_epoch)
                    
                    # Define comparison function for combined score
                    def is_better_combined(new, old):
                        return new[0] > old[0] and new[1] > old[1]
                    
                    added, rank = update_top_k(top_combined, combined_entry, is_better_combined, k)
                    if added:
                        save_state_dict(model, os.path.join(opt.model_saving, 
                                                            f'model-highest-rank{rank}.pth'))
                        save_state_dict(ema.module, os.path.join(opt.model_saving, 
                                                                f'ema-model-highest-rank{rank}.pth'))
                    
                    # Handle fmax score
                    fmax_entry = (fmax_score, current_epoch)
                    
                    def is_better_fmax(new, old):
                        return new[0] > old[0]
                    
                    added, rank = update_top_k(top_fmax, fmax_entry, is_better_fmax, k)
                    if added:
                        save_state_dict(model, os.path.join(opt.model_saving, 
                                                            f'fmax-highest-rank{rank}.pth'))
                        save_state_dict(ema.module, os.path.join(opt.model_saving, 
                                                                f'ema-fmax-highest-rank{rank}.pth'))
                    
                    # Handle AuPRC score
                    auprc_entry = (AuPRC_score, current_epoch)
                    
                    def is_better_auprc(new, old):
                        return new[0] > old[0]
                    
                    added, rank = update_top_k(top_auprc, auprc_entry, is_better_auprc, k)
                    if added:
                        save_state_dict(model, os.path.join(opt.model_saving, 
                                                            f'AuPRC-highest-rank{rank}.pth'))
                        save_state_dict(ema.module, os.path.join(opt.model_saving, 
                                                                f'ema-AuPRC-highest-rank{rank}.pth'))
                    
                    # Print current status
                    print(f'Top {k} combined scores: {top_combined}')
                    print(f'Top {k} fmax scores: {top_fmax}')
                    print(f'Top {k} AuPRC scores: {top_auprc}')
                    
                    # Save current model
                    save_state_dict(model, os.path.join(opt.model_saving, "model-last.pth"))
                    save_state_dict(ema.module, os.path.join(opt.model_saving, "ema-model-last.pth"))
        
        current_epoch += 1

    # Clean up after training
    if local_rank == 0 and not is_optuna_trial:
        with open(saving_info_path, "wb") as h:
            pickle.dump(trainInfoList, h)
        wandb.finish()
        
    total_time = time.time() - start_time
    if local_rank == 0:
        print(f'Total training time: {total_time:.4f} seconds')

    # Return metrics for Optuna
    return highest_fmax, highest_AuPRC

def save_state_dict(model: T.Union[Arch, nn_parallel.DistributedDataParallel], saving_path: str):
    torch.save(model.state_dict(), saving_path)

def update_top_k(top_list, new_entry, is_better_func, k):
    """Update a top-k list with a new entry.
    
    Args:
        top_list: List of entries to update
        new_entry: New entry to potentially add
        is_better_func: Function that takes (new, old) entries and returns True if new is better
        k: Maximum number of entries to keep
        
    Returns:
        (added, rank): Tuple of whether entry was added and its rank (0 if not added)
    """
    added = False
    rank = 0
    
    # Check if new_entry is better than any existing entry
    for i, old_entry in enumerate(top_list):
        if is_better_func(new_entry, old_entry):
            top_list.insert(i, new_entry)
            added = True
            if len(top_list) > k:
                top_list.pop()  # Remove the worst entry
            break
    
    # If not added and list not full, append
    if not added and len(top_list) < k:
        top_list.append(new_entry)
        added = True
    
    # If added, determine the rank
    if added:
        rank = top_list.index(new_entry) + 1
    
    return added, rank

def freeze_msa_encoder(model: Arch):
  """freeze the msa encoder and gnet layers in architecture"""
  assert isinstance(model, Arch)
  assert isinstance(model.pre_model, nn.Module)
  assert isinstance(model.gnet, nn.Module)

  for param in model.pre_model.parameters():
      param.requires_grad = False

  for param in model.gnet.parameters():
      param.requires_grad = False
  print("-------------------------------------------------------------")
  print("#############################################################")
  print("freezed the pre_model and gnet")
  print("#############################################################")
  print("-------------------------------------------------------------")