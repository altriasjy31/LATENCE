import optuna
from optuna.trial import Trial
from argparse import Namespace
import os
import torch
import torch.distributed as dist
from experiments.go.train import training_prog
import wandb
from time import strftime, localtime
import json
import gc

# Optuna objective function for hyperparameter optimization
def objective(trial, base_opt, local_rank, world_size):
    """Define the objective function for Optuna to optimize."""
    # Create a copy of the base options to avoid modifying the original
    opt = Namespace(**vars(base_opt))
    
    # Only suggest hyperparameters on rank 0
    if local_rank == 0:
        # Suggest values for hyperparameters to optimize
        # opt.lr = trial.suggest_float("learning_rate", 0.001, 0.005)
        opt.lr = trial.suggest_float("learning_rate", 0.0005, 0.002)
        # opt.lr_pct_start = trial.suggest_categorical("lr_pct_start", [0.025, 0.05, 0.1, 0.15, 0.2])
        opt.weight_decay = trial.suggest_float("weight_decay", 0.005, 0.05)
        # opt.ema_decay = trial.suggest_float("ema_decay", 0.99, 0.999)
        
        # Log the trial parameters
        print(f"Trial #{trial.number}:")
        print(f"  Learning Rate: {opt.lr}")
        print(f"  LR Percent Start: {opt.lr_pct_start}")
        print(f"  Weight Decay: {opt.weight_decay}")
        print(f"  EMA Decay: {opt.ema_decay}")
    
    # Broadcast hyperparameters from rank 0 to all other processes
    if world_size > 1:
        hp_tensor = torch.tensor([opt.lr, opt.lr_pct_start, opt.weight_decay, opt.ema_decay], 
                               device=f"cuda:{local_rank}",
                               dtype=torch.float16)
        dist.broadcast(hp_tensor, src=0)
        
        # Update options with broadcast values on other ranks
        if local_rank != 0:
            opt.lr = hp_tensor[0].item()
            opt.lr_pct_start = hp_tensor[1].item()
            opt.weight_decay = hp_tensor[2].item()
            opt.ema_decay = hp_tensor[3].item()
    
    # Run training with suggested hyperparameters
    best_fmax, best_auprc = training_prog(opt, local_rank, world_size, is_optuna_trial=True)
    
    # Return the metric to optimize (using fmax score as the primary metric)
    return 2*best_fmax*best_auprc / (best_fmax + best_auprc)

def run_optuna_optimization(base_opt, local_rank, world_size, n_trials=20):
    """Run Optuna optimization to find the best hyperparameters."""
    # Create and run the study only on rank 0
    if local_rank == 0:
        # Set up wandb for this optimization run
        timestamp = strftime("%y%m%d%H%M%S", localtime())
        wandb_name = f"optuna_search_{timestamp}"
        wandb.init(project=f'{base_opt.wandb}', name=wandb_name,
                   group="optuna_search",)
        
        # Create the study
        study = optuna.create_study(direction="maximize", 
                                   study_name=f"optuna_hp_search_{timestamp}")
        
        # Optimize with the objective function
        study.optimize(lambda trial: objective(trial, base_opt, local_rank, world_size), 
                      n_trials=n_trials,
                      gc_after_trial=True)
        
        # Print and save the best parameters
        print("\n" + "="*50)
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value ( Best 2*Fmax*AUPR/(Fmax+AUPR) ): {trial.value:.6f}")
        print("  Parameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # Save best parameters to file
        best_params_path = os.path.join(base_opt.model_saving, f"best_params_{timestamp}.json")
        with open(best_params_path, "w") as f:
            json.dump(trial.params, f, indent=2)
        
        print(f"Best parameters saved to: {best_params_path}")
        print("="*50)
        
        # Log best parameters to wandb
        wandb.log({"best_fmax": trial.value})
        for key, value in trial.params.items():
            wandb.log({f"best_{key}": value})
            
        wandb.finish()
        
        return trial.params
    else:
        # Non-master processes just participate in the trials
        for i in range(n_trials):
            objective(None, base_opt, local_rank, world_size)
        return None
