import sys
import os
import torch
prj_dir = os.path.dirname(os.path.dirname(__file__))
if not prj_dir in sys.path:
    sys.path.append(prj_dir)

import typing as T
from models.utils import parsing
from experiments.go.train import training_prog
# from experiments.go.exp_train import exp_training_prog
from experiments.go.eval import evaluation_prog
from experiments.preprocess.utils import get_surfix
from argparse import ArgumentParser, Namespace
from experiments.go.run_training import run_optuna_optimization
import torch.distributed as dist

def main():
    """
    # Dataset Parameters
    dataset_path: str,
    working_dir: str,
    n_classes: int,
    mode: str, # [train, test]
    task: str, # [cc, mf, bp]
    file_type:str, # file surfix like npy
    top_k: int, # the number of top sequence in msa for selection
    max_len: int, # the max length of sequence

    # MSAModel Parameters
    output_nc, # the number of output channels
    ngf: int, # the number of encoder last layer filters
    netG: str, # the type of encoder name
    n_layers: int, # the numger of cnn decoder layers
    no_dropout: bool,
    norm: str, # the norm type
    init_type: str,
    init_gain: float,

    # Training Parameters
    epochs: int,
    lr: float,
    batch_size: int,
    validation_split: float,
    shuffle_dataset: bool
    model_saving: str
    """
    parser = ArgumentParser(description='GO prediction: GenDis model construction')

    # positional arguments
    parser.add_argument("file_address", metavar="dataset_path")
    parser.add_argument("working_dir", help="the msa data directory")
    parser.add_argument("model_saving", help="the path for model saving")

    # config filepath
    parser.add_argument("-c", "--config", type=str,
                        help="config filepath")

    # optional arguments
    parser.add_argument("--in-channels", dest="in_channels", type=int, default=21)
    parser.add_argument("--out-channels-G", dest="out_channels_G", type=int, default=21)
    parser.add_argument("--num-classes",dest="num_classes", type=int, default=19939)
    parser.add_argument("--mode", choices=["train","test"], default="train")
    parser.add_argument("--task",choices=["cellular_component",
                                          "molecular_function",
                                          "biological_process"], default="biological_process")
    parser.add_argument("--top-k",dest="top_k",type=int,default=40,
                        help="select the top k sequence ib msa")
    parser.add_argument("--max-len",dest="max_len", type=int, default=2000,
                        help="the max lenght of sequence for using")
    parser.add_argument("--dataloader-num-workers", dest="dataloader_num_workers", type=int, default=5,
                        help="the number of workers that dataloader will need")

    parser.add_argument("--msa-embedding-dim", dest="msa_embedding_dim", type=int, default=21,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-encoding-strategy", dest="msa_encoding_strategy", type=str, 
                        choices=["one_hot", "emb", "emb_plus_one_hot", "emb_plus_pssm","fast_dca"], 
                        default="emb_plus_one_hot",
                        help="parameters for encoding the msa file")
    parser.add_argument('--ngf', type=int, default=64, 
                        help='# of gen filters in the last conv layer')
    parser.add_argument('--netG', type=str, default='resnet_9blocks', 
                        help="specify generator architecture " +\
                        "[resnet_9blocks | resnet_6blocks | resnet_4blocks | " +
                        "resnet_2blocks | resnet_oneblock | none]")
    parser.add_argument("--no-antialias", dest="no_antialias", action="store_true",
                        help="use dilated convolution blocks in generator")
    parser.add_argument("--no-antialias-up", dest="no_antialias_up", action="store_true",
                        help="use dilated convolution_transposed blocks in generator")
    parser.add_argument('--ndf', type=int, default=64, 
                            help='# of dis filters in the last conv layer')
    parser.add_argument('--netD', type=str, default='resnet50', 
                        help='specify discriminator architecture')
    parser.add_argument("--dilation", nargs=3, dest="replace_stride_with_dilation", 
                        default=[False, False, False], type=bool,
                        help="using dilation to replace the stride in resnet")

    parser.add_argument('--no-dropout', dest="no_dropout",action='store_true', 
                        help='no dropout for the generator')
    parser.add_argument('--normG', type=str, default='instance',
                        help='for generator, instance normalization or batch normalization [instance | batch | none]')

    parser.add_argument("--input-size", nargs=2, default=[2000,40], type=int)    
    parser.add_argument("--permute-dims", nargs=4, default=[0,3,2,1], type=int)

    parser.add_argument('--init-type', dest="init_type", type=str, default='normal', 
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init-gain', dest="init_gain", type=float, default=0.02, 
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument("--loss", type=str, default="ASL",
                        help="the loss function for training [CE | Focal | ASL]")
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--optim-eps", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--no-model-weight-decay", action="store_true",
                        help="using the weight_decay of optim instead of using 'add_weight_decay' for model")
    parser.add_argument("--ema-decay", type=float)

    parser.add_argument("--epochs",type=int, default=100)
    parser.add_argument("--stop-epoch", dest="stop_epoch", default=60, type=int)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--lr-policy", dest="lr_policy", default='cycle', 
                        help='learning rate policy. [cycle | multi]')
    parser.add_argument("--lr-pct-start", default=0.2,type=float,
                        help="The percentage of the cycle (in number of steps) spent increasing the learning rate.")
    parser.add_argument("--lr-cycle-three-phase", action="store_true",
                        help="enabling Three phase in OneCycleLR")
    parser.add_argument("--lr-final-div-factor", type=float, default=10000,
                        help="the final learning rate = initial lr / lr_final_div_factor")
    parser.add_argument("--milestones", nargs="+", type=int, default=[5,20],
                        help="List of epoch indices. Must be increasing. Multisteplr")

    parser.add_argument('--batch-size', dest="batch_size",default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument("--validation-split", dest="validation_split", type=float, default=0.0,
                        help="the ratio of size about validation dataset  and original dataset")
    parser.add_argument("--extra-val", type=str, default=None,
                        help="the extra validation dataset (e.g., valid)")
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--load",dest="trained_model",
                        help="the model state dict filename")
    parser.add_argument("--no-eval", dest="no_eval", action="store_true",
                        help="not evaluate the result")
    parser.add_argument("--prediction-save", dest="prediction_save",
                        help="the file address for saving prediction result")
    prog_name, _ = get_surfix(os.path.basename(__file__))
    parser.add_argument("--for-retrain", dest="for_retrain", type=str,
                        help="the model state dict for retrain")
    parser.add_argument("--full-fine-tuning", action="store_true",
                        help="not freeze")
    
    # augmentation
    parser.add_argument("--smoothing", type=float, default=0.,
                        help="label smoothing")
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                         help="mixup alpha")
    
    parser.add_argument("--torch-compile", action="store_true",
                        help="using compile in torch 2")
    
    parser.add_argument("--accum-steps", default=1, type=int,
                        help="accumulation steps of gradient accumulation")
    
    # top models
    parser.add_argument("--top-k-models", type=int, default=3,
                        help="saving top k models")

    parser.add_argument('--wandb', type=str, default=f'som-{prog_name}', 
            help="wandb project name")

    # Add Optuna-related arguments
    parser.add_argument('--use-optuna', action='store_true', help='Use Optuna for hyperparameter optimization')
    parser.add_argument('--optuna-trials', type=int, default=20, help='Number of Optuna trials to run')
    
    opt = parsing(parser)

    mode = opt.mode


    # world_size = torch.npu.device_count()
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f"world_size:{world_size},local_rank{local_rank}")
    # Set master address and port if not specified
    if os.environ.get("MASTER_ADDR") is None:
        os.environ["MASTER_ADDR"] = "localhost"
    if os.environ.get("MASTER_PORT") is None:
        os.environ["MASTER_PORT"] = "60010"
    
    # Initialize the process group using environment variables
    dist.init_process_group(backend='nccl', init_method='env://')

    if mode == "train":
        # Reduce batch size for optuna trials
        bs_rate = 0.75
        batch_size = opt.batch_size
        if opt.use_optuna:
            opt.batch_size = int(opt.batch_size * bs_rate)
            # Run Optuna hyperparameter optimization
            best_params = run_optuna_optimization(opt, local_rank, world_size, opt.optuna_trials)
            
            if local_rank == 0:
                print("\nOptuna optimization completed!")
                print(f"Best parameters: {best_params}")
                
                # Update options with best parameters for a final training run
                opt.lr = best_params["learning_rate"]
                opt.lr_pct_start = best_params["lr_pct_start"]
                opt.weight_decay = best_params["weight_decay"]
                opt.ema_decay = best_params["ema_decay"]
                
                print("\nStarting final training with best parameters...")
        # Using the original batch size for final training
        opt.batch_size = batch_size
        training_prog(opt,local_rank, world_size)
    elif mode == "test":
        evaluation_prog(opt,local_rank,world_size)
    elif mode == "ind_test":
        evaluation_prog(opt,local_rank,world_size)
    # elif mode == "exp_trian":
    #     # Reduce batch size for optuna trials
    #     bs_rate = 0.75
    #     batch_size = opt.batch_size
    #     if opt.use_optuna:
    #         raise NotImplementError(f"the exp train mode does not support optuna")
    #     opt.batch_size = batch_size
    #     exp_training_prog(opt, local_rank, world_size)
    else:
        raise NotImplementedError(f"the mode of {mode} is not implemented")

if __name__ == "__main__":
    main()