## generator and discriminator
import typing as T
import functools as ft
from argparse import ArgumentParser, Namespace
import pickle
import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn.parallel as nn_parallel
import numpy as np

import os
import sys

prj_dir = os.path.dirname(os.path.dirname(__file__))
if prj_dir not in sys.path:
    sys.path.append(prj_dir)

from experiments.msa import MSAEncoder

import models.architecture as arch
from models.architecture import get_norm_layer, init_net
from models.architecture import ResnetGenerator
from timm_models import timm_create_model
from timm_models import TimmResNet

from models.utils import parsing

import helper_functions.aug as aug

cuda = True if torch.cuda.is_available() else False

class Arch(nn.Module):
    def __init__(self, opt : Namespace):
        """
        in_channels: int
        out_channels_G: int
        num_classes: int
        top_k: int
        max_len: int
        msa_cutoff: float
        msa_penalty: float
        msa_embedding_dim: int
        msa_encoding_strategy: str
        ngf: int
        netG: str
        no_antialias: bool
        no_antialias_up: bool
        load_gen: Optional[str]
        freeze_gen: Optional[str]
        ndf: int
        netD: str
        replace_stride_with_dilation: List[bool]
        no_dropout: bool
        normG: str
        normD: str
        init_type: str
        init_gain: float
        gpu_ids: List[int]
        no_jit: bool
        """
        super(Arch, self).__init__()

        self.in_channels = opt.in_channels
        self.out_channels_G = opt.out_channels_G
        self.num_classes = opt.num_classes
        self.gpu_ids = opt.gpu_ids

        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        need_init = True

        pre_model = MSAEncoder(opt.msa_embedding_dim, 
                                encoding_strategy=opt.msa_encoding_strategy)
        # pre_model = nn.Embedding(21, embedding_dim)

        # the weight of pre_model should also be loaded
        self.pre_model = init_net(pre_model, init_type=opt.init_type, init_gain=opt.init_gain,
                                  gpu_ids=opt.gpu_ids, initialize_weights=need_init)

        # nets = []
        # input: b * 21 * seqLen * topK
        # update
        # input: b * 441 * seqLen(patched) * seqLen(patched)
        # defaults is b * 441 * 128 * 128
        
        self.gnet = define_G(self.in_channels, self.out_channels_G,ngf=opt.ngf, netG=opt.netG,
                             norm=opt.normG, use_dropout=not opt.no_dropout,
                             init_type=opt.init_type, init_gain=opt.init_gain,
                             no_antialias=opt.no_antialias, no_antialias_up=opt.no_antialias_up,
                             gpu_ids=opt.gpu_ids,
                             need_init_weights=need_init)

        params = opt.__dict__
        ks = get_init_keys(name2model[opt.netD])
        ropt = {"params": {k: params[k] for k in ks},
                "head_params": {}}
        self.rnet = define_D(self.out_channels_G, self.num_classes, netD=opt.netD,
                             init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids,
                             **ropt)
    
    def get_fc(self):
        model = self.rnet.module if isinstance(self.rnet, nn_parallel.DataParallel) \
            else self.rnet
        if hasattr(model, "fc"):
            fc = model.fc
        elif hasattr(model, "head"):
            fc = model.head
        else:
            raise NotImplementedError("only support fc or head")
        assert isinstance(fc, nn.Module)
        return fc
    
    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def freeze_all_except_fc(self):
        self.freeze_all()
        fc = self.get_fc()
        for p in fc.parameters():
            p.requires_grad = True

    def _forward_encoded(
        self,
        x: Tensor,
        permute_dims: T.Tuple[int, int, int, int] = (0, 3, 2, 1),
        return_embedding: bool = False,
    ):
        """
        x is expected to be encoded MSA representation before permute.
        Usually shape: [B, K, L, C].
        """

        x = x.permute(*permute_dims).contiguous()

        if self.gnet is not None:
            x = self.gnet(x)

        assert hasattr(self.rnet, "forward_features"), \
            "cannot return embedding, since rnet does not support forward_features"

        h = self.rnet.forward_features(x)

        assert hasattr(self.rnet, "forward_head"), \
            "cannot return embedding, since rnet does not support forward_head"

        logits = self.rnet.forward_head(h)

        if return_embedding:
            return logits, h

        return logits

    def _forward_msa_multi_view(
        self,
        x_encoded: Tensor,
        view_params_list: T.List[T.Dict],
        permute_dims: T.Tuple[int, int, int, int] = (0, 3, 2, 1),
        return_embedding: bool = False,
        return_format: str = "tuple",
    ):
        """
        Multi-view MSA augmentation.

        This method assumes x_encoded is already produced by self.pre_model(x).

        It creates multiple augmented views from the same encoded MSA, concatenates
        them on batch dimension, and performs only one pass through gnet/rnet.

        Motivation:
            Avoid doing two separate forwards through BatchNorm-containing rnet
            before a single backward, which can trigger autograd version errors
            due to in-place updates of BatchNorm running statistics.

        Args:
            x_encoded:
                Encoded MSA tensor, usually [B, K, L, C].
            view_params_list:
                List of msa_view_params dictionaries. Each dict is passed to
                aug.apply_msa_view_aug_encoded.
            permute_dims:
                Permutation used by _forward_encoded.
            return_embedding:
                Whether to return backbone embedding h.
            return_format:
                "tuple":
                    return (logits_views, h_views) if return_embedding else logits_views,
                    where logits_views is a tuple/list with one tensor per view.
                "cat":
                    return concatenated tensors directly.

        Returns:
            If return_format == "tuple":
                return_embedding=False:
                    (logits_v1, logits_v2, ...)
                return_embedding=True:
                    ((logits_v1, logits_v2, ...), (h_v1, h_v2, ...))

            If return_format == "cat":
                return_embedding=False:
                    logits_cat
                return_embedding=True:
                    logits_cat, h_cat
        """
        if not isinstance(view_params_list, (list, tuple)):
            raise TypeError(
                f"view_params_list must be list/tuple, got {type(view_params_list)}"
            )

        if len(view_params_list) == 0:
            raise ValueError("view_params_list must contain at least one view params dict")

        views = []
        for vp in view_params_list:
            if vp is None:
                # Optional identity view.
                views.append(x_encoded)
            else:
                views.append(
                    aug.apply_msa_view_aug_encoded(
                        x_encoded,
                        **vp,
                    )
                )

        batch_size = x_encoded.shape[0]
        x_cat = torch.cat(views, dim=0)

        out = self._forward_encoded(
            x_cat,
            permute_dims=permute_dims,
            return_embedding=return_embedding,
        )

        if return_format == "cat":
            return out

        if return_format != "tuple":
            raise ValueError(f"Unknown return_format: {return_format}")

        if return_embedding:
            logits_cat, h_cat = out
            logits_views = torch.split(logits_cat, batch_size, dim=0)
            h_views = torch.split(h_cat, batch_size, dim=0)
            return logits_views, h_views

        logits_cat = out
        logits_views = torch.split(logits_cat, batch_size, dim=0)
        return logits_views

    def forward(
        self,
        x: Tensor,
        permute_dims: T.Tuple[int, int, int, int] = (0, 3, 2, 1),
        aug_params: T.Optional[T.Dict] = None,
        return_embedding: bool = False,
        **kwargs,
    ):
        """
        Four modes:

        1. aug_params is None:
           normal forward.

        2. aug_params contains "y":
           old mixup path, for compatibility with train.py.

        3. aug_params["aug_type"] == "msa_view":
           single weak/strong MSA view augmentation for semi-supervised training.

        4. aug_params["aug_type"] == "msa_multi_view":
           multiple MSA views are generated from the same encoded MSA, concatenated
           on batch dimension, and passed through gnet/rnet once. This is useful
           when the downstream loss needs multiple views but rnet contains
           BatchNorm layers.
        """

        # Always encode raw MSA first.
        x = self.pre_model(x)

        # ------------------------------------------------------------
        # Case 1: no augmentation
        # ------------------------------------------------------------
        if aug_params is None:
            return self._forward_encoded(
                x,
                permute_dims=permute_dims,
                return_embedding=return_embedding,
            )

        # ------------------------------------------------------------
        # Case 2: old mixup path
        # Keep this for compatibility with train.py.
        # ------------------------------------------------------------
        if "y" in aug_params:
            y = aug_params["y"]

            assert isinstance(y, Tensor), "Y must be a Tensor"

            x_mix, y_mix, lam = aug.mixup_msa_data(
                x,
                y,
                alpha=aug_params.get("mixup_alpha", 0.2),
            )

            out = self._forward_encoded(
                x_mix,
                permute_dims=permute_dims,
                return_embedding=return_embedding,
            )

            if return_embedding:
                logits, h = out
                return logits, y_mix, h

            return out, y_mix

        aug_type = aug_params.get("aug_type", None)

        # ------------------------------------------------------------
        # Case 3: single MSA view augmentation path
        # Existing behavior. Keep for compatibility.
        # ------------------------------------------------------------
        if aug_type == "msa_view":
            msa_view_params = aug_params.get("msa_view_params", {})

            x_aug = aug.apply_msa_view_aug_encoded(
                x,
                **msa_view_params,
            )

            return self._forward_encoded(
                x_aug,
                permute_dims=permute_dims,
                return_embedding=return_embedding,
            )

        # ------------------------------------------------------------
        # Case 4: multi-view MSA augmentation path
        # New behavior. Used to avoid multiple BN forwards before backward.
        # ------------------------------------------------------------
        if aug_type == "msa_multi_view":
            view_params_list = aug_params.get("views", None)

            if view_params_list is None:
                # Also support a more explicit alias.
                view_params_list = aug_params.get("msa_view_params_list", None)

            if view_params_list is None:
                raise ValueError(
                    "aug_type='msa_multi_view' requires aug_params['views'] "
                    "or aug_params['msa_view_params_list']"
                )

            return_format = aug_params.get("return_format", "tuple")

            return self._forward_msa_multi_view(
                x_encoded=x,
                view_params_list=view_params_list,
                permute_dims=permute_dims,
                return_embedding=return_embedding,
                return_format=return_format,
            )

        # ------------------------------------------------------------
        # Unknown aug_params
        # ------------------------------------------------------------
        if aug_type is None:
            # Be conservative: no augmentation.
            return self._forward_encoded(
                x,
                permute_dims=permute_dims,
                return_embedding=return_embedding,
            )

        raise ValueError(f"Unknown aug_type: {aug_type}")


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None,
             need_init_weights = True):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2.
        need_init_weights (bool) -- control whether to initialize the net weights
    """

    norm_layer = get_norm_layer(norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, 
                              n_blocks=9, opt=opt)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, 
                              n_blocks=6, opt=opt)
    elif netG == 'resnet_4blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, 
                              n_blocks=4, opt=opt)
    elif netG == "resnet_2blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up,
                              n_blocks=2, opt=opt)
    elif netG == "resnet_oneblock":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up,
                              n_blocks=1, opt=opt)
    elif netG is None or netG == "none":
        net = None
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=need_init_weights) \
            if net is not None else net

name2net = {
    "timm_resnet50d": ft.partial(timm_create_model, variant="resnet50d"),
    "timm_resnetaa50d": ft.partial(timm_create_model, variant="resnetaa50d"),
    "timm_ecaresnet50d": ft.partial(timm_create_model, variant="ecaresnet50d"),
    "timm_ecaresnet101d": ft.partial(timm_create_model, variant="ecaresnet101d"),
    "timm_ecaresnet152d": ft.partial(timm_create_model, variant="ecaresnet152d"),
    "timm_ecaresnet101t": ft.partial(timm_create_model, variant="ecaresnet101t"),
    "timm_ecaresnetblur101d": ft.partial(timm_create_model, variant="ecaresnetblur101d"),
    "timm_ecaresnetblur152d": ft.partial(timm_create_model, variant="ecaresnetblur152d"),
    "timm_ecaresnetaa101d": ft.partial(timm_create_model, variant="ecaresnetaa101d"),
}

name2model = {
    "timm_resnet50d": TimmResNet, "timm_resnetaa50d": TimmResNet,
    "timm_ecaresnet50d": TimmResNet, 
    "timm_ecaresnet101d": TimmResNet,
    "timm_ecaresnet152d": TimmResNet,
    "timm_ecaresnet101t": TimmResNet,
    "timm_ecaresnetblur101d": TimmResNet,
    "timm_ecaresnetblur152d": TimmResNet,
    "timm_ecaresnetaa101d": TimmResNet,
}

def get_init_keys(net):
    init_keys = []

    return init_keys

def define_D(in_channels : int, num_classes : int, netD : str,
             init_type='normal', init_gain=0.02, gpu_ids = [], need_init_weights = True,
             **opt):
        assert  name2net.get(netD) is not None, \
            f"{netD} is not implemented"
        
        params = opt["params"]
        norm_type = params.get("normD", None)
        if norm_type is not None:
            del params["normD"]
            params["norm_layer"] = get_norm_layer(norm_type)
        
        rnet =  name2net[netD](in_channels=in_channels, num_classes=num_classes, **params)
        if need_init_weights:
            rnet = init_net(rnet, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)

        return rnet

def main():
    """
    """
    parser = ArgumentParser()

    parser.add_argument("--in-channels", dest="in_channels", type=int, default=21)
    parser.add_argument("--out-channels-G", dest="out_channels_G", type=int, default=42)
    parser.add_argument("--num-classes",dest="num_classes", type=int, default=19939)

    parser.add_argument("--top-k",dest="top_k",type=int,default=100,
                        help="select the top k sequence ib msa")
    parser.add_argument("--max-len",dest="max_len", type=int, default=1000,
                        help="the max lenght of sequence for using")

    parser.add_argument("--msa-cutoff", dest="msa_cutoff", type=float,default=0.8,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-penalty", dest="msa_penalty", type=float,default=4.5,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-embedding-dim", dest="msa_embedding_dim", type=int, default=21,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-encoding-strategy", dest="msa_encoding_strategy", type=str, 
                        choices=["one_hot", "emb", "emb_plus_one_hot", "emb_plus_pssm","fast_dca"], 
                        default="emb_plus_one_hot",
                        help="parameters for encoding the msa file")

    parser.add_argument('--ngf', type=int, default=64, 
                        help='# of gen filters in the last conv layer')
    parser.add_argument('--netG', type=str, default='resnet_4blocks', 
                        help="specify generator architecture " +
                        "[resnet_9blocks | resnet_6blocks | resnet_4blocks]")
    parser.add_argument("--no-antialias", dest="no_antialias", action="store_true",
                        help="use dilated convolution blocks in generator")
    parser.add_argument("--no-antialias-up", dest="no_antialias_up", action="store_true",
                        help="use dilated convolution_transposed blocks in generator")
    parser.add_argument("--load-gen", dest="load_gen", type=str,
                        help="the path of pre-trained generator model")
    parser.add_argument("--freeze-gen", dest="freeze_gen", action="store_true",
                        help="control whether to freeze the generator model when training")
    parser.add_argument('--ndf', type=int, default=64, 
                            help='# of dis filters in the last conv layer')
    parser.add_argument('--netD', type=str, default='resnet50v2', 
                        help='specify discriminator architecture')

    parser.add_argument('--no-dropout', dest="no_dropout",action='store_true', 
                        help='no dropout for the generator', default=False)
    parser.add_argument('--normG', type=str, default='instance', 
                        help='for generator, instance normalization or batch normalization [instance | batch | none]')  
     
    parser.add_argument('--init-type', dest="init_type", type=str, default='xavier', 
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init-gain', dest="init_gain", type=float, default=0.02, 
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--gpu-ids', dest="gpu_ids",type=str, default='0,1', 
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument("--no-jit", dest="no_jit", action="store_true",
                        help="not use torch.jit.script")
    opt : Namespace
    opt = parsing(parser)

    print(opt)

    model_arch = Arch(opt).cuda()
    print(model_arch)
    x = torch.randint(0, 21,(12, 40, 2000)).cuda()
    x = model_arch(x)

if __name__ == "__main__":
    main()
