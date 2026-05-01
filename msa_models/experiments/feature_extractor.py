# import os
# import pickle
# import sys
# import pathlib
# prj_dir = str(pathlib.Path(__file__).parent.parent)
# if prj_dir not in sys.path:
#   sys.path.append(prj_dir)

# import argparse as P
# import typing as T
# import torch
# import torch as th
# import torch.nn as nn
# # import torch.utils.data as UD
# from torch.utils.data import Dataset, DataLoader 
# import torch.nn.functional as F
# from torch.amp.autocast_mode import autocast
# import functools as ft
# import re
# import numpy as np
# import tqdm

# import experiments.msa as D
# import models.gendis as G

# ANYSUFFIX = re.compile(r"\.[^.]*$")
# REMOVE_ANYSUFFIX: T.Callable[[str],str] = lambda x: ANYSUFFIX.sub("", x)

# def get_activation(activation: T.Dict[str, th.Tensor], name: str = "feature"):
#   """
#   """
#   def hook(model: nn.Module, input: th.Tensor, output: th.Tensor):
#     activation[name] = output.detach()
#   return hook

# def activation_with_avgpool(activation: T.Dict[str, th.Tensor], name: str = "feature"):
#   def hook(model: nn.Module, input: th.Tensor, output: th.Tensor):
#     activation[name] = output.detach()
#   return hook

# def meta_extractor(efunc: T.Callable[..., None]):
#   def _wrapper_extractor(opt: P.Namespace, weight_path:str,
#                          name: str, **kwargs):
#     batch_size = opt.batch_size
#     dataset = D.MSADataset(opt.file_address, opt.working_dir, opt.mode, opt.task,
#         opt.num_classes, opt.top_k, opt.max_len,
#         need_proteins=not opt.no_ipr_input)
#     L = len(dataset)
#     size = round(L / batch_size)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=opt.dataloader_num_workers,
#                         pin_memory=True,
#                         )
#     model_weight = torch.load(weight_path)
#     model = G.Arch(opt)
#     model.load_state_dict(model_weight)
#     activation: T.Dict[str, th.Tensor] = {}
#     efunc(model, activation, name, **kwargs)
#     permute_dims = getattr(opt, "permute_dims", (0,3,2,1))
#     for i, (X, _) in tqdm.tqdm(enumerate(loader)):
#       d = (i+1)*batch_size
#       stop = d if d < L else L
#       paths = [dataset.get("msa", j) for j in range(i*batch_size, stop)]
#       # assert isinstance(paths[0], str)
#       if isinstance(X, torch.Tensor):
#           X = X.cuda()
#       else:
#           assert isinstance(X, T.List)
#           proteins, X = X
#           assert isinstance(X, torch.Tensor)
#           X = X.cuda()
#           model.set_proteins(proteins)
#       # print(input.shape)
#       # compute output
#       # print(torch.isnan(input))

#       with torch.no_grad():
#           with autocast():
#               model(X, permute_dims=permute_dims)
#       result = activation[name].cpu().detach()
#       if len(result.size()) > 2: result = result.flatten(1).numpy()
#       # print(f"saving batch [{i}/{size}]")
#       if (saving_dir := kwargs.get("saving_dir")) is not None:
#         for i, p in enumerate(paths):
#           assert isinstance(p, str)
#           p = os.path.join(saving_dir, f"{os.path.basename(REMOVE_ANYSUFFIX(p))}.npy")
#           # if not os.path.exists(p):
#           np.save(p, result[i])
#           # else:
#           #   print("{} existed".format(os.path.basename(p).removesuffix(".npy")))


#   return _wrapper_extractor

# def head_feature(arch: G.Arch, layername: str,
#                  activation: T.Dict[str, th.Tensor],
#                  name: str,
#                  **kwargs):
#   act = ft.partial(activation_with_avgpool, activation)
#   head = arch.get_fc()
#   assert hasattr(head, layername)
#   assert isinstance(getattr(head, layername), nn.Module)
#   if isinstance((h := getattr(head, layername)), nn.Module):
#     h.register_forward_hook(act[name]) # type: ignore

# @meta_extractor
# def sequence_feature(model: nn.Module, activation: T.Dict[str, th.Tensor], name: str, **kwargs):
#   """
#   """
#   act = ft.partial(activation_with_avgpool, activation)
#   # type(model) is the Arch class in ProFun-SOM project models/gendis.py
#   assert isinstance(model.rnet, nn.Module)
#   assert isinstance(model.rnet.module, nn.Module)
#   # assert isinstance(model.rnet.module.body, nn.Module)
#   # model.rnet.module.body.register_forward_hook(act(name)) # type: ignore
#   if hasattr(model.rnet.module, "global_pool"):
#     model.rnet.module.global_pool.register_forward_hook(act(name))
#   else:
#     raise ValueError("only support cnn")

# @meta_extractor
# def interpro_feature(model: nn.Module, activation: T.Dict[str, th.Tensor], name: str, **kwargs):
#   act = ft.partial(get_activation, activation)

#   if isinstance(model, G.Arch):
#     head = model.get_fc()
#     assert isinstance(head.init_ipr_proj, nn.Module)
#     head.init_ipr_proj.register_forward_hook(act(name)) # type: ignore
#   else:
#     raise NotImplementedError("only for ProFun-SOM")

# @meta_extractor
# def msa_feature(model: nn.Module, activation: T.Dict[str, th.Tensor], name: str, **kwargs):
#   act = ft.partial(get_activation, activation)

#   if isinstance(model, G.Arch):
#     head = model.get_fc()
#     assert isinstance(head.init_msa_proj, nn.Module)
#     head.init_msa_proj.register_forward_hook(act(name)) # type: ignore
#   else:
#     raise NotImplementedError("only for ProFun-SOM")

# @meta_extractor
# def cross_feature(model: nn.Module, activation: T.Dict[str, th.Tensor], name: str, **kwargs):
#   act = ft.partial(get_activation, activation)

#   if isinstance(model, G.Arch):
#     head = model.get_fc()
#     assert hasattr(head, "ln") # layernorm
#     assert isinstance(head.ln, nn.Module)
#     head.ln.register_forward_hook(act(name)) # type: ignore
#   else:
#     raise NotImplementedError("only for ProFun-SOM")

# # def predicted_score(opt: P.)
# if __name__ == "__main__":
#   parser = P.ArgumentParser()
#   parser.add_argument("config")
#   parser.add_argument("weight_path")
#   parser.add_argument("saving_dir")
#   parser.add_argument("-b", "--batch-size", type=int, default=4)
#   parser.add_argument("-e", "--extractor", choices=["interpro",
#                                                     "msa",
#                                                     "cross",
#                                                     "normal"])
#   parser.add_argument("-m", "--mode", choices=["train", "test", "valid"],
#                       type=str,
#                       default="test")
#   parser.add_argument("-dnw","--dataloader-num-workers", type=int, default=5)

#   cli_opt = parser.parse_args()
#   with open(cli_opt.config, "rb") as h:
#     config_dict = pickle.load(h)
#   config_dict["batch_size"] = cli_opt.batch_size
#   opt = P.Namespace(**config_dict)
#   setattr(opt, "mode", cli_opt.mode)
#   setattr(opt, "dataloader_num_workers", cli_opt.dataloader_num_workers)
#   match cli_opt.extractor:
#     case "normal":
#       sequence_feature(opt, cli_opt.weight_path, saving_dir=cli_opt.saving_dir,
#                        name="feat")
#     case "interpro":
#       interpro_feature(opt, cli_opt.weight_path, saving_dir=cli_opt.saving_dir,
#                        name=cli_opt.extractor)
#     case "msa":
#       msa_feature(opt, cli_opt.weight_path, saving_dir=cli_opt.saving_dir,
#                   name=cli_opt.extractor)
#     case "cross":
#       cross_feature(opt, cli_opt.weight_path, saving_dir=cli_opt.saving_dir,
#                   name=cli_opt.extractor)
