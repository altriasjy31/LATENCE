import os
import sys

prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if prj_dir not in sys.path:
    sys.path.append(prj_dir)

from models.gendis import Arch
import yaml
from argparse import Namespace, ArgumentParser
from typing import Dict, Union, List

## parsing configs from configs/
def building(config_path: str, device: Union[List[int], int, None]):
    with open(config_path, "r") as h:
        opt_dict: Dict = yaml.safe_load(h)
    
    with open(opt_dict["base"], "r") as h:
        config_dict: Dict = yaml.safe_load(h)
    
    opt_dict.update(config_dict)

    if isinstance(device, int):
        device = [device] if device != -1 else []
    elif device is None:
        device = []
    elif isinstance(device, List):
        pass
    else:
        raise TypeError("device should be List[int] or int or None, e.g. [0,1] or 1")

    opt_dict["gpu_ids"] = device 
    
    opt = Namespace(**opt_dict)

    return Arch(opt)

if __name__ == "__main__":
    parser = ArgumentParser()
    config_path = "configs/tsmf/bpo.yml"
    parser.add_argument("config_path", type=str, default=config_path)
    parser.add_argument("-d","--device", type=int, default=-1)
    opt = parser.parse_args()
    config_path = opt.config_path
    device= opt.device
    model_arch = building(config_path, device)
    print(model_arch)