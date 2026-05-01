import re
from subprocess import Popen, PIPE, STDOUT
from Bio.SeqIO import parse
import numpy as np
import torch
from torch import Tensor

def fasta2generator(fafilepath):
    with open(fafilepath, "r") as h:
        for record in parse(h, "fasta"):
            yield record

def get_surfix(filename : str):
    parser = re.compile(r"\.[^\.]+$")
    resu = parser.search(filename)
    if resu is not None:
        st = resu.start()
        prefix = filename[:st]
        surfix = filename[st+1:]
    else:
        prefix = filename
        surfix = ""

    return prefix, surfix

def cliprint(outstr : str):
    cmd = "printf '{}'".format(outstr)
    status = Popen(cmd, stdout=PIPE, stderr=PIPE,shell=True)

    _, error = status.communicate()

    assert error.decode() == "",\
        error.decode()

def reweight(msa_onehot : Tensor, cutoff : float):
    """
    """
    # wmat : Tensor
    epsilon = 1e-9
    # b, c, h, w = msa_onehot.shape
    # wmat = torch.tensordot(msa_onehot, msa_onehot, [[1,2],[1,2]])
    # thres = cutoff * h
    # msa_emb = msa_onehot.view((b, c, h*w))
    msa_onehot = msa_onehot.flatten(-2, -1) # b x c x e
    # wmat : Tensor = torch.matmul(msa_emb, 
    #                     torch.transpose(msa_emb, 0,1))
    # msa_onehot = torch.matmul(msa_onehot, msa_onehot.transpose(0, 1))
    msa_norm = torch.norm(msa_onehot,p="fro",dim=-1).unsqueeze(dim=-1) # b x c x 1
    msa_norm = torch.bmm(msa_norm, msa_norm.transpose(-2,-1)) # b x c x c
    msa_onehot = torch.bmm(msa_onehot,
                           msa_onehot.transpose(-2,-1)) # b x c x c
    msa_onehot /= (msa_norm + epsilon)
    # norm_r = torch.norm(msa_emb,2,dim=1).reshape((-1,1))
    # norm_r = torch.matmul(norm_r,
    #                       torch.transpose(norm_r, 0,1))
    # wmat /= (norm_r + epsilon)

    # print(f"wmat shape {wmat.shape}")
    # w : Tensor
    # weight : Tensor = 1. / torch.sum(torch.where(wmat > thres, 1., 0.), dim=-1)
    # print(f"w shape {w.shape}")
    # return weight
    # b x c
    return 1. / torch.sum(torch.where(msa_onehot > cutoff, torch.tensor([1.0]), torch.tensor([0.0])))

def msa2pssm(msa_onehot : Tensor, weight : Tensor):
    """
    msa_onehot: b x k x L x 21
    weigh: b x k
    the parameters weight, this tensor size will be changed by the effect of unsqueeze_
    """
    # b x 1 x 1
    w_sum = torch.sum(weight, dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    epsilon = 1e-9
    # L x 21
    # f = torch.sum(weight.reshape((b, k ,1, 1)) * msa_onehot, dim=0) / w_sum + epsilon
    weight = weight.unsqueeze(dim=-1).unsqueeze(dim=-1)
    f = torch.sum(weight * msa_onehot, dim=-3) / w_sum + epsilon # b x L x 21
    return f

# shrunk covariance inversion
def fast_dca(msa_onehot : Tensor, weight : Tensor, penalty : float):
    """
    msa_onehot: b x k x L x 21
    weight: b x k

    remember divide w means multple weight, since the meaning of w likes the 1 / weight
    the parameters weight, this tensor size will be changed by the effect of unsqueeze_
    """
    b, c, h, w = msa_onehot.shape
    # x = rearrange(msa_onehot, "b c h w -> c (h w)")
    # x = msa_onehot.reshape((b, c, h*w))
    x = msa_onehot.flatten(-2,-1) # b x k x e

    n_points = torch.sum(weight,dim=-1) - torch.sqrt(torch.mean(weight, dim=-1))
    n_points.unsqueeze_(dim=-1) # b x 1

    # mean_weight : Tensor = torch.sum(weight.reshape((-1,1)) * x, dim=0) / n_points
    weight.unsqueeze_(dim=-1) # b x k x 1
    mean_weight = torch.sum(weight * x, dim=-2) / n_points # b x e
    # the shape of mean_weight need to be, 1 x embeddings
    # weighted std
    # x = (x - mean_weight.reshape((1,-1))) * torch.sqrt(weight.reshape((-1,1)))
    mean_weight.unsqueeze_(dim=-2)  # b x 1 x e
    x = (x - mean_weight) * torch.sqrt(weight)
    # conv = torch.tensordot(x, x, dims=([0], [0])) / n_points
    x = torch.bmm(x.transpose(-2,-1), x) # b x e x e

    # conv_regular = conv + torch.eye(conv.shape[0]) * penalty / torch.sqrt(torch.mean(w))
    n = x.shape[-1]
    # regularization
    reg_term = 1 * penalty / torch.sqrt(torch.mean(weight, dim=-2)) # b x 1
    x[:, torch.arange(n), torch.arange(n)] += reg_term
    inv_x = torch.inverse(x)

    # l x l x hidden
    # x = rearrange(inv_conv, "(h1 w1) (h2 w2) -> h1 h2 (w1 w2)", h1=msa_onehot.shape[1], h2=msa_onehot.shape[1])
    # x : Tensor = inv_conv.view((h, w, h, w)).transpose(1,2).reshape((h,h,w*w))
    # return x.view((h, w, h, w)).transpose(1,2).flatten(2,3)
    # b x e x e -> b x h x w x e-> b x h x w x h x w -> b x h x h x w x w -> b x h x h x w*w
    # return inv_x.unflatten(-2, (h,w)).unflatten(-1, (h,w)).transpose_(-3,-2).flatten(-2,-1)
    return inv_x.view(b, h, w, h, w).permute(0,1,3,2,4).flatten(-2,-1)


ONE_TO_THREE_LETTER_MAP = {
    "R": "ARG",
    "H": "HIS",
    "K": "LYS",
    "D": "ASP",
    "E": "GLU",
    "S": "SER",
    "T": "THR",
    "N": "ASN",
    "Q": "GLN",
    "C": "CYS",
    "G": "GLY",
    "P": "PRO",
    "A": "ALA",
    "V": "VAL",
    "I": "ILE",
    "L": "LEU",
    "M": "MET",
    "F": "PHE",
    "Y": "TYR",
    "W": "TRP",
    "X": "UNK"
}

THREE_TO_ONE_LETTER_MAP = {
    "ARG": "R",
    "HIS": "H",
    "LYS": "K",
    "ASP": "D",
    "GLU": "E",
    "SER": "S",
    "THR": "T",
    "ASN": "N",
    "GLN": "Q",
    "CYS": "C",
    "GLY": "G",
    "PRO": "P",
    "ALA": "A",
    "VAL": "V",
    "ILE": "I",
    "LEU": "L",
    "MET": "M",
    "PHE": "F",
    "TYR": "Y",
    "TRP": "W",
    "DAR": "R",
    "DHI": "H",
    "DLY": "K",
    "DAS": "D",
    "DGL": "E",
    "DSN": "S",
    "DTH": "T",
    "DSG": "N",
    "DGN": "Q",
    "DCY": "C",
    "DPR": "P",
    "DAL": "A",
    "DVA": "V",
    "DIL": "I",
    "DLE": "L",
    "MSE": "M",
    "MED": "M",
    "DPN": "F",
    "DTY": "Y",
    "DTR": "W",
    "ASX": "B",
    "PYL": "O",
    "SEC": "U",
    "GLX": "Z",
    "UNK": "X"
}