import torch
import torch.nn as nn
import torch.nn.functional as F

activations = {"relu"     : F.relu,
               "identity" : nn.Identity(),
               "tanh"     : torch.tanh}
