from dataclasses import dataclass
import torch
import math


@dataclass
class Constant:
    """Class for non distribution related constants."""
    batch_size: int = 250
    test_batch_size: int = 1
    classes: int = 2
    train_epochs: int = 80
    samples: int = 1
    num_networks: int = 5
    save_period: int = 10
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_seed: int = 0


@dataclass
class DistConstant:
    """Class for all initialization constants for different distributions"""
    init_mu = (-0.2, 0.2)
    init_rho = (-5, -4)
    mixture_scale = 0.5
    sigma1 = torch.tensor([math.exp(-0)]).to(Constant.device)
    sigma2 = torch.tensor([math.exp(-6)]).to(Constant.device)
