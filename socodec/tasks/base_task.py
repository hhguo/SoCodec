import torch

from ..networks import build_network


class BaseTask(torch.nn.Module):

    def __init__(self,
                 modules,
                 mode: str = 'train',
                 **kwargs):
        super().__init__()
        self.add_network(modules)
        self.mode = mode

    def forward(self, *args, **kwargs):
        pass

    def add_network(self, modules):
        for name, network in build_network(modules):
            self.add_module(name, network)