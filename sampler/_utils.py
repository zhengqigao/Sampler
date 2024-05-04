from typing import TypeVar, Callable, Union, Optional
import torch
import torch.nn as nn
from ._common import Func, Distribution

## TODO: define an alias decorator, so maybe we can call ais or annealed_importance_sampling for the same function?
def _alias(*aliases):
    def decorator(func):
        for alias in aliases:
            globals()[alias] = func
        return func
    return decorator

def _get_params(module: Union[torch.nn.Module, torch.nn.DataParallel]):
    if isinstance(module, torch.nn.DataParallel):
        return _get_params(module.module)
    else:
        return module.parameters()


class _ModuleWrapper(nn.Module):
    def __init__(self, func: Func):
        super(_ModuleWrapper, self).__init__()
        self.func = func
    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

