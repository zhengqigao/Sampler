from typing import TypeVar, Callable, Union, Optional
import torch

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





