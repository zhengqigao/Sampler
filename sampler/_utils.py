from typing import TypeVar, Callable, Union, Optional
import torch

## TODO: define an alias decorator, so maybe we can call ais or annealed_importance_sampling for the same function?
def _alias(*aliases):
    def decorator(func):
        for alias in aliases:
            globals()[alias] = func
        return func
    return decorator


def _leapfrog(p: torch.Tensor,
              q: torch.Tensor,
              dp_dt: torch.Tensor,
              dq_dt: torch.Tensor,
              step_size: float,
              num_leapfrog: int):
    P_half = p - step_size / 2 * dp_dt
    Q = q + step_size * dq_dt
    for _ in range(num_leapfrog - 1):
        Dp = -step_size * dp_dt
        Dq = step_size * dq_dt
        P_half = P_half + Dp
        Q = Q + Dq


def _get_params(module: Union[torch.nn.Module, torch.nn.DataParallel]):
    if isinstance(module, torch.nn.DataParallel):
        return _get_params(module.module)
    else:
        return module.parameters()