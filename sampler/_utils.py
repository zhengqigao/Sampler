from typing import TypeVar, Callable, Union, Optional


## TODO: define an alias decorator, so maybe we can call ais or annealed_importance_sampling for the same function?
def _alias(*aliases):
    def decorator(func):
        for alias in aliases:
            globals()[alias] = func
        return func
    return decorator
