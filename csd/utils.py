import copy
import gc
from functools import wraps
from time import time
from typing import Callable, Dict, List, Tuple, TypeVar

import ray
import torch

T = TypeVar("T")


def lmap(func: Callable, x: List[T]) -> List[T]:
    return list(map(func, x))


def clean_gpu(x):
    del x
    torch.cuda.empty_cache()
    gc.collect()


def get_model_device(model: torch.nn.Module) -> torch.device:
    return list(model.state_dict().values())[0].device


def performance(f: T) -> T:
    @wraps(f)
    def wrapper(*args, **kwargs):
        device_type = get_model_device(args[0]).type
        return torch.autocast(device_type)(torch.no_grad()(f))(*args, **kwargs)

    return wrapper


def extract_tensors(m: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays, and return the stripped model and tensors.
    """
    tensors = []
    for _, module in m.named_modules():
        # Store the tensors in Python dictionaries
        params = {
            name: torch.clone(param).cpu().detach().numpy() for name, param in module.named_parameters(recurse=False)
        }
        buffers = {name: torch.clone(buf).cpu().detach().numpy() for name, buf in module.named_buffers(recurse=False)}
        tensors.append({"params": params, "buffers": buffers})

    # Make a copy of the original model and strip all tensors and
    # buffers out of the copy.
    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in [name for name, _ in module.named_parameters(recurse=False)] + [
            name for name, _ in module.named_buffers(recurse=False)
        ]:
            setattr(module, name, None)

    # Make sure the copy is configured for inference.
    m_copy.train(False)
    return m_copy, tensors


def replace_tensors(m: torch.nn.Module, tensors: List[Dict], device="cuda"):
    """
    Restore the tensors that extract_tensors() stripped out of a
    PyTorch model.
    :param no_parameters_objects: Skip wrapping tensors in
    ``torch.nn.Parameters`` objects (~20% speedup, may impact
    some models)
    """
    modules = [module for _, module in m.named_modules()]
    for module, tensor_dict in zip(modules, tensors):
        # There are separate APIs to set parameters and buffers.
        for name, array in tensor_dict["params"].items():
            module.register_parameter(
                name,
                torch.nn.Parameter(torch.as_tensor(array, device=device), requires_grad=False),
            )
        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(name, torch.as_tensor(array, device=device))


def push_model_to_plasma(model: torch.nn.Module) -> ray.ObjectRef:

    ref = ray.put(extract_tensors(model))
    clean_gpu(model)

    return ref


def load_from_plasma(ref, device="cuda"):
    skeleton, weights = ray.get(ref)
    replace_tensors(skeleton, weights, device=device)

    skeleton.eval().half().to(device)

    return skeleton
