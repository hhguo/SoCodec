from safetensors import safe_open

import contextlib
import functools
import glob
import importlib
import io
import inspect
import numpy as np
import os
import re
import torch


def to_model(x, device=None):
    if isinstance(x, (tuple, list)):
        x = [to_model(x) for x in x]
    elif isinstance(x, dict):
        x = {k: to_model(v) for k, v in x.items()}
    elif x is not None:
        x = to_gpu(torch.as_tensor(x)) if device is None else torch.as_tensor(x).to(device)
    return x

          
def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def get_mask_from_lengths(lengths, max_len=None):
    max_len = torch.max(lengths).item() if max_len is None else max_len
    ids = torch.arange(0, max_len).to(lengths.device)
    mask = ~(ids < lengths.unsqueeze(1)).bool()
    return mask


'''
Checkpoint Load & Save
'''
def load_checkpoint(model, checkpoint_object, strict=True):
    # Load checkpoint file
    if isinstance(checkpoint_object, dict):
        path = checkpoint_object.get('path', None)
        prefix = checkpoint_object.get('prefix', None)
        strict = checkpoint_object.get('strict', True)
    else:
        assert os.path.isfile(checkpoint_object)
        path, prefix = checkpoint_object, None

    ext = path.rsplit('.')[-1]
    if ext in ['pt', 'bin']:
        ckpt_dict = torch.load(path, map_location='cpu')
    elif ext == 'safetensors':
        with safe_open(path, framework="pt", device='cpu') as f:
            ckpt_dict = {k: f.get_tensor(k) for k in f.keys()}
    else:
        raise TypeError(f"Wrong file extension: {ext}")

    # Load model parameters
    if prefix:
        unwrapped_ckpt_dict = {}
        for key, value in ckpt_dict.items():
            head, tail = key.split('.', 1)
            if head == prefix:
                unwrapped_ckpt_dict[tail] = value
        ckpt_dict = unwrapped_ckpt_dict

    distributed_data_parallel = False
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        distributed_data_parallel = True
        dist_model, model = model, model.module

    try:
        model.load_state_dict(ckpt_dict, strict=True)
    except:
        model_dict = model.state_dict()

        size_mismatches = []
        for k, v in model_dict.items():
            if k in ckpt_dict and v.shape != ckpt_dict[k].shape:
                size_mismatches.append(k)
                ckpt_dict.pop(k)
        missing, unexpected = model.load_state_dict(ckpt_dict, strict=False)

        if missing or unexpected:
            missing_keys = ", ".join([f'"{k}"' for k in sorted(missing)])
            unexpected_keys = ", ".join([f'"{k}"' for k in sorted(unexpected)])
            error = f"Error(s) in loading state_dict for {model.__class__.__name__}:"
            if missing:
                error += f"\n    Missing key(s) in state_dict: {missing_keys}. You may ignore them if they are covered in safetensors."
            if unexpected:
                error += f"\n    Unexpected key(s) in state_dict: {unexpected_keys}"
            if len(size_mismatches) > 0:
                error += f"\n    Mismatched key(s) in state_dict: {size_mismatches}"
            if strict:
                raise RuntimeError(error)
            else:
                print(error)

    if distributed_data_parallel:
        dist_model.module = model
        model = dist_model

    print(f"Checkpoint ({path}) loading is completed.")
    return model


'''
Module Search
'''
def module_search(names, directory, package=None):
    anchors = [names] if isinstance(names, str) else names
    
    filepaths = glob.glob(os.path.join(directory, "*.py")) + \
                glob.glob(os.path.join(directory, "*", "__init__.py"))
    filepaths = [x[len(directory):] for x in filepaths]
    module_files = [x[: -3].replace(os.path.sep, '.').replace('.__init__', '')
                    for x in filepaths]
    module_files = [x for x in module_files if len(x) > 0]
    
    selected_modules = [None] * len(anchors)

    for i, name in enumerate(anchors):
        class_name = name.split('.')[-1]
        directory = name[: -len(class_name) - 1]
        search_space = module_files
        if directory != '':
            search_space = [package + '.' + directory]
        for module_file in search_space:
            modules = importlib.import_module(module_file, package=package)
            if not hasattr(modules, class_name):
                continue
            module = getattr(modules, class_name)
            path = inspect.getfile(module)

            if selected_modules[i] is not None:
                found_path = inspect.getfile(selected_modules[i])
                if found_path != path:
                    raise RuntimeError("Repeated Module for {}: {}, {}".format(
                        class_name, found_path, path))
                continue
            print('Load {} from file "{}"'.format(class_name, path))
            
            selected_modules[i] = module

    if None in selected_modules:
        raise RuntimeError('Cannot find modules for {}'.format(names))
    
    if isinstance(names, str):
        selected_modules = selected_modules[0]
    return selected_modules


def get_module(class_name, package=None):
    assert isinstance(class_name, str)
    if not package:
        package, class_name = class_name.rsplit('.', 1)
    target_class = getattr(
        importlib.import_module(package),
        class_name
    )
    return target_class


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))