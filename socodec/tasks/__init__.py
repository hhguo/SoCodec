from os.path import dirname

import torch

from ..utils.config import Config
from ..utils.utils import module_search


def build_task(config, mode='train'):
    # Check type of config
    if isinstance(config, str):
        config = Config(config)

    # Init Task
    config = config.copy()
    task_name = config.pop('_name')
    modules = config.pop('network')
    TaskClass = module_search(task_name, dirname(__file__), 'socodec.tasks')
    task = TaskClass(modules, mode=mode, **config)

    return task